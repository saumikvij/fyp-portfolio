"""EXPERIMENT (branch experiment/model-search): search over models, windows, targets and
signals, then feed the three finalist models into Black–Litterman. Design: DECISIONS MS1–MS3.

Run from the repo root:
    python -m notebooks.model_search                 # stage 1: search, development only
    python -m notebooks.model_search --stage final   # stage 2: finalists on the holdout (once)

Every setup is trained with the Phase 3 walk-forward (train on data up to each
month-end, forecast the next month). Setups are compared ONLY on forecasts made up to
``model_search.dev_end``; the holdout (later forecasts) is evaluated for the finalists
only. Outputs go to report/experiments/ and data/ml/search/ (so nothing in the main
report is touched). Re-running skips setups whose forecasts are already saved.
"""

from __future__ import annotations

import argparse
import time
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

from src.backtest import (black_litterman_strategy, equal_weight, markowitz_strategy,
                          rebalance_dates, run_backtest)
from src.data import load_config, load_processed
from src.features import FEATURE_SETS, add_targets, build_feature_panel, rank_normalize
from src.metrics import max_drawdown, performance_summary
from src.ml_eval import quantile_spread, r2_oos, rank_ic, summarize_series
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import HuberRegressor
from sklearn.neighbors import KNeighborsRegressor

from src.models import (ElasticNetModel, GridModel, HistoricalMeanModel, LightGBMModel,
                        MLPModel, RandomForestModel, walk_forward)

OUT = Path("report/experiments")


def build_model(family: str, cfg: dict, features: list[str]):
    ml, seed, every = cfg["ml"], cfg["seed"], cfg["ml"]["retrain_every_months"]
    if family == "Elastic net":
        e = ml["elastic_net"]
        return ElasticNetModel(e["alphas"], e["l1_ratios"], e["max_iter"], seed,
                               retrain_every=every, features=features)
    if family == "LightGBM":
        g = ml["lightgbm"]
        return LightGBMModel(g["num_leaves"], g["min_child_samples"], g["learning_rate"],
                             g["max_trees"], g["early_stopping_rounds"], g["feature_fraction"],
                             g["bagging_fraction"], g["bagging_freq"], g["lambda_l2"], seed,
                             retrain_every=every, features=features)
    if family == "MLP":
        m = ml["mlp"]
        return MLPModel(m["hidden_layers"], m["alphas"], m["learning_rate_init"], m["batch_size"],
                        m["max_epochs"], m["patience"], seed, retrain_every=every,
                        features=features)
    if family == "Random forest":
        r = cfg["model_search"]["random_forest"]
        return RandomForestModel(r["n_estimators"], r["max_depth"], r["min_samples_leaf"],
                                 r["max_features"], seed, retrain_every=every, features=features)
    if family == "Huber":
        h = cfg["model_search"]["huber"]
        return GridModel("Huber", lambda epsilon, alpha, seed: HuberRegressor(
            epsilon=epsilon, alpha=alpha, max_iter=2000),
            {"epsilon": h["epsilon"], "alpha": h["alpha"]}, seed, every, features)
    if family == "Extra trees":
        x = cfg["model_search"]["extra_trees"]
        return GridModel("Extra trees", lambda max_depth, min_samples_leaf, seed:
                         ExtraTreesRegressor(n_estimators=x["n_estimators"], max_depth=max_depth,
                                             min_samples_leaf=min_samples_leaf,
                                             max_features=x["max_features"], random_state=seed,
                                             n_jobs=-1),
                         {"max_depth": x["max_depth"], "min_samples_leaf": x["min_samples_leaf"]},
                         seed, every, features)
    if family == "KNN":
        k = cfg["model_search"]["knn"]
        return GridModel("KNN", lambda n_neighbors, seed: KNeighborsRegressor(
            n_neighbors=n_neighbors), {"n_neighbors": k["n_neighbors"]}, seed, every, features)
    raise ValueError(family)


def setup_id(family, window, target, fset) -> str:
    w = "expanding" if window is None else f"roll{window}m"
    return f"{family.lower().replace(' ', '_')}__{w}__{target}__{fset}"


def forecast_metrics(f, realized, hm, q) -> dict:
    ic = rank_ic(f, realized)
    s = summarize_series(ic) if ic.notna().any() else {"mean": np.nan, "t_stat_nw": np.nan}
    qs = summarize_series(quantile_spread(f, realized, q)["spread"])
    return {"ic_mean": s["mean"], "ic_t_nw": s["t_stat_nw"],
            "r2_vs_zero": r2_oos(f, realized), "r2_vs_hist_mean": r2_oos(f, realized, hm),
            "q_spread": qs["mean"], "months_flat": int(ic.isna().sum())}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["search", "final"], default="search")
    stage = ap.parse_args().stage
    cfg = load_config()
    ms, ml, bt = cfg["model_search"], cfg["ml"], cfg["backtest"]
    OUT.mkdir(parents=True, exist_ok=True)
    fc_dir = Path(ms["out_dir"])
    fc_dir.mkdir(parents=True, exist_ok=True)
    dev_end = pd.Timestamp(ms["dev_end"])

    data = load_processed(cfg)
    stocks = list(cfg["universe"])
    dates = rebalance_dates(data["returns_monthly"].index, bt["oos_start"])
    dev, hold = dates[dates <= dev_end], dates[dates > dev_end]
    panels = {fs: add_targets(rank_normalize(build_feature_panel(
        data, stocks, cfg["data"]["benchmark"], ml["windows"], ml["rsi_days"], fs)),
        data["returns_monthly"][stocks], data["rf_monthly"]["rf"]) for fs in ms["feature_sets"]}
    realized = panels["base"]["target"].unstack().loc[dates]
    hm = walk_forward(HistoricalMeanModel(), panels["base"], dates, ml["val_months"],
                      ml["min_train_months"]).forecasts
    print(f"Development: forecasts {dev[0]:%b %Y} – {dev[-1]:%b %Y} ({len(dev)}); "
          f"holdout: {hold[0]:%b %Y} – {hold[-1]:%b %Y} ({len(hold)})")

    # ------------------------------------------------------------ search (dev only)
    rows = []
    grid = list(product(ms["families"], ms["train_windows"], ms["targets"], ms["feature_sets"]))
    for i, (fam, win, tgt, fs) in enumerate(grid, 1):
        sid = setup_id(fam, win, tgt, fs)
        path = fc_dir / f"{sid}.parquet"
        t0 = time.time()
        if path.exists():
            f = pd.read_parquet(path)
        elif stage == "final":
            raise SystemExit(f"{sid} has not been searched yet; run stage 1 first.")
        else:
            res = walk_forward(build_model(fam, cfg, FEATURE_SETS[fs]), panels[fs], dates,
                               ml["val_months"], ml["min_train_months"],
                               train_window_months=win, target_mode=tgt)
            f = res.forecasts
            f.to_parquet(path)
        m = forecast_metrics(f.loc[dev], realized.loc[dev], hm.loc[dev], ml["quantiles"])
        rows.append({"setup": sid, "family": fam, "window": win or "expanding", "target": tgt,
                     "features": fs, **{f"dev_{k}": v for k, v in m.items()}})
        print(f"[{i:2d}/{len(grid)}] {sid:48s} dev IC {m['ic_mean']:+.4f} "
              f"(NW t {m['ic_t_nw']:+.2f})  {time.time() - t0:4.0f}s")
    log = pd.DataFrame(rows).sort_values("dev_ic_mean", ascending=False)
    log.to_csv(OUT / "search_log.csv", index=False)

    # ------------------------------------------------------------ finalists (rule MS3)
    best = log.groupby("family", sort=False).head(1)              # best setup per family
    finalists = best.sort_values("dev_ic_mean", ascending=False).head(ms["n_finalists"])
    print("\nBest setup per family (development score):")
    print(best[["family", "setup", "dev_ic_mean", "dev_ic_t_nw"]].to_string(index=False))
    print(f"\nFinalists by rule MS3: {', '.join(finalists['family'])}")
    if stage == "search":
        print("Stage 1 done. Holdout not touched; run with --stage final once, at the end.")
        return

    # ------------------------------------------------------------ holdout + Black–Litterman
    g = cfg["risk_tolerance"][cfg["black_litterman"]["headline"]["gamma"]]
    tau, c = cfg["black_litterman"]["tau"], bt["cost_bps"]
    rf = data["rf_monthly"]["rf"]
    wt = cfg["optimization"]["weight_tol"]
    original = pd.read_parquet(Path(cfg["phase3"]["forecast_dir"]) / "forecasts_elastic_net.parquet")
    strategies = {"S1 Equal weight": equal_weight(stocks),
                  f"S4 Markowitz γ={g}": markowitz_strategy(stocks, cfg, g),
                  "BL prior only": black_litterman_strategy(original, None, stocks, cfg, g, "A",
                                                            tau, views=False),
                  "S6 original (elastic net, Phase 4)": black_litterman_strategy(
                      original, None, stocks, cfg, g, "A", tau)}
    fc_final, rows = {}, []
    for _, r in finalists.iterrows():
        f = pd.read_parquet(fc_dir / f"{r['setup']}.parquet")
        fc_final[r["setup"]] = f
        mh = forecast_metrics(f.loc[hold], realized.loc[hold], hm.loc[hold], ml["quantiles"])
        rows.append({"setup": r["setup"], "dev_ic": r["dev_ic_mean"], "dev_ic_t": r["dev_ic_t_nw"],
                     "hold_ic": mh["ic_mean"], "hold_ic_t": mh["ic_t_nw"],
                     "hold_r2_vs_hist_mean": mh["r2_vs_hist_mean"]})
        strategies[f"S6 {r['family']} (search)"] = black_litterman_strategy(
            f, None, stocks, cfg, g, "A", tau)
    fin = pd.DataFrame(rows)
    fin.to_csv(OUT / "finalists_forecasts.csv", index=False)
    print("\nFinalists, forecast quality (development vs holdout, NW t in brackets):")
    print(fin.round(4).to_string(index=False))

    split = data["returns_monthly"].index[data["returns_monthly"].index > dev_end][0]
    port = {}
    for name, strat in strategies.items():
        res = run_backtest(name, strat, data, dates, wt)
        net = res.net(c)
        row = {}
        for label, seg in (("dev", net.loc[:split]), ("hold", net.loc[split:].iloc[1:])):
            p = performance_summary(seg, rf)
            row.update({f"{label}_ann_return": p["ann_return"], f"{label}_sharpe": p["sharpe"],
                        f"{label}_max_dd": max_drawdown(seg)})
        row["avg_turnover"] = float(res.turnover.iloc[1:].mean())
        port[name] = row
    port = pd.DataFrame(port).T
    port.to_csv(OUT / "finalists_portfolios.csv")
    print(f"\nPortfolios (γ={g}, {c} bps), development vs holdout:")
    print(port.round(3).to_string())
    print(f"\nOutputs in {OUT}/")


if __name__ == "__main__":
    main()

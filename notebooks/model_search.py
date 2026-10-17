"""EXPERIMENT (branch experiment/model-search): search over models, windows, targets and
signals, then feed the three finalist models into Black–Litterman. Design: DECISIONS MS1–MS3.

Run from the repo root:
    python -m notebooks.model_search                 # stage 1: search, development only
    python -m notebooks.model_search --stage final   # stage 2: finalists on the holdout (once)
    python -m notebooks.model_search --stage placebo # luck check: same search, shuffled targets

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
from src.experiment_models import ClassifierModel, shuffle_targets
from src.experiment_signals import NICHE_SET, build_niche_panel, rank_normalize_niche
from src.features import FEATURE_SETS, add_targets, build_feature_panel, rank_normalize
from src.metrics import max_drawdown, performance_summary
from src.ml_eval import quantile_spread, r2_oos, rank_ic, summarize_series
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.kernel_approximation import Nystroem
from sklearn.linear_model import LogisticRegression, QuantileRegressor, Ridge
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVR
import lightgbm as lgb
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import HuberRegressor
from sklearn.neighbors import KNeighborsRegressor

from src.models import (ElasticNetModel, GridModel, HistoricalMeanModel, LightGBMModel,
                        MLPModel, RandomForestModel, walk_forward)

OUT = Path("report/experiments")


def build_wide(family: str, cfg: dict, features: list[str], every: int):
    """Round 4 (MS8): the same families with the wider grids in ``model_search.round4.wide``."""
    w, seed, ml = cfg["model_search"]["round4"]["wide"][family], cfg["seed"], cfg["ml"]
    if family == "KNN":
        return GridModel("KNN", lambda n_neighbors, weights, seed: KNeighborsRegressor(
            n_neighbors=n_neighbors, weights=weights), w, seed, every, features)
    if family == "MLP":
        m = ml["mlp"]
        return MLPModel(w["hidden_layers"], w["alphas"], m["learning_rate_init"], m["batch_size"],
                        m["max_epochs"], m["patience"], seed, retrain_every=every,
                        features=features)
    if family == "LightGBM":
        g = ml["lightgbm"]
        return LightGBMModel(w["num_leaves"], w["min_child_samples"], g["learning_rate"],
                             g["max_trees"], g["early_stopping_rounds"], g["feature_fraction"],
                             g["bagging_fraction"], g["bagging_freq"], g["lambda_l2"], seed,
                             retrain_every=every, features=features)
    if family == "Hist gradient boosting":
        h = cfg["model_search"]["hist_gradient_boosting"]
        return GridModel("Hist gradient boosting", lambda max_depth, learning_rate,
                         min_samples_leaf, seed: HistGradientBoostingRegressor(
                             max_depth=max_depth, learning_rate=learning_rate,
                             max_iter=h["max_iter"], min_samples_leaf=min_samples_leaf,
                             random_state=seed), w, seed, every, features)
    if family == "Kernel ridge":
        k = cfg["model_search"]["kernel_ridge"]
        return GridModel("Kernel ridge", lambda gamma, alpha, seed: make_pipeline(
            Nystroem(gamma=gamma, n_components=k["n_components"], random_state=seed),
            Ridge(alpha=alpha)), w, seed, every, features)
    raise ValueError(family)


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
        m = RandomForestModel(r["n_estimators"], r["max_depth"], r["min_samples_leaf"],
                              r["max_features"], seed, retrain_every=every, features=features)
        m.max_samples = r.get("max_samples")
        return m
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
                                             max_features=x["max_features"],
                                             bootstrap=x.get("max_samples") is not None,
                                             max_samples=x.get("max_samples"), random_state=seed,
                                             n_jobs=-1),
                         {"max_depth": x["max_depth"], "min_samples_leaf": x["min_samples_leaf"]},
                         seed, every, features)
    if family == "KNN":
        k = cfg["model_search"]["knn"]
        return GridModel("KNN", lambda n_neighbors, seed: KNeighborsRegressor(
            n_neighbors=n_neighbors), {"n_neighbors": k["n_neighbors"]}, seed, every, features)
    if family == "PCA regression":
        p = cfg["model_search"]["pca_regression"]
        return GridModel("PCA regression", lambda n_components, alpha, seed: make_pipeline(
            PCA(n_components=n_components, random_state=seed), Ridge(alpha=alpha)),
            {"n_components": p["n_components"], "alpha": p["alpha"]}, seed, every, features)
    if family == "Hist gradient boosting":
        h = cfg["model_search"]["hist_gradient_boosting"]
        return GridModel("Hist gradient boosting", lambda max_depth, learning_rate, seed:
                         HistGradientBoostingRegressor(max_depth=max_depth,
                                                       learning_rate=learning_rate,
                                                       max_iter=h["max_iter"],
                                                       min_samples_leaf=h["min_samples_leaf"],
                                                       random_state=seed),
                         {"max_depth": h["max_depth"], "learning_rate": h["learning_rate"]},
                         seed, every, features)
    if family == "Kernel ridge":
        k = cfg["model_search"]["kernel_ridge"]
        return GridModel("Kernel ridge", lambda gamma, alpha, seed: make_pipeline(
            Nystroem(gamma=gamma, n_components=k["n_components"], random_state=seed),
            Ridge(alpha=alpha)), {"gamma": k["gamma"], "alpha": k["alpha"]}, seed, every,
            features)
    if family == "Logistic classifier":
        c = cfg["model_search"]["logistic_classifier"]
        return ClassifierModel("Logistic classifier", lambda C, seed: LogisticRegression(
            C=C, max_iter=2000), {"C": c["C"]}, seed, every, features)
    if family == "LightGBM classifier":
        c = cfg["model_search"]["lightgbm_classifier"]
        return ClassifierModel("LightGBM classifier", lambda num_leaves, n_estimators, seed:
                               lgb.LGBMClassifier(num_leaves=num_leaves, n_estimators=n_estimators,
                                                  learning_rate=0.05, min_child_samples=100,
                                                  subsample=0.8, subsample_freq=1,
                                                  colsample_bytree=0.8, random_state=seed,
                                                  deterministic=True, force_row_wise=True,
                                                  n_jobs=1, verbose=-1),
                               {"num_leaves": c["num_leaves"], "n_estimators": c["n_estimators"]},
                               seed, every, features)
    if family == "Quantile regression":
        q = cfg["model_search"]["quantile_regression"]
        return GridModel("Quantile regression", lambda alpha, seed: QuantileRegressor(
            quantile=0.5, alpha=alpha, solver="highs"), {"alpha": q["alpha"]}, seed, every,
            features)
    if family == "Naive Bayes classifier":
        return ClassifierModel("Naive Bayes classifier", lambda var_smoothing, seed: GaussianNB(
            var_smoothing=var_smoothing), {"var_smoothing": [1e-9]}, seed, every, features)
    if family == "Linear SVR":
        c = cfg["model_search"]["linear_svr"]
        return GridModel("Linear SVR", lambda C, seed: LinearSVR(
            C=C, epsilon=0.0, max_iter=20000, random_state=seed), {"C": c["C"]}, seed, every,
            features)
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
    ap.add_argument("--stage", choices=["search", "final", "placebo"], default="search")
    stage = ap.parse_args().stage
    cfg = load_config()
    ms, ml, bt = cfg["model_search"], cfg["ml"], cfg["backtest"]
    OUT.mkdir(parents=True, exist_ok=True)
    fc_dir = Path(ms["out_dir"])
    fc_dir.mkdir(parents=True, exist_ok=True)
    dev_end = pd.Timestamp(ms["dev_end"])
    cuts = [pd.Timestamp(c) for c in ms["dev_blocks"]]

    data = load_processed(cfg)
    stocks = list(cfg["universe"])
    dates = rebalance_dates(data["returns_monthly"].index, bt["oos_start"])
    dev, hold = dates[dates <= dev_end], dates[dates > dev_end]
    def make_panel(fs):
        if fs == "niche":
            raw = rank_normalize_niche(build_niche_panel(
                data, stocks, cfg["data"]["benchmark"], ml["windows"], ml["rsi_days"],
                Path(cfg["data"]["raw_dir"]) / "prices_raw.parquet"))
        else:
            raw = rank_normalize(build_feature_panel(data, stocks, cfg["data"]["benchmark"],
                                                     ml["windows"], ml["rsi_days"], fs))
        return add_targets(raw, data["returns_monthly"][stocks], data["rf_monthly"]["rf"])

    panels = {fs: make_panel(fs) for fs in ms["feature_sets"]}
    if stage == "placebo":
        # Same data and code, but each month's targets shuffled across stocks: any
        # "skill" found here is luck (MS7). Kept apart from the real search results.
        panels = {fs: shuffle_targets(p, ms["placebo"]["seed"]) for fs, p in panels.items()}
        fc_dir = Path(ms["out_dir"] + "_placebo")
        fc_dir.mkdir(parents=True, exist_ok=True)
    feature_lists = {**FEATURE_SETS, "niche": NICHE_SET}
    r4 = ms.get("round4")
    if r4 and stage != "placebo":
        feature_lists["risk"] = r4["risk_set"]
        panels["risk"] = panels["niche"]          # same rows; the model reads only the risk set
    realized = panels["base"]["target"].unstack().loc[dates]   # shuffled too in placebo
    hm = walk_forward(HistoricalMeanModel(), panels["base"], dates, ml["val_months"],
                      ml["min_train_months"]).forecasts
    print(f"Development: forecasts {dev[0]:%b %Y} – {dev[-1]:%b %Y} ({len(dev)}); "
          f"holdout: {hold[0]:%b %Y} – {hold[-1]:%b %Y} ({len(hold)})")

    def score(sid, fam, win, tgt, fs, f) -> dict:
        """Development-period scores only (overall and per block, MS3 + MS5)."""
        m = forecast_metrics(f.loc[dev], realized.loc[dev], hm.loc[dev], ml["quantiles"])
        ic = rank_ic(f.loc[dev], realized.loc[dev])
        block_ic = [ic[(ic.index > a) & (ic.index <= b)].mean()
                    for a, b in zip([pd.Timestamp.min] + cuts, cuts + [dev_end])]
        return {"setup": sid, "family": fam, "window": win or "expanding", "target": tgt,
                "features": fs, **{f"dev_{k}": v for k, v in m.items()},
                **{f"block{j + 1}_ic": v for j, v in enumerate(block_ic)},
                "consistent": bool(all(np.nan_to_num(v, nan=-1) > 0 for v in block_ic))}

    # ------------------------------------------------------------ search (dev only)
    rows = []
    fams = ms["placebo"]["families"] if stage == "placebo" else ms["families"]
    grid = list(product(fams, ms["train_windows"], ms["targets"], ms["feature_sets"]))
    for i, (fam, win, tgt, fs) in enumerate(grid, 1):
        sid = setup_id(fam, win, tgt, fs)
        path = fc_dir / f"{sid}.parquet"
        t0 = time.time()
        if path.exists():
            f = pd.read_parquet(path)
        elif stage == "final":
            raise SystemExit(f"{sid} has not been searched yet; run stage 1 first.")
        else:
            res = walk_forward(build_model(fam, cfg, feature_lists[fs]), panels[fs], dates,
                               ml["val_months"], ml["min_train_months"],
                               train_window_months=win, target_mode=tgt)
            f = res.forecasts
            f.to_parquet(path)
        rows.append(score(sid, fam, win, tgt, fs, f))
        print(f"[{i:3d}/{len(grid)}] {sid:56s} done  {time.time() - t0:4.0f}s")
    if r4 and stage != "placebo":
        grid4 = list(product(r4["families"], ms["train_windows"], ms["targets"],
                             r4["feature_sets"], r4["retrain_every"]))
        for i, (fam, win, tgt, fs, every) in enumerate(grid4, 1):
            sid = setup_id(fam, win, tgt, fs) + f"__wide_r{every}"
            path = fc_dir / f"{sid}.parquet"
            t0 = time.time()
            if path.exists():
                f = pd.read_parquet(path)
            elif stage == "final":
                raise SystemExit(f"{sid} has not been searched yet; run stage 1 first.")
            else:
                f = walk_forward(build_wide(fam, cfg, feature_lists[fs], every), panels[fs],
                                 dates, ml["val_months"], ml["min_train_months"],
                                 train_window_months=win, target_mode=tgt).forecasts
                f.to_parquet(path)
            rows.append(score(sid, fam, win, tgt, fs, f))
            print(f"[R4 {i:3d}/{len(grid4)}] {sid:66s} done  {time.time() - t0:4.0f}s")

    if ms.get("combine") and stage != "placebo":
        for win, tgt, fs in product(ms["train_windows"], ms["targets"], ms["feature_sets"]):
            members = [pd.read_parquet(fc_dir / f"{setup_id(fam, win, tgt, fs)}.parquet")
                       for fam in ms["families"]]
            f = sum(members) / len(members)
            sid = setup_id("Average of all", win, tgt, fs)
            f.to_parquet(fc_dir / f"{sid}.parquet")
            rows.append(score(sid, "Average of all", win, tgt, fs, f))
    log = pd.DataFrame(rows).sort_values("dev_ic_mean", ascending=False)
    if stage == "placebo":
        log.to_csv(OUT / "placebo_log.csv", index=False)
        print(f"\nPLACEBO ({len(log)} setups, shuffled targets): best development IC "
              f"{log['dev_ic_mean'].max():+.4f}; 95th percentile {log['dev_ic_mean'].quantile(.95):+.4f}; "
              f"positive in all blocks: {int(log['consistent'].sum())} of {len(log)} "
              f"({log['consistent'].mean():.0%}; pure luck ≈ 12.5%)")
        return
    log.to_csv(OUT / "search_log.csv", index=False)

    # ------------------------------------------------------------ finalists (rule MS3)
    # Rule MS5: only setups with positive IC in every development block qualify; the best
    # qualifying setup per family goes forward, ranked by development IC.
    qualified = log[log["consistent"]]
    best = qualified.groupby("family", sort=False).head(1)
    finalists = best.sort_values("dev_ic_mean", ascending=False).head(ms["n_finalists"])
    print(f"\nSetups positive in all {len(cuts) + 1} development blocks: "
          f"{len(qualified)} of {len(log)}")
    if len(finalists) < ms["n_finalists"]:
        print(f"Only {len(finalists)} famil{'y' if len(finalists) == 1 else 'ies'} "
              "have a consistent setup.")
    print("\nBest consistent setup per family (development score, by block):")
    print(best[["family", "setup", "dev_ic_mean", "block1_ic", "block2_ic", "block3_ic"]]
          .round(4).to_string(index=False))
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
        ic_h = rank_ic(f.loc[hold], realized.loc[hold])
        rows.append({"setup": r["setup"], "dev_ic": r["dev_ic_mean"], "dev_ic_t": r["dev_ic_t_nw"],
                     "hold_ic": mh["ic_mean"], "hold_ic_t": mh["ic_t_nw"],
                     "hold_r2_vs_hist_mean": mh["r2_vs_hist_mean"],
                     **{f"hold_ic_{y}": v for y, v in ic_h.groupby(ic_h.index.year).mean().items()}})
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

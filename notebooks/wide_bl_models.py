"""EXPERIMENT (branch experiment/model-search): Black–Litterman with every model's views on
the wide universe, and whether up/down models help (DECISIONS WU5).

Run:  python -m notebooks.wide_bl_models

Universe: all wide-universe stocks with a share count (~1,047). Same portfolio settings
as WU4: Ledoit–Wolf Σ, 2% cap per stock, γ = 5, τ = 0.05, Ω-A, 10 bps.

Views: for each of the 15 model families, the setup with the best development rank IC
in the wide search (development data only, rule fixed before running); the average of
those 15; and one extra up/down model, a logistic classifier of "excess return > 0"
(``ClassifierModel(label="sign")``), trained like every other model. The up/down group
is the three "beats the median" classifiers plus this one.

Reports, development (2016–21) and holdout (2022–26, post-holdout) separately:
portfolio return / volatility / Sharpe / drawdown / CVaR / turnover, up/down accuracy of
each model's views, and paired Sharpe tests (block bootstrap, Holm) of every BL + views
portfolio against BL prior only. Each backtest is saved, so a re-run skips finished ones.
"""

from __future__ import annotations

import copy
import time
from pathlib import Path

import numpy as np
import pandas as pd

from notebooks.model_search import build_model, setup_id
from src.backtest import (black_litterman_strategy, buy_and_hold, equal_weight,
                          rebalance_dates, run_backtest)
from src.data import load_config
from src.experiment_signals import NICHE_SET
from src.experiment_universe import WIDE, load_wide
from src.features import FEATURE_SETS
from src.metrics import (cvar_historical, holm_adjust, max_drawdown, performance_summary,
                         sharpe_difference_bootstrap)
from src.models import walk_forward

OUT = Path("report/experiments")
UPDOWN = ["Logistic classifier", "LightGBM classifier", "Naive Bayes classifier",
          "Logistic (up vs down)"]


def chosen_views(cfg: dict, dates) -> dict[str, pd.DataFrame]:
    """Forecasts used as views: best development-IC setup per family, their average, and
    the plain up/down logistic classifier (trained here if not saved yet)."""
    ms = cfg["model_search"]
    fc_dir = Path(ms["wide"]["out_dir"])
    log = pd.read_csv(OUT / "wide_search_log.csv")
    best = log.sort_values("ic_all", ascending=False).groupby("family").head(1)
    views = {r.family: pd.read_parquet(fc_dir / f"{r.setup}.parquet") for r in best.itertuples()}
    views["Average of 15 models"] = sum(views.values()) / len(views)

    fs = best.set_index("family").loc["Logistic classifier", "features"]
    sid = setup_id("Logistic classifier", None, "raw", fs).replace("logistic_classifier",
                                                                   "logistic_updown")
    path = fc_dir / f"{sid}.parquet"
    if not path.exists():
        t0 = time.time()
        model = build_model("Logistic classifier", cfg, {**FEATURE_SETS, "niche": NICHE_SET}[fs])
        model.label = "sign"
        panel = pd.read_parquet(WIDE / f"panel_{fs}.parquet")
        f = walk_forward(model, panel, dates, cfg["ml"]["val_months"],
                         cfg["ml"]["min_train_months"]).forecasts
        f.to_parquet(path)
        print(f"Trained the up/down logistic classifier ({sid}) in {time.time() - t0:.0f}s")
    views["Logistic (up vs down)"] = pd.read_parquet(path)
    print("Views used (best development IC per family):")
    for r in best.itertuples():
        print(f"  {r.family:24s} {r.setup}")
    return views


def accuracy(f: pd.DataFrame, realized: pd.DataFrame) -> dict[str, float]:
    """Up/down accuracy of views: absolute (forecast > 0 vs realized > 0) and relative
    (forecast above the month's median forecast vs realized above the month's median)."""
    f, realized = f.reindex_like(realized), realized
    ok = realized.notna() & f.notna()
    absolute = ((f > 0) == (realized > 0))[ok].stack().mean()
    rel = ((f.gt(f.median(axis=1), axis=0)) == (realized.gt(realized.median(axis=1), axis=0)))
    return {"acc_up_down": float(absolute), "acc_vs_median": float(rel[ok].stack().mean())}


def main() -> None:
    cfg = load_config()
    ms, bt = cfg["model_search"], cfg["backtest"]
    port = ms["wide"]["portfolio"]
    wide = load_wide()
    uni = wide["universe"]
    stocks = [s for s in uni.index if s in wide["shares_outstanding"].index]
    dates = rebalance_dates(wide["returns_monthly"].index, bt["oos_start"])
    dev_end = pd.Timestamp(ms["dev_end"])
    split = wide["returns_monthly"].index[wide["returns_monthly"].index > dev_end][0]
    rf = wide["rf_monthly"]["rf"]
    g, tau = cfg["risk_tolerance"]["medium"], cfg["black_litterman"]["tau"]
    c = copy.deepcopy(cfg)
    c["backtest"].update(cov_method=port["cov_method"], w_max=port["w_max"])
    c["universe"] = {s: uni.loc[s, "sector"] for s in stocks}

    views = chosen_views(cfg, dates)
    realized = pd.read_parquet(WIDE / "panel_base.parquet")["target"].unstack()
    realized = realized.reindex(index=dates, columns=stocks)

    strategies = {"1/N": equal_weight(stocks), "SPY": buy_and_hold(cfg["data"]["benchmark"]),
                  "BL prior only": black_litterman_strategy(views["Elastic net"], None, stocks,
                                                            c, g, "A", tau, views=False)}
    for name, f in views.items():
        strategies[f"BL + {name}"] = black_litterman_strategy(f.reindex(columns=stocks), None,
                                                              stocks, c, g, "A", tau)
    cache = Path(ms["wide"]["out_dir"]) / "bl_runs"
    cache.mkdir(exist_ok=True)
    nets, rows = {}, []
    for i, (name, strat) in enumerate(strategies.items(), 1):
        path = cache / f"{name.replace(' ', '_').replace('/', '-')}.parquet"
        t0 = time.time()
        if path.exists():
            saved = pd.read_parquet(path)
        else:
            res = run_backtest(name, strat, wide, dates, cfg["optimization"]["weight_tol"])
            saved = pd.DataFrame({"net": res.net(bt["cost_bps"]), "turnover": res.turnover,
                                  "holdings": (res.weights > 1e-5).sum(axis=1).values})
            saved.to_parquet(path)
        net = saved["net"]
        nets[name] = net
        model = name.removeprefix("BL + ")
        row = {"strategy": name, "group": ("up/down" if model in UPDOWN else
                                           "return model" if name.startswith("BL + ") else "benchmark"),
               "avg_turnover": float(saved["turnover"].iloc[1:].mean()),
               "avg_holdings": float(saved["holdings"].mean())}
        for label, seg in (("dev", net.loc[:split]), ("hold", net.loc[split:].iloc[1:]),
                           ("all", net)):
            p = performance_summary(seg, rf)
            row.update({f"{label}_return": p["ann_return"], f"{label}_vol": p["ann_vol"],
                        f"{label}_sharpe": p["sharpe"], f"{label}_max_dd": max_drawdown(seg),
                        f"{label}_cvar95": cvar_historical(seg)})
        if model in views:
            for label, d in (("dev", dates[dates <= dev_end]), ("hold", dates[dates > dev_end])):
                for k, v in accuracy(views[model].loc[d], realized.loc[d]).items():
                    row[f"{label}_{k}"] = v
        rows.append(row)
        pd.DataFrame(rows).to_csv(OUT / "wide_bl_models.csv", index=False)
        print(f"[{i:2d}/{len(strategies)}] {name:34s} done in {time.time() - t0:5.0f}s", flush=True)

    table = pd.DataFrame(rows)
    up_all = (realized > 0).stack().mean()
    print(f"\nAlways guessing 'up': accuracy {up_all:.1%} (all months)")

    p2 = cfg["phase2"]
    tests = []
    for name in [s for s in nets if s.startswith("BL + ")]:
        for label, sl in (("hold", slice(split, None)), ("all", slice(None, None))):
            a, b = nets[name].loc[sl], nets["BL prior only"].loc[sl]
            if label == "hold":
                a, b = a.iloc[1:], b.iloc[1:]
            boot = sharpe_difference_bootstrap(a, b, rf, block=p2["sharpe_test_block"],
                                               n_boot=p2["sharpe_test_boot"], seed=cfg["seed"])
            tests.append({"strategy": name, "period": label, "sharpe_diff": boot["diff"],
                          "ci_lo": boot["ci_lo"], "ci_hi": boot["ci_hi"], "boot_p": boot["p_value"]})
    tests = pd.DataFrame(tests)
    tests["boot_p_holm"] = holm_adjust(tests["boot_p"])
    tests.to_csv(OUT / "wide_bl_models_tests.csv", index=False)

    cols = ["strategy", "group", "dev_return", "hold_return", "all_return", "dev_sharpe",
            "hold_sharpe", "hold_max_dd", "avg_turnover", "hold_acc_up_down", "hold_acc_vs_median"]
    print("\n" + table[cols].round(3).to_string(index=False))
    print("\nPaired Sharpe tests vs BL prior only (holdout = post-holdout):")
    print(tests.round(3).to_string(index=False))


if __name__ == "__main__":
    main()

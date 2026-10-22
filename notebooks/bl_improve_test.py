"""EXPERIMENT (branch experiment/model-search): better ways for Black–Litterman to use the
ML forecasts (DECISIONS BL1).

Run:  python -m notebooks.bl_improve_test

51 stocks, γ = 5, τ = 0.05, 10 bps, Dec 2015 – Jul 2026 (2022–26 post-holdout). Phase 3
elastic-net (main model) and LightGBM forecasts, each with: Phase 4 views + Ω-A;
IC-scaled views; strong-signal views; one relative top-minus-bottom view; and Ω-A or
IC-scaled views with tracking-error-targeted confidence (2% a year). Placebo: the same
methods with the elastic-net forecasts shuffled across stocks each month (no information).
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.backtest import equal_weight, rebalance_dates, run_backtest
from src.data import load_config, load_processed
from src.experiment_bl import improved_bl_strategy
from src.metrics import (cvar_historical, holm_adjust, max_drawdown, performance_summary,
                         sharpe_difference_bootstrap)

OUT = Path("report/experiments")
TE = 0.02 / np.sqrt(12)
METHODS = {"Ω-A views (Phase 4)": ("omega_a", None),
           "IC-scaled views": ("ic", None),
           "Strong-signal views": ("ic_strong", None),
           "One relative view": ("relative", None),
           "Ω-A views, TE 2%": ("omega_a", TE),
           "IC-scaled views, TE 2%": ("ic", TE)}


def main() -> None:
    cfg = load_config()
    bt, ms = cfg["backtest"], cfg["model_search"]
    data = load_processed(cfg)
    stocks = list(cfg["universe"])
    dates = rebalance_dates(data["returns_monthly"].index, bt["oos_start"])
    split = data["returns_monthly"].index[data["returns_monthly"].index > pd.Timestamp(ms["dev_end"])][0]
    rf = data["rf_monthly"]["rf"]
    g, tau = cfg["risk_tolerance"]["medium"], cfg["black_litterman"]["tau"]
    fdir = Path(cfg["phase3"]["forecast_dir"])
    fc = {"Elastic net": pd.read_parquet(fdir / "forecasts_elastic_net.parquet"),
          "LightGBM": pd.read_parquet(fdir / "forecasts_lightgbm.parquet")}
    rng = np.random.default_rng(ms["placebo"]["seed"])
    en = fc["Elastic net"]
    fc["Placebo (shuffled EN)"] = en.apply(lambda r: pd.Series(rng.permutation(r.values),
                                                               index=r.index), axis=1)

    strategies = {("—", "1/N"): equal_weight(stocks),
                  ("—", "Prior only"): improved_bl_strategy(en, stocks, cfg, g, tau, "ic",
                                                            ic_window=1, ic_prior=10**9)}
    for model, f in fc.items():
        for mname, (method, te) in METHODS.items():
            strategies[(model, mname)] = improved_bl_strategy(f, stocks, cfg, g, tau, method, te)

    rows, nets = [], {}
    for (model, mname), strat in strategies.items():
        t0 = time.time()
        res = run_backtest(mname, strat, data, dates, cfg["optimization"]["weight_tol"])
        net, daily = res.net(bt["cost_bps"]), res.daily_net(bt["cost_bps"])
        nets[(model, mname)] = net
        row = {"model": model, "method": mname, "growth_10k": float(10_000 * (1 + net).prod()),
               "avg_turnover": float(res.turnover.iloc[1:].mean())}
        for label, seg, dseg in (("dev", net.loc[:split], daily.loc[:split]),
                                 ("hold", net.loc[split:].iloc[1:], daily.loc[daily.index > split]),
                                 ("all", net, daily)):
            p = performance_summary(seg, rf)
            row.update({f"{label}_return": p["ann_return"], f"{label}_sharpe": p["sharpe"],
                        f"{label}_max_dd_daily": max_drawdown(dseg),
                        f"{label}_cvar95": cvar_historical(seg)})
        rows.append(row)
        pd.DataFrame(rows).to_csv(OUT / "bl_improve.csv", index=False)
        print(f"{model:22s} {mname:24s} done in {time.time() - t0:4.0f}s", flush=True)
    table = pd.DataFrame(rows)

    tests = []
    for key in [k for k in nets if k[0] != "—"]:
        for label, sl in (("dev", slice(None, split)), ("hold", slice(split, None)),
                          ("all", slice(None, None))):
            a, b = nets[key].loc[sl], nets[("—", "Prior only")].loc[sl]
            if label == "hold":
                a, b = a.iloc[1:], b.iloc[1:]
            boot = sharpe_difference_bootstrap(a, b, rf, block=cfg["phase2"]["sharpe_test_block"],
                                               n_boot=cfg["phase2"]["sharpe_test_boot"],
                                               seed=cfg["seed"])
            tests.append({"model": key[0], "method": key[1], "period": label,
                          "sharpe_diff_vs_prior": boot["diff"], "boot_p": boot["p_value"]})
    tests = pd.DataFrame(tests)
    tests["boot_p_holm"] = holm_adjust(tests["boot_p"])
    tests.to_csv(OUT / "bl_improve_tests.csv", index=False)

    pd.set_option("display.width", 250)
    cols = ["model", "method", "dev_return", "hold_return", "all_return", "growth_10k",
            "dev_sharpe", "hold_sharpe", "hold_max_dd_daily", "avg_turnover"]
    print("\n" + table[cols].round(3).to_string(index=False))
    print("\nPaired Sharpe tests vs prior only:")
    print(tests.round(3).to_string(index=False))


if __name__ == "__main__":
    main()

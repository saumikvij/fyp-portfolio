"""EXPERIMENT (branch experiment/model-search): volatility targeting (MS16).

Run:  python -m notebooks.voltarget_test
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from notebooks.bl_vol_test import vol_forecasts
from src.backtest import black_litterman_strategy, history_at, rebalance_dates, run_backtest
from src.data import load_config, load_processed
from src.estimators import sample_covariance
from src.experiment_vol import dr_covariance
from src.metrics import (holm_adjust, max_drawdown, performance_summary,
                         sharpe_difference_bootstrap, sharpe_difference_memmel)

OUT = Path("report/experiments")
TARGET = 0.10


def main() -> None:
    cfg = load_config()
    data = load_processed(cfg)
    stocks = list(cfg["universe"])
    dates = rebalance_dates(data["returns_monthly"].index, cfg["backtest"]["oos_start"])
    dpm = cfg["data"]["trading_days_per_year"] / 12
    win = cfg["backtest"]["cov_window_days"]
    c = cfg["backtest"]["cost_bps"] / 1e4
    rf = data["rf_monthly"]["rf"]
    ret_fc = pd.read_parquet(Path(cfg["phase3"]["forecast_dir"]) / "forecasts_elastic_net.parquet")
    vol_fc = vol_forecasts(data, stocks, dates, cfg, Path(cfg["data"]["raw_dir"]) / "prices_raw.parquet")

    base = run_backtest("BL prior", black_litterman_strategy(
        ret_fc, None, stocks, cfg, cfg["risk_tolerance"]["medium"], "A",
        cfg["black_litterman"]["tau"], views=False), data, dates, cfg["optimization"]["weight_tol"])
    me = data["returns_monthly"].index
    pred = {"sample Σ": {}, "ML Σ": {}, "last month realized": {}}
    for t in dates:
        w = base.weights.loc[t].reindex(stocks).fillna(0).values
        hist = history_at(data, t)
        S1 = sample_covariance(hist.returns_daily[stocks], t, win, dpm).values
        S2 = dr_covariance(hist, stocks, vol_fc, win, dpm).values
        pred["sample Σ"][t] = np.sqrt(12 * w @ S1 @ w)
        pred["ML Σ"][t] = np.sqrt(12 * w @ S2 @ w)
        last = hist.returns_daily[stocks].iloc[-21:].values @ w
        pred["last month realized"][t] = last.std() * np.sqrt(252)

    gross, turnover = base.gross, base.turnover
    split = me[me > pd.Timestamp(cfg["model_search"]["dev_end"])][0]
    rows, nets = {}, {}
    variants = {"No scaling": pd.Series(1.0, index=dates)}
    for k, v in pred.items():
        variants[f"Target 10%, {k}"] = pd.Series(v).clip(lower=1e-6).rdiv(TARGET).clip(upper=1.0)
    for name, s in variants.items():
        s_hold = pd.Series(s.values, index=gross.index)          # exposure set at t, held over s
        ds = s_hold.diff().abs().fillna(s_hold.iloc[0])
        net = s_hold * gross + (1 - s_hold) * rf.reindex(gross.index) \
            - c * (s_hold * turnover + ds)
        nets[name] = net
        row = {"avg_exposure": float(s_hold.mean())}
        for label, seg in (("dev", net.loc[:split]), ("hold", net.loc[split:].iloc[1:])):
            p = performance_summary(seg, rf)
            vol_m = seg.rolling(12).std() * np.sqrt(12)
            row.update({f"{label}_sharpe": p["sharpe"], f"{label}_return": p["ann_return"],
                        f"{label}_vol": p["ann_vol"], f"{label}_max_dd": max_drawdown(seg),
                        f"{label}_vol_miss": float((vol_m - TARGET).abs().mean())})
        rows[name] = row
    t = pd.DataFrame(rows).T
    t.to_csv(OUT / "voltarget.csv")
    print("BL prior portfolio, 51 stocks, γ = 5, 10 bps (holdout = post-holdout).")
    print("vol_miss = average gap between 12-month realized volatility and the 10% target.")
    print(t.round(3).to_string())

    # Paired Sharpe-difference tests (block bootstrap + Memmel), Holm over the family.
    p2 = cfg["phase2"]
    ml = "Target 10%, ML Σ"
    tests = []
    for other in ["No scaling", "Target 10%, sample Σ", "Target 10%, last month realized"]:
        for label, sl in (("dev", slice(None, split)), ("hold", slice(split, None)),
                          ("all", slice(None, None))):
            a, b = nets[ml].loc[sl], nets[other].loc[sl]
            if label == "hold":
                a, b = a.iloc[1:], b.iloc[1:]
            boot = sharpe_difference_bootstrap(a, b, rf, block=p2["sharpe_test_block"],
                                               n_boot=p2["sharpe_test_boot"], seed=cfg["seed"])
            tests.append({"vs": other, "period": label, "sharpe_diff": boot["diff"],
                          "ci_lo": boot["ci_lo"], "ci_hi": boot["ci_hi"], "boot_p": boot["p_value"],
                          "memmel_p": sharpe_difference_memmel(a, b, rf)["p_value"]})
    tests = pd.DataFrame(tests)
    tests["boot_p_holm"] = holm_adjust(tests["boot_p"])
    tests.to_csv(OUT / "voltarget_tests.csv", index=False)
    print(f"\nPaired tests: {ml} minus each alternative (annualized Sharpe difference):")
    print(tests.round(3).to_string(index=False))


if __name__ == "__main__":
    main()

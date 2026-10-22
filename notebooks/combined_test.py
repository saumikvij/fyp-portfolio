"""EXPERIMENT (branch experiment/model-search): Black–Litterman with combined views and the
ML risk matrix on the 51 stocks (DECISIONS WU7).

Run:  python -m notebooks.combined_test

Views: industry-rotation sector views (MS15.2), Phase 3 LightGBM stock views, optionally
IPCA stock views (MS14), stacked in one P (``combined_bl_strategy``). Σ: sample, or
D·R·D with elastic-net volatility forecasts (MS15.3). γ = 5, τ = 0.05, Ω-A, 10 bps.
Chosen after the holdout was seen, so 2022–26 results are post-holdout.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from notebooks.bl_vol_test import vol_forecasts
from src.backtest import rebalance_dates, run_backtest
from src.data import load_config, load_processed
from src.experiment_sectors import combined_bl_strategy, sector_returns
from src.experiment_universe import load_wide
from src.metrics import (cvar_historical, holm_adjust, max_drawdown, performance_summary,
                         sharpe_difference_bootstrap)

OUT = Path("report/experiments")


def main() -> None:
    cfg = load_config()
    ms, bt = cfg["model_search"], cfg["backtest"]
    data = load_processed(cfg)
    stocks = list(cfg["universe"])
    dates = rebalance_dates(data["returns_monthly"].index, bt["oos_start"])
    split = data["returns_monthly"].index[data["returns_monthly"].index > pd.Timestamp(ms["dev_end"])][0]
    rf = data["rf_monthly"]["rf"]
    g, tau = cfg["risk_tolerance"]["medium"], cfg["black_litterman"]["tau"]

    rot = pd.read_parquet(Path(ms["out_dir"]) / "sector_rotation_forecasts.parquet")
    sret = sector_returns(load_wide())
    lgbm = pd.read_parquet(Path(cfg["phase3"]["forecast_dir"]) / "forecasts_lightgbm.parquet")
    ipca = pd.read_parquet(Path(ms["out_dir"]) / "ipca__roll60m__raw__risk.parquet")
    vol = vol_forecasts(data, stocks, dates, cfg, Path(cfg["data"]["raw_dir"]) / "prices_raw.parquet")

    def s(views, sector=False, ml_sigma=False):
        return combined_bl_strategy(views, rot if sector else None, sret if sector else None,
                                    stocks, cfg, g, tau, vol if ml_sigma else None)

    strategies = {
        "No AI views (prior), sample Σ": s({}),
        "No AI views (prior), ML Σ": s({}, ml_sigma=True),
        "Rotation only": s({}, sector=True),
        "LightGBM only": s({"LightGBM": lgbm}),
        "Rotation + LightGBM": s({"LightGBM": lgbm}, sector=True),
        "Rotation + LightGBM, ML Σ": s({"LightGBM": lgbm}, sector=True, ml_sigma=True),
        "Rotation + LightGBM + IPCA, ML Σ": s({"LightGBM": lgbm, "IPCA": ipca}, sector=True,
                                              ml_sigma=True),
    }
    rows, nets = [], {}
    for name, strat in strategies.items():
        res = run_backtest(name, strat, data, dates, cfg["optimization"]["weight_tol"])
        net, daily = res.net(bt["cost_bps"]), res.daily_net(bt["cost_bps"])
        nets[name] = net
        row = {"strategy": name, "growth_10k": float(10_000 * (1 + net).prod()),
               "avg_turnover": float(res.turnover.iloc[1:].mean())}
        for label, seg, dseg in (("dev", net.loc[:split], daily.loc[:split]),
                                 ("hold", net.loc[split:].iloc[1:], daily.loc[daily.index > split]),
                                 ("all", net, daily)):
            p = performance_summary(seg, rf)
            row.update({f"{label}_return": p["ann_return"], f"{label}_vol": p["ann_vol"],
                        f"{label}_sharpe": p["sharpe"], f"{label}_max_dd_daily": max_drawdown(dseg),
                        f"{label}_cvar95": cvar_historical(seg)})
        rows.append(row)
        print(f"{name:36s} done", flush=True)
    table = pd.DataFrame(rows)
    table.to_csv(OUT / "combined_bl.csv", index=False)

    base = "No AI views (prior), sample Σ"
    tests = []
    for name in [n for n in nets if n != base]:
        for label, sl in (("dev", slice(None, split)), ("hold", slice(split, None)),
                          ("all", slice(None, None))):
            a, b = nets[name].loc[sl], nets[base].loc[sl]
            if label == "hold":
                a, b = a.iloc[1:], b.iloc[1:]
            boot = sharpe_difference_bootstrap(a, b, rf, block=cfg["phase2"]["sharpe_test_block"],
                                               n_boot=cfg["phase2"]["sharpe_test_boot"],
                                               seed=cfg["seed"])
            tests.append({"strategy": name, "period": label, "sharpe_diff": boot["diff"],
                          "boot_p": boot["p_value"]})
    tests = pd.DataFrame(tests)
    tests["boot_p_holm"] = holm_adjust(tests["boot_p"])
    tests.to_csv(OUT / "combined_bl_tests.csv", index=False)

    cols = ["strategy", "dev_return", "hold_return", "all_return", "growth_10k", "dev_sharpe",
            "hold_sharpe", "hold_max_dd_daily", "all_cvar95", "avg_turnover"]
    pd.set_option("display.width", 250)
    print("\n" + table[cols].round(3).to_string(index=False))
    print(f"\nPaired Sharpe tests vs '{base}':")
    print(tests.round(3).to_string(index=False))


if __name__ == "__main__":
    main()

"""EXPERIMENT: one table of every return model fed into Black–Litterman (Ω-A, γ = 5,
10 bps, 51 stocks), plus baselines. Development 2016–2021, holdout 2022–2026 (the holdout
was used for selection checks, so it is post-holdout for most rows).

Run:  python -m notebooks.summary_all
"""
from pathlib import Path

import numpy as np
import pandas as pd

from src.backtest import (black_litterman_strategy, equal_weight, markowitz_strategy,
                          rebalance_dates, run_backtest)
from src.data import load_config, load_processed
from src.experiment_sectors import sector_bl_strategy, sector_returns
from src.experiment_universe import load_wide
from src.features import add_targets, build_feature_panel, rank_normalize
from src.metrics import max_drawdown, performance_summary
from src.ml_eval import rank_ic, summarize_series

S = "data/ml/search/"
MODELS = {
    "Elastic net (Phase 3)": "data/ml/forecasts_elastic_net.parquet",
    "LightGBM (Phase 3)": "data/ml/forecasts_lightgbm.parquet",
    "MLP (Phase 3)": "data/ml/forecasts_mlp.parquet",
    "Kernel ridge (search finalist)": S + "kernel_ridge__roll48m__raw__risk__wide_r12.parquet",
    "KNN (search finalist)": S + "knn__roll48m__demeaned__risk__wide_r12.parquet",
    "LightGBM (search finalist)": S + "lightgbm__roll48m__demeaned__risk__wide_r12.parquet",
    "IPCA": S + "ipca__roll60m__raw__risk.parquet",
    "Learning to rank": "data/ml/search_ltr/ltr__large51__roll60m__base.parquet",
}


def main():
    cfg = load_config()
    d = load_processed(cfg)
    st = list(cfg["universe"])
    dates = rebalance_dates(d["returns_monthly"].index, cfg["backtest"]["oos_start"])
    dev_end = pd.Timestamp(cfg["model_search"]["dev_end"])
    dev, hold = dates[dates <= dev_end], dates[dates > dev_end]
    split = d["returns_monthly"].index[d["returns_monthly"].index > dev_end][0]
    rf = d["rf_monthly"]["rf"]
    g, tau, c = cfg["risk_tolerance"]["medium"], cfg["black_litterman"]["tau"], cfg["backtest"]["cost_bps"]
    p = add_targets(rank_normalize(build_feature_panel(d, st, "SPY", cfg["ml"]["windows"],
                                                       cfg["ml"]["rsi_days"])),
                    d["returns_monthly"][st], d["rf_monthly"]["rf"])
    y = p["target"].unstack()
    ref = pd.read_parquet(MODELS["Elastic net (Phase 3)"])
    wide = load_wide()
    sret = sector_returns(wide)
    sector_fc = pd.read_parquet(S + "sector_rotation_forecasts.parquet")

    strategies = {"1/N (no ML)": (equal_weight(st), None),
                  "Markowitz (no ML)": (markowitz_strategy(st, cfg, g), None),
                  "BL market view only (no ML)": (black_litterman_strategy(
                      ref, None, st, cfg, g, "A", tau, views=False), None)}
    for name, path in MODELS.items():
        f = pd.read_parquet(path)
        strategies[f"BL + {name}"] = (black_litterman_strategy(f, None, st, cfg, g, "A", tau), f)
    strategies["BL + Industry rotation"] = (sector_bl_strategy(sector_fc, sret, st, cfg, g, tau), None)

    rows = []
    for name, (strat, f) in strategies.items():
        res = run_backtest(name, strat, d, dates, cfg["optimization"]["weight_tol"])
        net = res.net(c)
        row = {"strategy": name}
        if f is not None:
            yy = y.reindex(index=f.index, columns=f.columns)
            ok = f.notna() & yy.notna()
            row["up_down_acc"] = float((np.sign(f) == np.sign(yy))[ok].sum().sum() / ok.sum().sum())
            for lab, part in (("dev", dev), ("hold", hold)):
                row[f"{lab}_ic"] = summarize_series(rank_ic(f.loc[part], yy.loc[part]))["mean"]
        for lab, seg in (("dev", net.loc[:split]), ("hold", net.loc[split:].iloc[1:])):
            row[f"{lab}_sharpe"] = performance_summary(seg, rf)["sharpe"]
        row["dev_return"] = performance_summary(net.loc[:split], rf)["ann_return"]
        row["hold_return"] = performance_summary(net.loc[split:].iloc[1:], rf)["ann_return"]
        row["all_return"] = performance_summary(net, rf)["ann_return"]
        row["growth_10k"] = float(10_000 * (1 + net).prod())
        row["hold_max_dd"] = max_drawdown(net.loc[split:].iloc[1:])
        row["turnover"] = float(res.turnover.iloc[1:].mean())
        rows.append(row)
    t = pd.DataFrame(rows).set_index("strategy")
    t.to_csv("report/experiments/summary_all_models.csv")
    pd.set_option("display.width", 200)
    print(t.round(3).to_string())


if __name__ == "__main__":
    main()

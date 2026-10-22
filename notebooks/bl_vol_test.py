"""EXPERIMENT (branch experiment/model-search): Black–Litterman with the ML risk matrix
(MS15.5). Σ = D·R·D with elastic-net volatility forecasts vs the sample Σ, for the prior
alone and with the Phase 4 elastic-net return views; 2016–2026 (development / holdout,
holdout post-holdout) and the 2008–2009 crisis. γ = 5, 10 bps, 51 (crisis: 50) stocks.

Run:  python -m notebooks.bl_vol_test
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from notebooks.model_search import build_model
from src.backtest import black_litterman_strategy, rebalance_dates, run_backtest
from src.data import load_config, load_processed
from src.experiment_crisis import CRISIS, load_crisis
from src.experiment_signals import load_ohlc
from src.experiment_vol import VOL_FEATURES, bl_with_vol_sigma, vol_panel
from src.features import FEATURES, add_targets, build_feature_panel, rank_normalize
from src.metrics import cvar_historical, max_drawdown, performance_summary
from src.models import walk_forward

OUT = Path("report/experiments")


def vol_forecasts(data, stocks, dates, cfg, raw_path):
    ml = cfg["ml"]
    panel = vol_panel(data, stocks, cfg["data"]["benchmark"],
                      load_ohlc(raw_path, data["prices"].index, stocks))
    return walk_forward(build_model("Elastic net", cfg, VOL_FEATURES), panel, dates,
                        ml["val_months"], ml["min_train_months"]).forecasts


def evaluate(data, stocks, dates, cfg, ret_fc, vol_fc, split=None):
    g, tau = cfg["risk_tolerance"]["medium"], cfg["black_litterman"]["tau"]
    strategies = {
        "Prior only, sample Σ": black_litterman_strategy(ret_fc, None, stocks, cfg, g, "A", tau,
                                                         views=False),
        "Prior only, ML Σ": bl_with_vol_sigma(vol_fc, ret_fc, stocks, cfg, g, tau, views=False),
        "Prior + EN views, sample Σ": black_litterman_strategy(ret_fc, None, stocks, cfg, g, "A",
                                                               tau),
        "Prior + EN views, ML Σ": bl_with_vol_sigma(vol_fc, ret_fc, stocks, cfg, g, tau),
    }
    rf = data["rf_monthly"]["rf"]
    rows = {}
    for name, strat in strategies.items():
        res = run_backtest(name, strat, data, dates, cfg["optimization"]["weight_tol"])
        net, daily = res.net(cfg["backtest"]["cost_bps"]), res.daily_net(cfg["backtest"]["cost_bps"])
        parts = {"all": (net, daily)} if split is None else {
            "dev": (net.loc[:split], daily.loc[:split]),
            "hold": (net.loc[split:].iloc[1:], daily.loc[daily.index > split])}
        row = {"avg_turnover": float(res.turnover.iloc[1:].mean())}
        for label, (m, d) in parts.items():
            p = performance_summary(m, rf)
            row.update({f"{label}_sharpe": p["sharpe"], f"{label}_vol": p["ann_vol"],
                        f"{label}_max_dd": max_drawdown(d), f"{label}_cvar_d": cvar_historical(d)})
        rows[name] = row
    return pd.DataFrame(rows).T


def main() -> None:
    cfg = load_config()
    ml = cfg["ml"]
    data = load_processed(cfg)
    stocks = list(cfg["universe"])
    dates = rebalance_dates(data["returns_monthly"].index, cfg["backtest"]["oos_start"])
    ret_fc = pd.read_parquet(Path(cfg["phase3"]["forecast_dir"]) / "forecasts_elastic_net.parquet")
    vol_fc = vol_forecasts(data, stocks, dates, cfg, Path(cfg["data"]["raw_dir"]) / "prices_raw.parquet")
    dev_end = pd.Timestamp(cfg["model_search"]["dev_end"])
    split = data["returns_monthly"].index[data["returns_monthly"].index > dev_end][0]
    t1 = evaluate(data, stocks, dates, cfg, ret_fc, vol_fc, split)
    t1.to_csv(OUT / "bl_vol_2016_2026.csv")
    print("51 stocks, 2016–2026 (holdout = post-holdout):")
    print(t1.round(3).to_string())

    crisis = load_crisis(cfg)
    cs = crisis["stocks"]
    ccfg = {**cfg, "universe": {s: cfg["universe"][s] for s in cs}}
    cdates = rebalance_dates(crisis["returns_monthly"].index, "2008-01-01")
    base = add_targets(rank_normalize(build_feature_panel(
        crisis, cs, cfg["data"]["benchmark"], ml["windows"], ml["rsi_days"], "base")),
        crisis["returns_monthly"][cs], crisis["rf_monthly"]["rf"])
    cret = walk_forward(build_model("Elastic net", ccfg, FEATURES), base, cdates,
                        ml["val_months"], ml["min_train_months"]).forecasts
    cvol = vol_forecasts(crisis, cs, cdates, ccfg, CRISIS / "raw" / "prices_raw.parquet")
    t2 = evaluate(crisis, cs, cdates, ccfg, cret, cvol)
    t2.to_csv(OUT / "bl_vol_crisis.csv")
    print("\n50 stocks, 2008–2009 crisis:")
    print(t2.round(3).to_string())


if __name__ == "__main__":
    main()

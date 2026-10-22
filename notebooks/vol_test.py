"""EXPERIMENT (branch experiment/model-search): ML volatility forecasting, MS15 strategy 3.

Run:  python -m notebooks.vol_test

1. Forecast next-month log volatility of each of the 51 stocks with: trailing 63-day vol
   (naive), HAR linear regression, elastic net, LightGBM (walk-forward, chronological
   validation). Accuracy by mean squared error of log volatility and rank correlation,
   development vs holdout (holdout flagged post-holdout).
2. Minimum-variance portfolios with Σ = D·R·D from each forecast vs the sample Σ:
   realized volatility, drawdown, and how well the predicted portfolio volatility
   matched the realized one (the "risk level" a user is promised).
3. The same on the 2008–2009 crisis data.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from notebooks.model_search import build_model
from src.backtest import min_variance_strategy, rebalance_dates, run_backtest
from src.data import load_config, load_processed
from src.experiment_crisis import CRISIS, load_crisis
from src.experiment_signals import load_ohlc
from src.experiment_vol import VOL_FEATURES, dr_covariance, min_var_with, vol_panel
from src.metrics import max_drawdown
from src.ml_eval import rank_ic, summarize_series
from src.models import GridModel, walk_forward

OUT = Path("report/experiments")


def models(cfg):
    return {
        "HAR (linear)": lambda: GridModel("HAR", lambda fit_intercept, seed: LinearRegression(),
                                          {"fit_intercept": [True]}, cfg["seed"],
                                          features=["lv_5", "lv_21", "lv_63"]),
        "Elastic net": lambda: build_model("Elastic net", cfg, VOL_FEATURES),
        "LightGBM": lambda: build_model("LightGBM", cfg, VOL_FEATURES),
    }


def run(data, stocks, dates, cfg, raw_path, label, split=None):
    ml = cfg["ml"]
    ohlc = load_ohlc(raw_path, data["prices"].index, stocks)
    panel = vol_panel(data, stocks, cfg["data"]["benchmark"], ohlc)
    realized = panel["target"].unstack().reindex(dates)
    fcs = {"Naive (trailing 63-day)": panel["lv_63"].unstack().reindex(dates)}
    for name, fn in models(cfg).items():
        fcs[name] = walk_forward(fn(), panel, dates, ml["val_months"], ml["min_train_months"]).forecasts
    parts = {"all": dates} if split is None else {"dev": dates[dates <= split], "hold": dates[dates > split]}
    # Diebold–Mariano: monthly average loss difference (squared log-vol error), NW t-stat.
    print(f"\n{label}: Diebold–Mariano tests (negative = elastic net more accurate)")
    for base in ("Naive (trailing 63-day)", "HAR (linear)"):
        for p, d in parts.items():
            diff = (((fcs["Elastic net"] - realized) ** 2).loc[d].mean(axis=1)
                    - ((fcs[base] - realized) ** 2).loc[d].mean(axis=1))
            s_ = summarize_series(diff)
            print(f"  Elastic net vs {base:24s} {p}: mean loss diff {s_['mean']:+.4f} "
                  f"(NW t {s_['t_stat_nw']:+.2f})")
    print(f"\n{label}: forecast accuracy (MSE of log vol; rank corr with realized)")
    for name, f in fcs.items():
        out = []
        for p, d in parts.items():
            mse = float(((f.loc[d] - realized.loc[d]) ** 2).stack().mean())
            ic = summarize_series(rank_ic(f.loc[d], realized.loc[d]))["mean"]
            out.append(f"{p}: MSE {mse:.4f}, rank corr {ic:.3f}")
        print(f"  {name:24s} " + " | ".join(out))

    # Portfolios: minimum variance with each covariance.
    wt, c = cfg["optimization"]["weight_tol"], cfg["backtest"]["cost_bps"]
    strategies = {"Sample Σ": min_variance_strategy(stocks, cfg)}
    for name in ("Naive (trailing 63-day)", "HAR (linear)", "LightGBM", "Elastic net"):
        strategies[f"D·R·D, {name}"] = min_var_with(fcs[name], stocks, cfg)
    print(f"\n{label}: minimum-variance portfolios")
    rows = {}
    for name, strat in strategies.items():
        res = run_backtest(name, strat, data, dates, wt)
        net = res.net(c)
        rows[name] = {"realized_vol": float(net.std() * np.sqrt(12)),
                      "max_dd_daily": max_drawdown(res.daily_net(c)),
                      "avg_turnover": float(res.turnover.iloc[1:].mean())}
        if split is not None:
            h = net.loc[net.index > split]
            rows[name]["hold_vol"] = float(h.std() * np.sqrt(12))
    t = pd.DataFrame(rows).T
    print(t.round(4).to_string())

    # Risk-level accuracy: for the same portfolio (sample-Σ minimum variance), how close is
    # its predicted next-month volatility to the realized one, under each risk model?
    res = run_backtest("mv", min_variance_strategy(stocks, cfg), data, dates, wt)
    daily = res.daily_gross
    dpm = cfg["data"]["trading_days_per_year"] / 12
    from src.estimators import sample_covariance
    err = {k: [] for k in ("Sample Σ", "D·R·D, Elastic net")}
    me = data["returns_monthly"].index
    for t_ in dates:
        w = res.weights.loc[t_].reindex(stocks).fillna(0).values
        s_ = me[me > t_][0]
        realized_vol = float(daily.loc[(daily.index > t_) & (daily.index <= s_)].std() * np.sqrt(dpm))
        hist = __import__("src.backtest", fromlist=["history_at"]).history_at(data, t_)
        S1 = sample_covariance(hist.returns_daily[stocks], t_, cfg["backtest"]["cov_window_days"], dpm)
        S2 = dr_covariance(hist, stocks, fcs["Elastic net"], cfg["backtest"]["cov_window_days"], dpm)
        for k, S in (("Sample Σ", S1), ("D·R·D, Elastic net", S2)):
            err[k].append(np.log(np.sqrt(w @ S.values @ w)) - np.log(realized_vol))
    print(f"\n{label}: predicted vs realized portfolio volatility (log error)")
    for k, e in err.items():
        e = np.array(e)
        print(f"  {k:20s} bias {e.mean():+.3f}, RMSE {np.sqrt((e ** 2).mean()):.3f}")
    return fcs, t


def main() -> None:
    cfg = load_config()
    bt = cfg["backtest"]
    data = load_processed(cfg)
    stocks = list(cfg["universe"])
    dates = rebalance_dates(data["returns_monthly"].index, bt["oos_start"])
    _, t51 = run(data, stocks, dates, cfg, Path(cfg["data"]["raw_dir"]) / "prices_raw.parquet",
                 "51 stocks, 2016–2026", split=pd.Timestamp(cfg["model_search"]["dev_end"]))
    t51.to_csv(OUT / "vol_portfolios_51.csv")

    crisis = load_crisis(cfg)
    cs = crisis["stocks"]
    cdates = rebalance_dates(crisis["returns_monthly"].index, "2008-01-01")
    ccfg = {**cfg, "universe": {s: cfg["universe"][s] for s in cs}}
    _, tc = run(crisis, cs, cdates, ccfg, CRISIS / "raw" / "prices_raw.parquet",
                "50 stocks, 2008–2009 crisis")
    tc.to_csv(OUT / "vol_portfolios_crisis.csv")


if __name__ == "__main__":
    main()

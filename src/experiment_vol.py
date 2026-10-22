"""EXPERIMENT (branch experiment/model-search): ML volatility forecasting, MS15 strategy 3.

Target: log of each stock's realized daily volatility over the next month. Inputs (all
known at t, kept as levels, not ranked): log volatility over the last 5, 21, 63 and 252
trading days, log high–low range volatility (21 days), log market volatility (21 days).
Baselines: plain trailing 63-day volatility, and a HAR-style linear regression on the
5/21/63-day volatilities (Corsi 2009; "HARd to Beat", 2024).

The forecasts build a covariance Σ = D·R·D: D = forecast monthly volatilities,
R = trailing 252-day correlation matrix.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.backtest import History, _constraints
from src.optimization import min_variance

VOL_FEATURES = ["lv_5", "lv_21", "lv_63", "lv_252", "lrange_21", "lmkt_21"]


def vol_panel(data: dict, stocks: list[str], bench: str, ohlc: dict | None = None) -> pd.DataFrame:
    """Features and next-month log-volatility target at every month-end (no look-ahead:
    features use days <= t, the target uses the days of the following month)."""
    r = data["returns_daily"]
    me = data["returns_monthly"].index
    nxt = pd.Series(list(me[1:]) + [pd.NaT], index=me)
    frames = {}
    for t in me:
        past = r.loc[:t]
        if len(past) < 253:
            continue
        f = pd.DataFrame(index=stocks)
        for w in (5, 21, 63, 252):
            f[f"lv_{w}"] = np.log(past[stocks].iloc[-w:].std() + 1e-8)
        if ohlc is not None:
            h, lo = ohlc["High"].loc[:t, stocks], ohlc["Low"].loc[:t, stocks]
            f["lrange_21"] = np.log(np.sqrt((np.log(h / lo) ** 2).iloc[-21:].mean()
                                            / (4 * np.log(2))) + 1e-8)
        else:
            f["lrange_21"] = f["lv_21"]
        f["lmkt_21"] = float(np.log(past[bench].iloc[-21:].std() + 1e-8))
        s = nxt[t]
        if pd.notna(s):
            fut = r.loc[(r.index > t) & (r.index <= s), stocks]
            f["target"] = np.log(fut.std() + 1e-8)
        else:
            f["target"] = np.nan
        f["target_date"] = s
        frames[t] = f
    return pd.concat(frames, names=["date", "ticker"])


def dr_covariance(hist: History, stocks: list[str], vol_fc: pd.DataFrame, window: int,
                  days_per_month: float) -> pd.DataFrame:
    """Σ = D·R·D with D = exp(forecast log daily vol)·√days_per_month and R the
    trailing-``window`` correlation of daily returns (all data <= t)."""
    x = hist.returns_daily[stocks].iloc[-window:]
    R = x.corr().values
    d = np.exp(vol_fc.loc[hist.t, stocks].values) * np.sqrt(days_per_month)
    return pd.DataFrame(R * np.outer(d, d), index=stocks, columns=stocks)


def min_var_with(vol_fc: pd.DataFrame, stocks: list[str], cfg: dict):
    """Minimum-variance strategy (long-only, capped) using the D·R·D covariance."""
    def strategy(hist: History) -> pd.Series:
        S = dr_covariance(hist, stocks, vol_fc, cfg["backtest"]["cov_window_days"],
                          cfg["data"]["trading_days_per_year"] / 12)
        return min_variance(S, _constraints(hist, cfg), cfg["optimization"]["solver"]).weights
    return strategy

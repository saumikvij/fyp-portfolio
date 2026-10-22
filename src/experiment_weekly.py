"""EXPERIMENT (branch experiment/model-search): weekly version of the data (DECISIONS E1).

The backtest engine and the models work on "periods" (the rows of ``returns_monthly``),
so the weekly test swaps in weekly tables under the same keys: week-ends = last trading
day of each calendar week; weekly returns compounded from daily; weekly risk-free return
(1 + annual yield at the previous week-end)^(1/52) − 1, the weekly analogue of
``src.data.risk_free_rates``. Features keep their monthly definitions, with "months"
counted as blocks of 21 trading days back from each week-end (windows in trading days,
as pre-registered in E1).
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd

from src.features import add_targets, features_at, rank_normalize


def weekly_data(data: dict) -> dict:
    """Copy of the processed data with weekly tables under the monthly keys."""
    rd = data["returns_daily"]
    days = rd.index
    week = days.to_period("W-FRI")
    week_ends = pd.Series(days, index=days).groupby(week).max().values
    week_ends = pd.DatetimeIndex(week_ends)
    if (week == week[-1]).sum() < 3:          # drop a trailing partial week
        keep = week != week[-1]
        rd, days, week, week_ends = rd[keep], days[keep], week[keep], week_ends[:-1]
    rw = (1 + rd).groupby(week).prod() - 1
    rw.index = week_ends
    annual = data["rf_daily"]["annual"].reindex(days).ffill()
    rf = ((1 + annual.reindex(week_ends).shift(1)) ** (1 / 52) - 1).iloc[1:]
    out = dict(data)
    out["returns_monthly"] = rw.iloc[1:]
    out["prices_monthly"] = data["prices"].reindex(week_ends).iloc[1:]
    out["rf_monthly"] = rf.rename("rf").to_frame()
    return out


def weekly_feature_panel(data: dict, weekly: dict, stocks: list[str], bench: str,
                         windows: dict, rsi_days: int, block_days: int = 21) -> pd.DataFrame:
    """Rank-normalized base features at every week-end with next-week targets."""
    prices, volume = data["prices"], data["volume"]
    pos = prices.index.get_indexer(weekly["returns_monthly"].index)
    frames = {}
    t0 = time.time()
    for t, i in zip(weekly["returns_monthly"].index, pos):
        if i <= max(windows["y1"], 12 * block_days):      # same warm-up as monthly (1 year)
            continue
        pm = prices.iloc[[i - block_days * k for k in range(12, -1, -1)]]
        f = features_at(prices.iloc[:i + 1], volume.iloc[:i + 1], pm, bench, stocks,
                        windows, rsi_days)
        frames[t] = f
    raw = pd.concat(frames, names=["date", "ticker"]).dropna()
    print(f"weekly features: {raw.shape} in {time.time() - t0:.0f}s", flush=True)
    return add_targets(rank_normalize(raw), weekly["returns_monthly"][stocks],
                       weekly["rf_monthly"]["rf"])


def weekly_cfg(cfg: dict) -> dict:
    """Config with horizon-dependent settings rescaled to keep the same calendar length
    (E1): retrain every 52 weeks, validation 104 weeks, Ω-A window 104 weeks with a
    52-week warm-up, covariance scaled ×5 (trading_days_per_year/12 is used as the
    days-per-period factor)."""
    import copy
    c = copy.deepcopy(cfg)
    c["ml"].update(retrain_every_months=52, val_months=104, min_train_months=104)
    c["black_litterman"].update(omega_a_window=104, omega_a_prior_months=52)
    c["data"]["trading_days_per_year"] = 60
    for fam, over in c["model_search"]["wide"]["overrides"].items():
        c["model_search"][fam].update(over)
    return c


def weekly_metrics(r: pd.Series, rf: pd.Series) -> dict:
    from src.metrics import annualized_return, annualized_vol, cvar_historical, max_drawdown, sharpe_ratio
    return {"return": annualized_return(r, 52), "vol": annualized_vol(r, 52),
            "sharpe": sharpe_ratio(r, rf.reindex(r.index), 52), "max_dd": max_drawdown(r),
            "cvar95": cvar_historical(r)}

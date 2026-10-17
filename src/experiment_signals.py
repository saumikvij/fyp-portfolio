"""EXPERIMENT (branch experiment/model-search): niche signals for round 3 (DECISIONS MS7).

Every signal at month-end t uses only data dated <= t. OHLC prices come from the raw
download and are split-adjusted with the ratio Adj Close / Close, so overnight and
intraday returns are not distorted by splits.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.features import FEATURE_SETS, build_feature_panel

NICHE = ["season", "overnight_1m", "intraday_1m", "range_vol_1m", "trend_200d", "mom_accel",
         "idio_mom_12_1", "pos_days_3m", "volume_shock", "downside_beta_1y", "coskew_1y",
         "sector_mom", "sector_rev_1m"]
MARKET_STATE = ["mkt_vol_1m", "rf_level", "mkt_mom_12m"]   # same for every stock; unranked
NICHE_SET = FEATURE_SETS["extended"] + NICHE + MARKET_STATE


def load_ohlc(raw_path: str | Path, calendar: pd.DatetimeIndex, stocks: list[str]) -> dict:
    """Split-adjusted daily Open, High, Low, Close on the trading calendar."""
    raw = pd.read_parquet(raw_path)
    wide = {c: raw[c].unstack("Ticker").reindex(calendar)[stocks]
            for c in ("Open", "High", "Low", "Close", "Adj Close")}
    k = wide["Adj Close"] / wide["Close"]
    return {c: (wide[c] * k).ffill(limit=5) for c in ("Open", "High", "Low", "Close")}


def niche_at(t, data, ohlc, stocks, sectors, bench) -> pd.DataFrame:
    """The 13 stock-level niche signals at month-end t (rows = stocks)."""
    px = data["prices"].loc[:t]
    r = px.pct_change(fill_method=None)
    rs, rb = r[stocks], r[bench]
    rm = data["returns_monthly"].loc[:t, stocks]
    o, h, lo, c = (ohlc[k].loc[:t] for k in ("Open", "High", "Low", "Close"))
    out = pd.DataFrame(index=stocks, dtype=float)

    # Seasonality: average return of the calendar month t+1 in up to 5 past years.
    nxt_month = (t + pd.offsets.MonthEnd(1)).month
    same = rm[rm.index.month == nxt_month].iloc[-5:]
    out["season"] = same.mean() if len(same) else np.nan

    m1 = 21
    out["overnight_1m"] = np.log(o / c.shift(1)).iloc[-m1:].sum()
    out["intraday_1m"] = np.log(c / o).iloc[-m1:].sum()
    out["range_vol_1m"] = np.sqrt((np.log(h / lo) ** 2).iloc[-m1:].mean() / (4 * np.log(2)))
    out["trend_200d"] = px[stocks].iloc[-1] / px[stocks].iloc[-200:].mean() - 1

    pm = data["prices_monthly"].loc[:t, stocks]
    if len(pm) > 12:
        out["mom_accel"] = (pm.iloc[-1] / pm.iloc[-7] - 1) - (pm.iloc[-7] / pm.iloc[-13] - 1)
    y = rs.iloc[-252:]
    b = rb.iloc[-252:]
    beta = y.apply(lambda col: np.cov(col, b)[0, 1]) / b.var()
    resid = y - np.outer(b, beta)
    out["idio_mom_12_1"] = resid.iloc[:-m1].sum()
    out["pos_days_3m"] = (rs.iloc[-63:] > 0).mean()
    v = data["volume"].loc[:t, stocks]
    out["volume_shock"] = np.log(v.iloc[-5:].mean() / v.iloc[-63:].mean())
    down = b < 0
    out["downside_beta_1y"] = y[down].apply(lambda col: np.cov(col, b[down])[0, 1]) / b[down].var()
    bd, yd = b - b.mean(), y - y.mean()
    out["coskew_1y"] = (yd.mul(bd ** 2, axis=0)).mean() / (yd.std() * (bd ** 2).mean())

    mom = pm.iloc[-2] / pm.iloc[-13] - 1 if len(pm) > 12 else pd.Series(np.nan, index=stocks)
    out["sector_mom"] = mom.groupby(sectors).transform("mean")
    rev = pm.iloc[-1] / pm.iloc[-2] - 1 if len(pm) > 1 else pd.Series(np.nan, index=stocks)
    out["sector_rev_1m"] = rev - rev.groupby(sectors).transform("mean")
    return out


def market_state_at(t, data, bench) -> dict:
    """Market-wide context at t (fixed scaling, no fitted parameters)."""
    p = data["prices"].loc[:t, bench]
    r = p.pct_change(fill_method=None)
    return {"mkt_vol_1m": float(r.iloc[-21:].std() * np.sqrt(252)),
            "rf_level": float(data["rf_daily"]["annual"].loc[:t].iloc[-1]),
            "mkt_mom_12m": float(p.iloc[-1] / p.iloc[-253] - 1)}


def build_niche_panel(data, stocks, bench, windows, rsi_days, raw_path) -> pd.DataFrame:
    """Raw extended features + niche signals + market state at every month-end.

    Stock-level columns are later rank-normalized like the others; the market-state
    columns are kept as they are (``rank_normalize_niche``).
    """
    base = build_feature_panel(data, stocks, bench, windows, rsi_days, "extended")
    ohlc = load_ohlc(raw_path, data["prices"].index, stocks)
    sectors = data["sectors"]["sector"].reindex(stocks)
    frames = {}
    for t in base.index.get_level_values("date").unique():
        f = niche_at(t, data, ohlc, stocks, sectors, bench)
        for k, v in market_state_at(t, data, bench).items():
            f[k] = v
        frames[t] = f
    niche = pd.concat(frames, names=["date", "ticker"])
    panel = base.join(niche)[NICHE_SET]
    return panel.dropna()


def rank_normalize_niche(panel: pd.DataFrame) -> pd.DataFrame:
    """Rank stock-level columns within each month to [−0.5, 0.5]; keep market state raw."""
    stock_cols = [c for c in panel.columns if c not in MARKET_STATE]
    g = panel[stock_cols].groupby(level="date")
    ranked = (g.rank(method="average") - 1) / (g.transform("count") - 1) - 0.5
    return ranked.join(panel[MARKET_STATE])

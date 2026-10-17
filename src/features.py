"""Point-in-time features and targets for the return forecasts (plan Section 4.1).

At each month-end t every feature of stock i uses only prices and volumes dated <= t.
The target is the stock's excess return over the *next* month, which is realized at
the next month-end s; ``target_date`` records s so training sets can be cut exactly.

Features (13; the plan's "1-month momentum" and "1-month reversal" are the same
signal, last month's return, so it appears once as ``rev_1m``):

    rev_1m       return over the last month
    mom_3m       return over the last 3 months
    mom_6m       return over the last 6 months
    mom_12_1     return from 12 months ago to 1 month ago (skips the last month)
    vol_1m       std of daily returns, last 21 trading days
    vol_3m       std of daily returns, last 63 trading days
    beta_1y      beta to SPY over the last 252 trading days
    idio_vol_3m  std of the residual of daily returns on SPY, last 63 trading days
    max_ret_1m   largest daily return, last 21 trading days
    dist_52w     price / 252-day high − 1
    volume_trend log(average volume last 21 days / average volume last 252 days)
    rsi          relative strength index: 100 − 100 / (1 + avg gain / avg loss), 14 days
    max_dd_1y    largest peak-to-trough fall within the last 252 trading days
"""

from __future__ import annotations

import numpy as np
import pandas as pd

FEATURES = ["rev_1m", "mom_3m", "mom_6m", "mom_12_1", "vol_1m", "vol_3m", "beta_1y",
            "idio_vol_3m", "max_ret_1m", "dist_52w", "volume_trend", "rsi", "max_dd_1y"]

# Extra price-based signals for the model search (DECISIONS MS2):
#   rev_1w           return over the last 5 trading days
#   skew_3m          skewness of daily returns, last 63 trading days
#   amihud_3m        Amihud illiquidity: mean |daily return| / dollar volume, last 63 days
#   mom_sector_rel   mom_12_1 minus the average mom_12_1 of the stock's sector that month
#   vol_ratio        vol_1m / vol_3m (is volatility rising or falling?)
EXTRA_FEATURES = ["rev_1w", "skew_3m", "amihud_3m", "mom_sector_rel", "vol_ratio"]
FEATURE_SETS = {"base": FEATURES, "extended": FEATURES + EXTRA_FEATURES}


def _max_drawdown_columns(prices: np.ndarray) -> np.ndarray:
    """Max drawdown of each column of a price window (rows = days)."""
    running_max = np.maximum.accumulate(prices, axis=0)
    return (1 - prices / running_max).max(axis=0)


def _rsi(prices: pd.DataFrame, days: int) -> pd.Series:
    """Simple-average RSI over the last ``days`` price changes of each column."""
    diff = prices.diff().iloc[-days:]
    gain = diff.clip(lower=0).mean()
    loss = (-diff.clip(upper=0)).mean()
    rs = gain / loss.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).fillna(100.0)    # no losses at all -> RSI 100


def features_at(prices: pd.DataFrame, volume: pd.DataFrame, prices_monthly: pd.DataFrame,
                bench: str, stocks: list[str], windows: dict, rsi_days: int) -> pd.DataFrame:
    """All features for every stock at one date, from data that ends at that date.

    Parameters
    ----------
    prices : daily adjusted prices up to and including t (stocks and ``bench``).
    volume : daily volume up to t.
    prices_monthly : month-end prices up to t (last row is t).
    bench : benchmark column (SPY) for beta and residual volatility.
    stocks : stock columns.
    windows : trading-day windows {"m1": 21, "m3": 63, "y1": 252}.
    rsi_days : RSI window.

    Returns
    -------
    DataFrame (stock x feature); NaN where the history is too short.

    Theorem: none (plan Section 4.1).
    """
    m1, m3, y1 = windows["m1"], windows["m3"], windows["y1"]
    r = prices.pct_change(fill_method=None)
    pm = prices_monthly[stocks]
    out = pd.DataFrame(index=stocks, columns=FEATURES, dtype=float)

    def monthly_ret(a: int, b: int) -> pd.Series:
        """Return from the month-end a months back to the month-end b months back."""
        if len(pm) <= a:
            return pd.Series(np.nan, index=stocks)
        return pm.iloc[-1 - b] / pm.iloc[-1 - a] - 1

    out["rev_1m"] = monthly_ret(1, 0)
    out["mom_3m"] = monthly_ret(3, 0)
    out["mom_6m"] = monthly_ret(6, 0)
    out["mom_12_1"] = monthly_ret(12, 1)

    if len(r) > y1:
        rs, rb = r[stocks], r[bench]
        out["vol_1m"] = rs.iloc[-m1:].std()
        out["vol_3m"] = rs.iloc[-m3:].std()
        ry, by = rs.iloc[-y1:], rb.iloc[-y1:]
        out["beta_1y"] = ry.apply(lambda c: np.cov(c, by)[0, 1]) / by.var()
        r3, b3 = rs.iloc[-m3:], rb.iloc[-m3:]
        beta3 = r3.apply(lambda c: np.cov(c, b3)[0, 1]) / b3.var()
        out["idio_vol_3m"] = (r3 - np.outer(b3, beta3)).std()
        out["max_ret_1m"] = rs.iloc[-m1:].max()
        p_y = prices[stocks].iloc[-y1:]
        out["dist_52w"] = p_y.iloc[-1] / p_y.max() - 1
        v = volume[stocks]
        out["volume_trend"] = np.log(v.iloc[-m1:].mean() / v.iloc[-y1:].mean())
        out["rsi"] = _rsi(prices[stocks], rsi_days)
        out["max_dd_1y"] = _max_drawdown_columns(p_y.values)
    return out


def extra_features_at(prices: pd.DataFrame, volume: pd.DataFrame, close: pd.DataFrame,
                      stocks: list[str], windows: dict) -> pd.DataFrame:
    """Extra signals (``EXTRA_FEATURES`` except the sector-relative one) at one date.

    Parameters
    ----------
    prices : daily adjusted prices up to and including t.
    volume : daily share volume up to t.
    close : daily unadjusted close up to t (for dollar volume).
    stocks : stock columns.
    windows : trading-day windows {"w1": 5, "m1": 21, "m3": 63, ...}.

    Returns
    -------
    DataFrame (stock x rev_1w, skew_3m, amihud_3m, vol_ratio); NaN if history is short.

    Theorem: none (DECISIONS MS2).
    """
    w1, m1, m3 = windows["w1"], windows["m1"], windows["m3"]
    out = pd.DataFrame(index=stocks, columns=["rev_1w", "skew_3m", "amihud_3m", "vol_ratio"],
                       dtype=float)
    if len(prices) <= m3:
        return out
    p = prices[stocks]
    r = p.pct_change(fill_method=None)
    out["rev_1w"] = p.iloc[-1] / p.iloc[-1 - w1] - 1
    out["skew_3m"] = r.iloc[-m3:].skew()
    dollar_vol = (close[stocks] * volume[stocks]).iloc[-m3:]
    out["amihud_3m"] = (r.iloc[-m3:].abs() / dollar_vol.where(dollar_vol > 0)).mean() * 1e9
    out["vol_ratio"] = r.iloc[-m1:].std() / r.iloc[-m3:].std()
    return out


def build_feature_panel(data: dict, stocks: list[str], bench: str, windows: dict,
                        rsi_days: int, feature_set: str = "base") -> pd.DataFrame:
    """Raw features at every month-end, each computed from data sliced to that date.

    Parameters
    ----------
    data : output of ``src.data.load_processed``.
    stocks : the universe.
    bench : benchmark column.
    windows, rsi_days : feature windows (config ``ml``).
    feature_set : "base" (the 13 ``FEATURES``) or "extended" (plus ``EXTRA_FEATURES``).

    Returns
    -------
    DataFrame indexed by (date, ticker) with the columns of ``FEATURE_SETS[feature_set]``;
    dates with any missing feature (the first year) are dropped.

    Theorem: none (plan Section 4.1; no look-ahead by construction, tested).
    """
    prices, volume, pm = data["prices"], data["volume"], data["prices_monthly"]
    extended = feature_set == "extended"
    if feature_set not in FEATURE_SETS:
        raise ValueError(f"Unknown feature set {feature_set!r}")
    sectors = data["sectors"]["sector"].reindex(stocks) if extended else None
    frames = {}
    for t in pm.index:
        f = features_at(prices.loc[:t], volume.loc[:t], pm.loc[:t], bench, stocks,
                        windows, rsi_days)
        if extended:
            f = f.join(extra_features_at(prices.loc[:t], volume.loc[:t], data["close"].loc[:t],
                                         stocks, windows))
            f["mom_sector_rel"] = f["mom_12_1"] - f["mom_12_1"].groupby(sectors).transform("mean")
            f = f[FEATURE_SETS["extended"]]
        if not f.isna().any().any():
            frames[t] = f
    panel = pd.concat(frames, names=["date", "ticker"])
    return panel.astype(float)


def rank_normalize(panel: pd.DataFrame) -> pd.DataFrame:
    """Cross-sectional rank of each feature at each date, scaled to [−0.5, 0.5].

    Ranks remove outliers and level shifts over time (plan Section 4.1), so a model
    sees only where a stock stands relative to the others that month.

    Parameters
    ----------
    panel : raw features indexed by (date, ticker).

    Returns
    -------
    Same shape; ties get their average rank.

    Theorem: none.
    """
    ranks = panel.groupby(level="date").rank(method="average")
    n = panel.groupby(level="date").transform("count")
    return (ranks - 1) / (n - 1) - 0.5


def add_targets(panel: pd.DataFrame, returns_monthly: pd.DataFrame,
                rf_monthly: pd.Series) -> pd.DataFrame:
    """Attach the next-month excess return and the date it is realized.

    For a row dated t, ``target`` = r_{i,s} − r_f,s and ``target_date`` = s, the next
    month-end. Rows whose next month is not in the data get NaN targets (they can
    still be forecast, never trained on).

    Parameters
    ----------
    panel : features indexed by (date, ticker).
    returns_monthly : monthly returns (row s = month ending s).
    rf_monthly : risk-free return for the month ending s.

    Returns
    -------
    ``panel`` with two extra columns, ``target`` and ``target_date``.

    Theorem: none (plan Section 4.1, target definition).
    """
    me = returns_monthly.index
    nxt = pd.Series(list(me[1:]) + [pd.NaT], index=me)
    excess = returns_monthly.sub(rf_monthly.reindex(me), axis=0)
    dates = panel.index.get_level_values("date")
    tickers = panel.index.get_level_values("ticker")
    target_date = nxt.reindex(dates).values
    target = [excess.at[s, k] if pd.notna(s) else np.nan
              for s, k in zip(target_date, tickers)]
    out = panel.copy()
    out["target"] = np.asarray(target, dtype=float)
    out["target_date"] = pd.DatetimeIndex(target_date)
    return out

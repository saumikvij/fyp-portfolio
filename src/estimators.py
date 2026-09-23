"""Point-in-time estimators of the inputs to mean-variance: Σ and historical μ.

Every function taking a date t uses only rows dated <= t (plan Section 6.1).
Units: monthly, because the portfolio is rebalanced and held monthly.
"""

from __future__ import annotations

import pandas as pd


def sample_covariance(returns_daily: pd.DataFrame, t: pd.Timestamp, window: int,
                      days_per_month: float) -> pd.DataFrame:
    """Sample covariance of daily returns over a trailing window, scaled to monthly.

    Σ_monthly = days_per_month × Σ_daily, which assumes daily returns are
    uncorrelated over time (plan Section 6.4).

    Parameters
    ----------
    returns_daily : daily returns (date x ticker).
    t : estimation date; only rows dated <= t are used.
    window : number of trailing trading days (e.g. 252).
    days_per_month : trading days per month (252 / 12 = 21).

    Returns
    -------
    DataFrame (ticker x ticker), the monthly covariance matrix Σ.

    Raises
    ------
    ValueError if fewer than ``window`` days are available up to t.

    Theorem: none directly; Σ ≻ 0 is the assumption of T1–T5.
    """
    hist = returns_daily.loc[:t]
    if len(hist) < window:
        raise ValueError(f"Only {len(hist)} days before {t.date()}, need {window}")
    return hist.iloc[-window:].cov() * days_per_month


def historical_mean(returns_monthly: pd.DataFrame, t: pd.Timestamp,
                    window: int | None = None) -> pd.Series:
    """Historical mean of monthly returns: the classical Markowitz μ (strategy S4).

    Parameters
    ----------
    returns_monthly : monthly returns; row s is the return over the month ending s.
    t : estimation date; only months ending <= t are used.
    window : number of trailing months, or None for an expanding window.

    Returns
    -------
    Series (ticker), the estimated expected monthly return μ.

    Raises
    ------
    ValueError if no month ends on or before t.

    Theorem: none.
    """
    hist = returns_monthly.loc[:t]
    if window is not None:
        hist = hist.iloc[-window:]
    if hist.empty:
        raise ValueError(f"No monthly returns on or before {t.date()}")
    return hist.mean()

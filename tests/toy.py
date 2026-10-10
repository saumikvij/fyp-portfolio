"""Small synthetic dataset in the same format as ``src.data.load_processed``."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.data import daily_returns, month_end_prices, monthly_returns


def make_toy_data(n_stocks: int = 6, n_days: int = 1300, seed: int = 0) -> dict:
    """Prices with a common market factor, plus volume, monthly tables and rf.

    Returns
    -------
    dict with prices, close, shares_outstanding, volume, returns_daily,
    prices_monthly, returns_monthly, rf_monthly (DataFrame with column ``rf``) and
    sectors; stocks are S0..S{n-1},
    the benchmark is SPY.
    """
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2015-01-01", periods=n_days, name="Date")
    stocks = [f"S{i}" for i in range(n_stocks)]
    mkt = rng.normal(0.0004, 0.01, n_days)
    betas = rng.uniform(0.5, 1.5, n_stocks)
    rets = mkt[:, None] * betas + rng.normal(0.0002, 0.012, (n_days, n_stocks))
    prices = pd.DataFrame(100 * np.cumprod(1 + np.column_stack([rets, mkt]), axis=0),
                          index=idx, columns=stocks + ["SPY"])
    volume = pd.DataFrame(rng.lognormal(14, 0.4, (n_days, n_stocks + 1)), index=idx,
                          columns=prices.columns)
    pm = month_end_prices(prices)
    rm = monthly_returns(pm)
    shares = pd.DataFrame({"shares_outstanding": rng.integers(1, 20, n_stocks + 1) * 1e8},
                          index=stocks + ["SPY"])
    return {
        "prices": prices, "close": prices * 1.0, "shares_outstanding": shares,
        "volume": volume, "returns_daily": daily_returns(prices),
        "prices_monthly": pm, "returns_monthly": rm,
        "rf_monthly": pd.DataFrame({"rf": 0.001}, index=rm.index),
        "sectors": pd.DataFrame({"sector": ["X", "Y"] * (n_stocks // 2)}, index=stocks),
    }

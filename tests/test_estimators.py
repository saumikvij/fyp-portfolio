"""Phase 1 tests: covariance and historical-mean estimators (values and no look-ahead)."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.data import load_config, load_processed
from src.estimators import historical_mean, sample_covariance
from tests.pit import assert_uses_only_past

CFG = load_config()
DAYS_PER_MONTH = CFG["data"]["trading_days_per_year"] / 12


@pytest.fixture
def daily():
    idx = pd.bdate_range("2020-01-01", periods=400, name="Date")
    rng = np.random.default_rng(3)
    return pd.DataFrame(rng.normal(0, 0.01, (400, 4)), index=idx, columns=list("ABCD"))


@pytest.fixture
def monthly():
    idx = pd.date_range("2015-01-31", periods=60, freq="ME")
    rng = np.random.default_rng(4)
    return pd.DataFrame(rng.normal(0.01, 0.05, (60, 4)), index=idx, columns=list("ABCD"))


def test_covariance_values(daily):
    t = daily.index[300]
    got = sample_covariance(daily, t, 252, DAYS_PER_MONTH)
    window = daily.loc[:t].iloc[-252:].values
    np.testing.assert_allclose(got.values, np.cov(window, rowvar=False) * DAYS_PER_MONTH)
    assert list(got.index) == list(got.columns) == list("ABCD")


def test_covariance_needs_full_window(daily):
    with pytest.raises(ValueError, match="need 252"):
        sample_covariance(daily, daily.index[100], 252, DAYS_PER_MONTH)


@pytest.mark.parametrize("i", [260, 330])
def test_covariance_no_lookahead(daily, i):
    assert_uses_only_past(lambda d, t: sample_covariance(d, t, 252, DAYS_PER_MONTH),
                          daily, daily.index[i])


def test_historical_mean_values(monthly):
    t = monthly.index[29]
    pd.testing.assert_series_equal(historical_mean(monthly, t), monthly.iloc[:30].mean())
    pd.testing.assert_series_equal(historical_mean(monthly, t, window=12),
                                   monthly.iloc[18:30].mean())


def test_historical_mean_mid_month_date_uses_last_complete_month(monthly):
    t = monthly.index[29] + pd.Timedelta(days=10)
    pd.testing.assert_series_equal(historical_mean(monthly, t), monthly.iloc[:30].mean())


@pytest.mark.parametrize("window", [None, 12])
def test_historical_mean_no_lookahead(monthly, window):
    assert_uses_only_past(lambda d, t: historical_mean(d, t, window), monthly,
                          monthly.index[40])


def test_historical_mean_needs_data(monthly):
    with pytest.raises(ValueError):
        historical_mean(monthly, pd.Timestamp("2010-01-01"))


# --------------------------------------------------------------------------- real data

HAVE_DATA = (Path(CFG["data"]["processed_dir"]) / "prices.parquet").exists()


@pytest.mark.data
@pytest.mark.skipif(not HAVE_DATA, reason="run `python -m src.data` first")
def test_real_covariance_is_positive_definite():
    """Σ ≻ 0 is the standing assumption of T1–T5; check it on the real data."""
    d = load_processed(CFG)
    stocks = list(CFG["universe"])
    rets = d["returns_daily"][stocks]
    for t in d["returns_monthly"].index[12::12]:
        S = sample_covariance(rets, t, CFG["backtest"]["cov_window_days"], DAYS_PER_MONTH)
        assert np.linalg.eigvalsh(S.values).min() > 0, t

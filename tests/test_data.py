"""Phase 0 tests: config, cleaning rules, point-in-time checks, processed data."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.data import (clean_prices, daily_returns, flag_extreme_returns, load_config,
                      load_processed, month_end_dates, month_end_prices, monthly_returns,
                      risk_free_rates)
from tests.pit import assert_point_in_time

CFG = load_config()


@pytest.fixture
def calendar():
    # Business days across a few month boundaries (Jan-Apr 2020).
    return pd.bdate_range("2020-01-02", "2020-04-30", name="Date")


@pytest.fixture
def prices(calendar):
    rng = np.random.default_rng(1)
    rets = rng.normal(0.0005, 0.01, size=(len(calendar), 3))
    return pd.DataFrame(100 * np.cumprod(1 + rets, axis=0), index=calendar,
                        columns=["A", "B", "C"])


# --------------------------------------------------------------------------- config


def test_universe_matches_plan():
    sectors = pd.Series(CFG["universe"])
    assert 40 <= len(sectors) <= 60
    assert sectors.index.is_unique
    assert sectors.nunique() == 11                       # all GICS sectors
    assert sectors.value_counts().between(4, 6).all()    # 4-6 per sector
    assert CFG["data"]["benchmark"] not in sectors.index


def test_config_dates_ordered():
    d = CFG["data"]
    start, oos, end = (pd.Timestamp(x) for x in
                       (d["start"], CFG["backtest"]["oos_start"], d["end"]))
    assert start < oos < end
    assert end == end + pd.offsets.MonthEnd(0), "end must be a calendar month-end"


# --------------------------------------------------------------------------- cleaning


def test_clean_fills_short_gap_with_last_price(prices, calendar):
    raw = prices.copy()
    raw.iloc[10:12, 0] = np.nan
    out = clean_prices(raw, calendar, max_ffill_days=5, max_missing_frac=0.05)
    assert (out.iloc[10:12, 0] == prices.iloc[9, 0]).all()
    assert not out.isna().any().any()


def test_clean_rejects_long_gap(prices, calendar):
    raw = prices.copy()
    raw.iloc[10:17, 1] = np.nan
    with pytest.raises(ValueError, match="Gaps longer"):
        clean_prices(raw, calendar, max_ffill_days=5, max_missing_frac=0.5)


def test_clean_rejects_sparse_ticker(prices, calendar):
    raw = prices.copy()
    raw.iloc[5:50:3, 2] = np.nan
    with pytest.raises(ValueError, match="Too many missing"):
        clean_prices(raw, calendar, max_ffill_days=5, max_missing_frac=0.01)


def test_clean_rejects_ticker_starting_late(prices, calendar):
    raw = prices.copy()
    raw.iloc[0, 0] = np.nan
    with pytest.raises(ValueError, match="first trading day"):
        clean_prices(raw, calendar, max_ffill_days=5, max_missing_frac=0.05)


def test_clean_treats_nonpositive_price_as_missing(prices, calendar):
    raw = prices.copy()
    raw.iloc[20, 0] = 0.0
    out = clean_prices(raw, calendar, max_ffill_days=5, max_missing_frac=0.05)
    assert out.iloc[20, 0] == prices.iloc[19, 0]


def test_clean_aligns_to_calendar(prices, calendar):
    extra = prices.copy()
    extra.loc[pd.Timestamp("2020-01-04")] = 1.0          # a Saturday: not a trading day
    out = clean_prices(extra.sort_index(), calendar, 5, 0.05)
    assert out.index.equals(calendar)


# --------------------------------------------------------------------------- returns


def test_daily_returns_values(prices):
    r = daily_returns(prices)
    assert r.index[0] == prices.index[1]
    np.testing.assert_allclose(r.iloc[0], prices.iloc[1] / prices.iloc[0] - 1)


def test_month_end_dates_are_last_trading_days():
    # 29 Feb 2020 is a Saturday, so February's last trading day is the 28th.
    cal = pd.bdate_range("2020-01-02", "2020-03-31")
    me = month_end_dates(cal)
    assert list(me) == [pd.Timestamp(x) for x in ("2020-01-31", "2020-02-28", "2020-03-31")]


def test_monthly_return_equals_compounded_daily(prices):
    r_m = monthly_returns(month_end_prices(prices))
    r_d = daily_returns(prices)
    for t in r_m.index:
        prev = r_m.index[r_m.index < t]
        start = prev[-1] if len(prev) else month_end_dates(prices.index)[0]
        window = r_d[(r_d.index > start) & (r_d.index <= t)]
        np.testing.assert_allclose(r_m.loc[t], (1 + window).prod() - 1)


def test_flag_extreme_returns():
    r = pd.DataFrame({"A": [0.01, 0.5], "B": [-0.45, 0.0]},
                     index=pd.to_datetime(["2020-01-02", "2020-01-03"]))
    out = flag_extreme_returns(r, 0.40)
    assert set(zip(out.Ticker, out.ret)) == {("A", 0.5), ("B", -0.45)}


# --------------------------------------------------------------------------- risk-free


def test_risk_free_conversion(calendar):
    irx = pd.Series(4.0, index=calendar)                  # 4% per year
    rf_d, rf_m = risk_free_rates(irx, calendar, 5, 252)
    np.testing.assert_allclose(rf_d["daily"], 1.04 ** (1 / 252) - 1)
    np.testing.assert_allclose(rf_m, 1.04 ** (1 / 12) - 1)
    assert rf_m.index.equals(month_end_dates(calendar)[1:])


def test_monthly_rf_uses_previous_month_end_yield(calendar):
    irx = pd.Series(np.arange(len(calendar), dtype=float) / 10, index=calendar)
    _, rf_m = risk_free_rates(irx, calendar, 5, 252)
    me = month_end_dates(calendar)
    for prev, t in zip(me[:-1], me[1:]):
        np.testing.assert_allclose(rf_m.loc[t], (1 + irx.loc[prev] / 100) ** (1 / 12) - 1)


# --------------------------------------------------------------------------- look-ahead
# Section 6.7: outputs dated <= t must not change when data after t changes.


@pytest.mark.parametrize("t", ["2020-01-31", "2020-02-14", "2020-03-31"])
def test_no_lookahead_cleaning_and_returns(prices, calendar, t):
    t = pd.Timestamp(t)
    gappy = prices.copy()
    gappy.iloc[[15, 16, 40, 41, 42], 0] = np.nan          # forward-fill must use past only

    def pipeline(px):
        clean = clean_prices(px, calendar, 5, 0.1)
        return clean, daily_returns(clean)

    assert_point_in_time(pipeline, gappy, t)


@pytest.mark.parametrize("t", ["2020-02-28", "2020-03-13", "2020-03-31"])
def test_no_lookahead_monthly(prices, t):
    t = pd.Timestamp(t)
    assert_point_in_time(lambda px: monthly_returns(month_end_prices(px)), prices, t)


@pytest.mark.parametrize("t", ["2020-02-28", "2020-03-16", "2020-03-31"])
def test_no_lookahead_risk_free(calendar, t):
    irx = pd.Series(np.linspace(1.0, 2.0, len(calendar)), index=calendar)
    assert_point_in_time(lambda s: risk_free_rates(s, calendar, 5, 252), irx,
                         pd.Timestamp(t))


# --------------------------------------------------------------------------- processed data

HAVE_DATA = (Path(CFG["data"]["processed_dir"]) / "prices.parquet").exists()
needs_data = pytest.mark.skipif(not HAVE_DATA, reason="run `python -m src.data` first")


@pytest.fixture(scope="module")
def data():
    return load_processed(CFG)


@pytest.mark.data
@needs_data
def test_processed_columns_and_dates(data):
    expected = list(CFG["universe"]) + [CFG["data"]["benchmark"]]
    start, end = pd.Timestamp(CFG["data"]["start"]), pd.Timestamp(CFG["data"]["end"])
    for key in ("prices", "volume", "returns_daily", "prices_monthly", "returns_monthly"):
        df = data[key]
        assert list(df.columns) == expected, key
        assert df.index.is_monotonic_increasing and df.index.is_unique, key
        assert df.index[0] >= start and df.index[-1] <= end, key
        assert not df.isna().any().any(), key
    assert (data["prices"] > 0).all().all()
    assert (data["volume"] >= 0).all().all()


@pytest.mark.data
@needs_data
def test_processed_sample_is_complete(data):
    # Whole sample available: roughly 252 trading days per year.
    p = data["prices"]
    years = (p.index[-1] - p.index[0]).days / 365.25
    assert abs(len(p) / years - 252) < 3
    assert data["prices_monthly"].index[-1] == pd.Timestamp(CFG["data"]["end"])


@pytest.mark.data
@needs_data
def test_processed_monthly_consistent_with_daily(data):
    rm = data["returns_monthly"]
    rd = data["returns_daily"]
    compounded = (1 + rd).groupby(rd.index.to_period("M")).prod() - 1
    # The first month has no previous month-end, so it has no monthly return.
    np.testing.assert_allclose(compounded.iloc[1:].values, rm.values, atol=1e-10)


@pytest.mark.data
@needs_data
def test_processed_risk_free(data):
    rf = data["rf_monthly"]["rf"]
    assert rf.index.equals(data["returns_monthly"].index)
    assert rf.between(-0.001, 0.01).all()            # T-bill: roughly 0-12% a year
    assert data["rf_daily"].index.equals(data["returns_daily"].index)


@pytest.mark.data
@needs_data
def test_processed_sectors_and_shares(data):
    assert data["sectors"]["sector"].to_dict() == CFG["universe"]
    shares = data["shares_outstanding"]["shares_outstanding"]
    assert list(shares.index) == list(CFG["universe"])
    assert (shares > 0).all()
    # All-class count can never be below the single-class count (e.g. GOOGL A vs A+B+C).
    assert (shares >= data["shares_outstanding"]["shares_primary"]).all()

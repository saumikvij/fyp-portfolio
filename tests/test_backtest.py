"""Phase 2 tests: walk-forward engine (returns, drift, turnover, costs) and no look-ahead."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.backtest import (History, buy_and_hold, clean_weights, equal_weight,
                          markowitz_strategy, min_variance_strategy, rebalance_dates,
                          run_backtest)
from src.data import (daily_returns, load_config, load_processed, month_end_prices,
                      monthly_returns)
from src.estimators import condition_number, sample_covariance
from tests.pit import perturb_after

CFG = load_config()
WTOL = CFG["optimization"]["weight_tol"]


def make_data(seed: int = 0, n_days: int = 400):
    """Toy dataset in the same format as ``load_processed`` (3 stocks + SPY)."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2019-01-01", periods=n_days, name="Date")
    cols = ["A", "B", "C", "SPY"]
    prices = pd.DataFrame(100 * np.cumprod(1 + rng.normal(0.0005, 0.012, (n_days, 4)), 0),
                          index=idx, columns=cols)
    rm = monthly_returns(month_end_prices(prices))
    return {
        "returns_daily": daily_returns(prices),
        "returns_monthly": rm,
        "rf_monthly": pd.DataFrame({"rf": 0.001}, index=rm.index),
        "sectors": pd.DataFrame({"sector": ["X", "X", "Y"]}, index=["A", "B", "C"]),
    }


def fixed(weights: dict):
    w = pd.Series(weights)
    return lambda hist: w


# --------------------------------------------------------------------------- dates


def test_rebalance_dates():
    me = pd.DatetimeIndex(["2015-11-30", "2015-12-31", "2016-01-29", "2016-02-29"])
    d = rebalance_dates(me, "2016-01-01")
    # Decide at the last month-end of 2015; the last month-end is only for measuring.
    assert list(d) == [pd.Timestamp("2015-12-31"), pd.Timestamp("2016-01-29")]


# --------------------------------------------------------------------------- mechanics


def test_returns_drift_and_turnover_by_hand():
    data = make_data()
    rm = data["returns_monthly"]
    dates = rm.index[2:5]
    res = run_backtest("fixed", fixed({"A": 0.5, "B": 0.5}), data, dates, WTOL)
    s = rm.index[3:6]
    np.testing.assert_allclose(res.gross.values, 0.5 * rm.loc[s, "A"] + 0.5 * rm.loc[s, "B"])
    assert res.turnover.iloc[0] == pytest.approx(1.0)       # first trade from cash
    # Month 2: weights drifted to 0.5(1+r_A)/(1+g), rebalanced back to 0.5 / 0.5.
    rA, rB = rm.loc[s[0], "A"], rm.loc[s[0], "B"]
    g = 0.5 * rA + 0.5 * rB
    expected = abs(0.5 - 0.5 * (1 + rA) / (1 + g)) + abs(0.5 - 0.5 * (1 + rB) / (1 + g))
    assert res.turnover.iloc[1] == pytest.approx(expected)


def test_costs():
    data = make_data()
    res = run_backtest("ew", equal_weight(["A", "B", "C"]), data,
                       data["returns_monthly"].index[2:8], WTOL)
    pd.testing.assert_series_equal(res.net(0), res.gross)
    c = 25 / 1e4
    np.testing.assert_allclose(res.net(25), (1 - c * res.turnover) * (1 + res.gross) - 1)
    assert (res.net(25) <= res.gross + 1e-15).all()


def test_daily_path_compounds_to_monthly_return():
    data = make_data()
    res = run_backtest("ew", equal_weight(["A", "B", "C"]), data,
                       data["returns_monthly"].index[2:8], WTOL)
    for cost in (0, 25):
        d = res.daily_net(cost)
        monthly_from_daily = (1 + d).groupby(d.index.to_period("M")).prod() - 1
        np.testing.assert_allclose(monthly_from_daily.values, res.net(cost).values)


def test_buy_and_hold_spy_matches_spy_and_never_trades():
    data = make_data()
    res = run_backtest("spy", buy_and_hold("SPY"), data, data["returns_monthly"].index[2:9],
                       WTOL)
    np.testing.assert_allclose(res.gross.values, data["returns_monthly"].loc[
        res.gross.index, "SPY"].values)
    assert res.turnover.iloc[0] == pytest.approx(1.0)
    np.testing.assert_allclose(res.turnover.iloc[1:], 0.0, atol=1e-12)


def test_invalid_weights_raise():
    data = make_data()
    with pytest.raises(ValueError, match="invalid weights"):
        run_backtest("bad", fixed({"A": 0.7}), data, data["returns_monthly"].index[2:4], WTOL)


def test_clean_weights():
    w = clean_weights(pd.Series({"A": 0.6, "B": 0.4 - 1e-9, "C": 1e-9}), 1e-5)
    assert w["C"] == 0.0 and w.sum() == pytest.approx(1.0)


# --------------------------------------------------------------------------- look-ahead


def test_strategy_never_sees_future_data():
    data = make_data()
    seen = []

    def spy_strategy(hist: History) -> pd.Series:
        for df in (hist.returns_daily, hist.returns_monthly, hist.rf_monthly):
            assert df.index.max() <= hist.t
        seen.append(hist.t)
        return pd.Series({"A": 1.0})

    dates = data["returns_monthly"].index[2:10]
    run_backtest("spy", spy_strategy, data, dates, WTOL)
    assert seen == list(dates)


@pytest.mark.parametrize("cut", [4, 7])
def test_backtest_results_unchanged_by_future_data(cut):
    """Scramble every table after month-end t: all results dated <= t are unchanged."""
    data = make_data()
    stocks = ["A", "B", "C"]
    cfg = {**CFG, "backtest": {**CFG["backtest"], "cov_window_days": 60, "w_max": 0.6}}
    dates = data["returns_monthly"].index[3:12]
    t = data["returns_monthly"].index[cut]
    moved = {k: (perturb_after(v, t) if k != "sectors" else v) for k, v in data.items()}
    for strat in (markowitz_strategy(stocks, cfg, 5.0), min_variance_strategy(stocks, cfg)):
        a = run_backtest("a", strat, data, dates, WTOL)
        b = run_backtest("b", strat, moved, dates, WTOL)
        pd.testing.assert_frame_equal(a.weights.loc[:t], b.weights.loc[:t])
        np.testing.assert_array_equal(a.net(10).loc[:t], b.net(10).loc[:t])
        np.testing.assert_array_equal(a.daily_net(10).loc[:t], b.daily_net(10).loc[:t])
        assert not a.net(10).loc[t:].iloc[1:].equals(b.net(10).loc[t:].iloc[1:])


# --------------------------------------------------------------------------- real data

HAVE_DATA = (Path(CFG["data"]["processed_dir"]) / "prices.parquet").exists()
needs_data = pytest.mark.skipif(not HAVE_DATA, reason="run `python -m src.data` first")


@pytest.mark.data
@needs_data
def test_real_backtest_constraints_and_dates():
    data = load_processed(CFG)
    stocks = list(CFG["universe"])
    dates = rebalance_dates(data["returns_monthly"].index, CFG["backtest"]["oos_start"])[:6]
    res = run_backtest("S4", markowitz_strategy(stocks, CFG, 5.0), data, dates, WTOL)
    assert res.gross.index[0] == pd.Timestamp("2016-01-29")   # first out-of-sample month
    assert (res.weights >= 0).all().all()
    assert (res.weights <= CFG["backtest"]["w_max"] + WTOL).all().all()
    np.testing.assert_allclose(res.weights.sum(axis=1), 1.0)


@pytest.mark.data
@needs_data
def test_real_condition_number_matches_numpy():
    data = load_processed(CFG)
    stocks = list(CFG["universe"])
    S = sample_covariance(data["returns_daily"][stocks], pd.Timestamp("2020-03-31"),
                          CFG["backtest"]["cov_window_days"], 21)
    assert condition_number(S) == pytest.approx(np.linalg.cond(S.values), rel=1e-8)

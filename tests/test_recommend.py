"""Phase 5 tests: the recommendation tool (amount + risk → constrained allocation), its
match with the backtested strategy, reproducibility and no look-ahead."""

import copy
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.backtest import (black_litterman_strategy, history_at, markowitz_strategy,
                          rebalance_dates, run_backtest, strategy_constraints)
from src.data import load_config, load_processed
from src.recommend import (UserRequest, check_constraints, forecasts_up_to, recommend,
                           resolve_as_of, sector_bounds, validate_request, whole_shares)
from tests.pit import perturb_after

CFG = load_config()
W_MAX = CFG["backtest"]["w_max"]
S_MAX = CFG["optimization"]["sector_max"]
TOL = CFG["optimization"]["weight_tol"]


# --------------------------------------------------------------------------- unit tests


def test_sector_rule_comes_from_config():
    """Every sector gets (0, sector_max); the user sets nothing (A4)."""
    sectors = pd.Series({"A": "X", "B": "Y", "C": "Y", "D": "Z"})
    assert S_MAX == 0.25
    assert sector_bounds(sectors, CFG) == {"X": (0.0, 0.25), "Y": (0.0, 0.25), "Z": (0.0, 0.25)}
    off = copy.deepcopy(CFG)
    off["optimization"]["sector_max"] = None
    assert sector_bounds(sectors, off) == {}


@pytest.mark.parametrize("req, msg", [
    (UserRequest(0, "medium"), "positive"),
    (UserRequest(-500, "low"), "positive"),
    (UserRequest(1000, "extreme"), "Risk tolerance"),
])
def test_invalid_requests_are_rejected(req, msg):
    sectors = pd.Series({f"S{i}": f"Sec{i % 5}" for i in range(20)})
    with pytest.raises(ValueError, match=msg):
        validate_request(req, sectors, CFG)


def test_infeasible_sector_rule_is_rejected():
    """5 sectors × a 15% cap leave room for 75% only: no portfolio exists."""
    sectors = pd.Series({f"S{i}": f"Sec{i % 5}" for i in range(20)})
    tight = copy.deepcopy(CFG)
    tight["optimization"]["sector_max"] = 0.15
    with pytest.raises(ValueError, match="room for only 75%"):
        validate_request(UserRequest(1000, "medium"), sectors, tight)
    validate_request(UserRequest(1000, "medium"), sectors, CFG)        # 25%: fine


def test_whole_shares_rounding_by_hand():
    target = pd.Series({"A": 0.5, "B": 0.3, "C": 0.2})
    prices = pd.Series({"A": 30.0, "B": 70.0, "C": 45.0})
    sectors = pd.Series({"A": "X", "B": "X", "C": "Y"})
    out = whole_shares(target, prices, 1000, 1.0, sectors, {})
    # Floors: A 16 ($480), B 4 ($280), C 4 ($180); cash $60. Gaps: A 20, B 20, C 20.
    # One more A ($30) brings A to 2 dollars from target (gap 20 → |−10|): improvement 10.
    # B ($70) and C ($45) would overshoot by more than they fill → not bought.
    assert out.loc["A", "shares"] == 17 and out.loc["B", "shares"] == 4
    assert out.loc["C", "shares"] == 4
    assert out["value"].sum() == pytest.approx(1000 - 30)


def test_whole_shares_respects_stock_and_sector_caps():
    rng = np.random.default_rng(0)
    tickers = [f"S{i}" for i in range(30)]
    sectors = pd.Series(["X", "Y", "Z"] * 10, index=tickers)
    # A valid target (as the optimizer produces): sector X exactly at its 30% cap,
    # every stock at or below the 5% cap.
    target = pd.Series(np.where(sectors == "X", 0.03, 0.035), index=tickers)
    prices = pd.Series(rng.uniform(5, 400, 30), index=tickers)
    bounds = {"X": (0.0, 0.3)}
    assert target.sum() == pytest.approx(1.0)
    for amount in (2_000, 25_000, 1_000_000):
        out = whole_shares(target, prices, amount, 0.05, sectors, bounds)
        w = out["weight"]
        assert (w <= 0.05 + 1e-12).all()
        assert w[out["sector"] == "X"].sum() <= 0.3 + 1e-12
        assert out["value"].sum() <= amount + 1e-9
        assert (out["shares"] == out["shares"].round()).all() and (out["shares"] > 0).all()


def test_check_constraints():
    sectors = pd.Series({"A": "X", "B": "X", "C": "Y"})
    w = pd.Series({"A": 0.1, "B": 0.0, "C": 0.9})
    c = check_constraints(w, sectors, 0.5, {"X": (0.0, 0.25), "Y": (0.0, 0.25)}, 1e-9)
    assert c["long_only"] and c["budget"]
    assert c["sector_max"] is False and c["cap"] is False
    assert set(c) == {"long_only", "cap", "sector_max", "budget"}


# --------------------------------------------------------------------------- real data

HAVE_DATA = (Path(CFG["data"]["processed_dir"]) / "close.parquet").exists()
needs_data = pytest.mark.skipif(not HAVE_DATA, reason="run `python -m src.data` first")
P3 = Path(CFG["phase3"]["forecast_dir"]) / "forecasts_elastic_net.parquet"


@pytest.fixture(scope="module")
def data():
    return load_processed(CFG)


@pytest.fixture(scope="module")
def fc_latest(data):
    return forecasts_up_to(data, CFG, resolve_as_of(data, None))


EXAMPLE = UserRequest(10_000, "medium")


def by_sector(w: pd.Series, data) -> pd.Series:
    return w.groupby(data["sectors"]["sector"].reindex(w.index)).sum()


@pytest.mark.data
@needs_data
def test_worked_example_satisfies_every_constraint(data, fc_latest):
    """The walkthrough example ($10,000, medium risk): the allocation satisfies every
    constraint, for the target weights and for the whole-share holdings."""
    rec = recommend(EXAMPLE, CFG, data, forecasts=fc_latest)
    assert all(rec.checks_target.values()), rec.checks_target
    assert all(rec.checks_holdings.values()), rec.checks_holdings
    t = rec.target
    assert t.sum() == pytest.approx(1.0) and (t >= 0).all() and (t <= W_MAX + TOL).all()
    assert (by_sector(t, data) <= S_MAX + TOL).all()
    h = rec.holdings
    assert (h["shares"] > 0).all() and (h["shares"] == h["shares"].round()).all()
    assert h["value"].sum() <= EXAMPLE.amount + 1e-6
    assert (h.groupby("sector")["value"].sum() <= S_MAX * EXAMPLE.amount + 1e-6).all()
    assert rec.summary["cash"] == pytest.approx(EXAMPLE.amount - h["value"].sum())


@pytest.mark.data
@needs_data
@pytest.mark.parametrize("as_of", ["2019-03-29", "2022-06-30", None])
def test_tool_target_equals_backtest_weights(data, as_of):
    """The tool runs exactly the evaluated strategy: its target weights at t equal the
    weights the S6 headline backtest holds from t (same forecasts, Σ, prior, Ω,
    constraints and dust cleaning)."""
    if not P3.exists():
        pytest.skip("run notebooks.phase3_forecasting first")
    t = resolve_as_of(data, as_of)
    stocks = list(CFG["universe"])
    rec = recommend(EXAMPLE, CFG, data, as_of=t)
    fc = pd.read_parquet(P3)
    if t not in fc.index:
        fc, _ = forecasts_up_to(data, CFG, t)
    gamma = CFG["risk_tolerance"]["medium"]
    strat = black_litterman_strategy(fc, None, stocks, CFG, gamma, "A",
                                     CFG["black_litterman"]["tau"])
    me = data["returns_monthly"].index
    if t == me[-1]:      # no next month to hold over: compare the strategy's own weights
        w = strat(history_at(data, t)).reindex(stocks).fillna(0.0)
        w = w.where(w.abs() >= TOL, 0.0)
        bt = w / w.sum()
    else:
        bt = run_backtest("S6", strat, data, pd.DatetimeIndex([t]), TOL).weights.loc[t]
    np.testing.assert_allclose(rec.target.reindex(stocks).values, bt.reindex(stocks).values,
                               atol=1e-9)
    assert strategy_constraints(history_at(data, t), CFG).sector_bounds == \
        sector_bounds(data["sectors"]["sector"], CFG)


@pytest.mark.data
@needs_data
def test_sector_rule_binds_in_the_backtest(data):
    """Not vacuous: without the rule the high-risk Markowitz portfolio puts more than
    25% in one sector in most months; with it, no month does."""
    stocks = list(CFG["universe"])
    dates = rebalance_dates(data["returns_monthly"].index, CFG["backtest"]["oos_start"])[:24]
    off = copy.deepcopy(CFG)
    off["optimization"]["sector_max"] = None
    sec = data["sectors"]["sector"]
    w_on = run_backtest("on", markowitz_strategy(stocks, CFG, 2.0), data, dates, TOL).weights
    w_off = run_backtest("off", markowitz_strategy(stocks, off, 2.0), data, dates, TOL).weights
    on, offs = w_on.T.groupby(sec).sum().T, w_off.T.groupby(sec).sum().T
    assert (on <= S_MAX + TOL).all().all()
    assert (offs.max(axis=1) > S_MAX + 0.01).sum() >= 12


@pytest.mark.data
@needs_data
def test_higher_risk_tolerance_gives_higher_volatility(data, fc_latest):
    vols = [recommend(UserRequest(10_000, r), CFG, data, forecasts=fc_latest)
            .summary["target_vol_pa"] for r in ("low", "medium", "high")]
    assert vols[0] < vols[1] < vols[2]


@pytest.mark.data
@needs_data
def test_amount_scales_shares_not_target_weights(data, fc_latest):
    small = recommend(UserRequest(10_000, "medium"), CFG, data, forecasts=fc_latest)
    large = recommend(UserRequest(1_000_000, "medium"), CFG, data, forecasts=fc_latest)
    pd.testing.assert_series_equal(small.target, large.target)
    # A larger amount tracks the target more closely (less lost to whole-share rounding).
    assert large.summary["max_abs_weight_gap"] < small.summary["max_abs_weight_gap"]
    assert large.summary["cash"] / 1_000_000 < small.summary["cash"] / 10_000


@pytest.mark.data
@needs_data
def test_recommendation_is_reproducible(data):
    a = recommend(EXAMPLE, CFG, data)
    b = recommend(EXAMPLE, CFG, data)
    pd.testing.assert_frame_equal(a.holdings, b.holdings)
    assert a.summary == b.summary


@pytest.mark.data
@needs_data
def test_forecasts_match_phase3(data, fc_latest):
    """The tool's model run reproduces the Phase 3 forecasts on every shared date."""
    f, _ = fc_latest
    if not P3.exists():
        pytest.skip("run notebooks.phase3_forecasting first")
    p3 = pd.read_parquet(P3)
    pd.testing.assert_frame_equal(f.loc[p3.index, p3.columns], p3, check_freq=False)


@pytest.mark.data
@needs_data
def test_past_recommendation_uses_no_later_data(data):
    """A recommendation as of a past date is identical when every table is scrambled
    after that date: it is exactly what the tool would have said then."""
    t = pd.Timestamp("2022-06-30")
    moved = {k: (perturb_after(v, t) if k not in ("sectors", "shares_outstanding") else v)
             for k, v in data.items()}
    a = recommend(EXAMPLE, CFG, data, as_of=t)
    b = recommend(EXAMPLE, CFG, moved, as_of=t)
    assert a.as_of == b.as_of == t
    pd.testing.assert_series_equal(a.target, b.target)
    pd.testing.assert_frame_equal(a.holdings, b.holdings)

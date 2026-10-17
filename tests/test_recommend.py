"""Phase 5 tests: the recommendation tool (inputs → constrained allocation), reproducibility
and no look-ahead."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.data import load_config, load_processed
from src.recommend import (UserRequest, check_constraints, forecasts_up_to, recommend,
                           resolve_as_of, sector_bounds, validate_request, whole_shares)
from tests.pit import perturb_after

CFG = load_config()
W_MAX = CFG["backtest"]["w_max"]
TOL = CFG["optimization"]["weight_tol"]


# --------------------------------------------------------------------------- unit tests


def test_sector_bounds_from_preferences():
    req = UserRequest(1000, "medium", exclude=["Energy"],
                      max_sector={"Information Technology": 0.2},
                      min_sector={"Health Care": 0.15, "Information Technology": 0.05})
    assert sector_bounds(req) == {"Energy": (0.0, 0.0), "Information Technology": (0.05, 0.2),
                                  "Health Care": (0.15, 1.0)}


def test_excluded_sector_cannot_be_reopened_by_a_minimum():
    req = UserRequest(1000, "medium", exclude=["Energy"], min_sector={"Energy": 0.1})
    assert sector_bounds(req)["Energy"] == (0.1, 0.0)            # lower > upper → rejected
    sectors = pd.Series({f"S{i}": ("Energy" if i < 4 else "Other") for i in range(20)})
    with pytest.raises(ValueError, match="minimum"):
        validate_request(req, sectors, CFG)


@pytest.mark.parametrize("req, msg", [
    (UserRequest(0, "medium"), "positive"),
    (UserRequest(1000, "extreme"), "Risk tolerance"),
    (UserRequest(1000, "medium", exclude=["Crypto"]), "Unknown sector"),
    (UserRequest(1000, "medium", max_sector={"Other": 1.5}), "between 0 and 1"),
    (UserRequest(1000, "medium", min_sector={"Energy": 0.6, "Other": 0.6}), "more than 100%"),
    (UserRequest(1000, "medium", min_sector={"Energy": 0.5}), "cannot reach"),
    (UserRequest(1000, "medium", exclude=["Other"]), "room for only"),
])
def test_invalid_or_infeasible_requests_are_rejected(req, msg):
    # 4 energy stocks + 16 others, cap 10%: Energy can hold at most 40%.
    sectors = pd.Series({f"S{i}": ("Energy" if i < 4 else "Other") for i in range(20)})
    with pytest.raises(ValueError, match=msg):
        validate_request(req, sectors, CFG)


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
    c = check_constraints(w, sectors, 0.5, {"X": (0.0, 0.2), "Y": (0.0, 0.0)}, 1e-9)
    assert c["long_only"] and c["sector_max"] is False and c["cap"] is False
    assert c["excluded_zero"] is False and c["budget"]


# --------------------------------------------------------------------------- real data

HAVE_DATA = (Path(CFG["data"]["processed_dir"]) / "close.parquet").exists()
needs_data = pytest.mark.skipif(not HAVE_DATA, reason="run `python -m src.data` first")


@pytest.fixture(scope="module")
def data():
    return load_processed(CFG)


@pytest.fixture(scope="module")
def fc_latest(data):
    return forecasts_up_to(data, CFG, resolve_as_of(data, None))


EXAMPLE = UserRequest(10_000, "medium", exclude=["Energy"],
                      max_sector={"Information Technology": 0.20})


@pytest.mark.data
@needs_data
def test_worked_example_satisfies_every_constraint(data, fc_latest):
    """The advisor's requested example: amount, risk tolerance and sector preferences
    determine an allocation that satisfies all portfolio constraints."""
    rec = recommend(EXAMPLE, CFG, data, forecasts=fc_latest)
    assert all(rec.checks_target.values()), rec.checks_target
    assert all(rec.checks_holdings.values()), rec.checks_holdings
    t = rec.target
    sectors = data["sectors"]["sector"]
    assert t.sum() == pytest.approx(1.0) and (t >= 0).all() and (t <= W_MAX + TOL).all()
    assert t[sectors.reindex(t.index) == "Energy"].sum() == 0.0
    assert t[sectors.reindex(t.index) == "Information Technology"].sum() <= 0.20 + TOL
    h = rec.holdings
    assert (h["shares"] > 0).all() and (h["shares"] == h["shares"].round()).all()
    assert h["value"].sum() <= EXAMPLE.amount + 1e-6
    assert rec.summary["cash"] == pytest.approx(EXAMPLE.amount - h["value"].sum())
    assert not (h["sector"] == "Energy").any()


@pytest.mark.data
@needs_data
def test_higher_risk_tolerance_gives_higher_volatility(data, fc_latest):
    vols = [recommend(UserRequest(10_000, r, exclude=["Energy"]), CFG, data,
                      forecasts=fc_latest).summary["target_vol_pa"]
            for r in ("low", "medium", "high")]
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
def test_minimum_sector_preference_is_met(data, fc_latest):
    rec = recommend(UserRequest(10_000, "medium", min_sector={"Health Care": 0.15}), CFG, data,
                    forecasts=fc_latest)
    hc = rec.target[data["sectors"]["sector"].reindex(rec.target.index) == "Health Care"].sum()
    assert hc >= 0.15 - TOL
    assert rec.checks_target["sector_min"]
    # Also after rounding to whole shares (measured against the full amount, cash included).
    held_hc = rec.holdings.loc[rec.holdings["sector"] == "Health Care", "value"].sum()
    assert held_hc >= 0.15 * 10_000
    assert rec.checks_holdings["sector_min"]


def test_whole_shares_refills_a_sector_minimum_after_rounding():
    target = pd.Series({"A": 0.08, "B": 0.07, "C": 0.85})
    prices = pd.Series({"A": 30.0, "B": 45.0, "C": 10.0})
    sectors = pd.Series({"A": "HC", "B": "HC", "C": "Other"})
    # Floors: A 2 ($60), B 1 ($45) → HC $105 < 15% of $1,000; C 85 ($850); cash $45.
    out = whole_shares(target, prices, 1000, 1.0, sectors, {"HC": (0.15, 1.0)})
    assert out.loc[out["sector"] == "HC", "value"].sum() >= 150


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
    path = Path(CFG["phase3"]["forecast_dir"]) / "forecasts_elastic_net.parquet"
    if not path.exists():
        pytest.skip("run notebooks.phase3_forecasting first")
    p3 = pd.read_parquet(path)
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

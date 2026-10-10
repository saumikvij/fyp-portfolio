"""Phase 3 tests: feature values, rank normalization, targets and no look-ahead."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.data import load_config, load_processed
from src.features import (FEATURES, add_targets, build_feature_panel, features_at,
                          rank_normalize)
from tests.pit import perturb_after
from tests.toy import make_toy_data

CFG = load_config()
W, RSI = CFG["ml"]["windows"], CFG["ml"]["rsi_days"]


@pytest.fixture(scope="module")
def toy():
    return make_toy_data()


@pytest.fixture(scope="module")
def stocks(toy):
    return [c for c in toy["prices"].columns if c != "SPY"]


def test_feature_values_by_hand(toy, stocks):
    t = toy["prices_monthly"].index[20]
    p, v, pm = toy["prices"].loc[:t], toy["volume"].loc[:t], toy["prices_monthly"].loc[:t]
    f = features_at(p, v, pm, "SPY", stocks, W, RSI)
    s = stocks[0]
    assert f.at[s, "rev_1m"] == pytest.approx(pm[s].iloc[-1] / pm[s].iloc[-2] - 1)
    assert f.at[s, "mom_12_1"] == pytest.approx(pm[s].iloc[-2] / pm[s].iloc[-13] - 1)
    r = p[s].pct_change()
    assert f.at[s, "vol_1m"] == pytest.approx(r.iloc[-21:].std())
    assert f.at[s, "max_ret_1m"] == pytest.approx(r.iloc[-21:].max())
    assert f.at[s, "dist_52w"] == pytest.approx(p[s].iloc[-1] / p[s].iloc[-252:].max() - 1)
    assert (f["dist_52w"] <= 0).all() and f["rsi"].between(0, 100).all()
    assert f["max_dd_1y"].between(0, 1).all()


def test_beta_one_and_zero_idio_vol_for_a_copy_of_spy(toy):
    data = {k: v.copy() for k, v in toy.items()}
    for key in ("prices", "volume", "prices_monthly"):
        data[key]["COPY"] = data[key]["SPY"]
    t = data["prices_monthly"].index[20]
    f = features_at(data["prices"].loc[:t], data["volume"].loc[:t],
                    data["prices_monthly"].loc[:t], "SPY", ["COPY"], W, RSI)
    assert f.at["COPY", "beta_1y"] == pytest.approx(1.0)
    assert f.at["COPY", "idio_vol_3m"] == pytest.approx(0.0, abs=1e-12)


def test_constant_volume_has_zero_trend(toy, stocks):
    v = toy["volume"].copy()
    v[:] = 1e6
    t = toy["prices_monthly"].index[20]
    f = features_at(toy["prices"].loc[:t], v.loc[:t], toy["prices_monthly"].loc[:t], "SPY",
                    stocks, W, RSI)
    np.testing.assert_allclose(f["volume_trend"], 0.0, atol=1e-12)


def test_panel_starts_when_history_is_complete(toy, stocks):
    panel = build_feature_panel(toy, stocks, "SPY", W, RSI)
    assert not panel.isna().any().any()
    first = panel.index.get_level_values("date").min()
    # Needs 12 months of month-end prices and 252 days of daily data.
    assert first >= toy["prices_monthly"].index[12]


def test_rank_normalize():
    idx = pd.MultiIndex.from_product([pd.to_datetime(["2020-01-31", "2020-02-29"]),
                                      list("ABCDE")], names=["date", "ticker"])
    raw = pd.DataFrame({"x": [5, 1, 3, 3, 9, 2, 2, 2, 2, 2]}, index=idx, dtype=float)
    r = rank_normalize(raw)
    np.testing.assert_allclose(r.loc["2020-01-31", "x"], [0.25, -0.5, -0.125, -0.125, 0.5])
    np.testing.assert_allclose(r.loc["2020-02-29", "x"], 0.0)    # all tied


def test_targets_are_next_month_excess_returns(toy, stocks):
    panel = rank_normalize(build_feature_panel(toy, stocks, "SPY", W, RSI))
    rm, rf = toy["returns_monthly"][stocks], toy["rf_monthly"]["rf"]
    out = add_targets(panel, rm, rf)
    t = out.index.get_level_values("date").unique()[3]
    s = rm.index[rm.index > t][0]
    assert (out.loc[t, "target_date"] == s).all()
    np.testing.assert_allclose(out.loc[t, "target"], (rm.loc[s] - rf.loc[s]).values)
    last = out.index.get_level_values("date").max()
    assert out.loc[last, "target"].isna().all()


@pytest.mark.parametrize("k", [16, 30])
def test_features_unchanged_by_future_data(toy, stocks, k):
    """Scramble prices and volumes after month-end t: features dated <= t are identical."""
    t = toy["prices_monthly"].index[k]
    moved = dict(toy)
    for key in ("prices", "volume", "prices_monthly"):
        moved[key] = perturb_after(toy[key], t)
    a = build_feature_panel(toy, stocks, "SPY", W, RSI)
    b = build_feature_panel(moved, stocks, "SPY", W, RSI)
    da, db = a.index.get_level_values("date"), b.index.get_level_values("date")
    pd.testing.assert_frame_equal(a[da <= t], b[db <= t])
    assert not a[da > t].equals(b[db > t])


HAVE_DATA = (Path(CFG["data"]["processed_dir"]) / "prices.parquet").exists()


@pytest.mark.data
@pytest.mark.skipif(not HAVE_DATA, reason="run `python -m src.data` first")
def test_real_panel_shape_and_range():
    d = load_processed(CFG)
    stocks = list(CFG["universe"])
    panel = rank_normalize(build_feature_panel(d, stocks, "SPY", W, RSI))
    assert list(panel.columns) == FEATURES
    assert panel.index.get_level_values("date").min() == pd.Timestamp("2011-01-31")
    assert panel.groupby(level="date").size().eq(len(stocks)).all()
    assert panel.min().min() >= -0.5 and panel.max().max() <= 0.5

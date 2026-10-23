"""D6 tests: the ML risk model, Σ = D·R·D, its use in Black–Litterman, and the
Diebold–Mariano test."""

import copy

import numpy as np
import pandas as pd
import pytest

from src.backtest import history_at
from src.black_litterman import bl_inputs
from src.data import load_config
from src.estimators import sample_covariance
from src.ml_eval import diebold_mariano
from src.risk_model import HARModel, VOL_FEATURES, dr_covariance, vol_forecasts, vol_panel
from src.models import walk_forward
from tests.pit import assert_uses_only_past, perturb_after
from tests.toy import make_toy_data

CFG = load_config()
STOCKS = [f"S{i}" for i in range(6)]


def toy_cfg():
    c = copy.deepcopy(CFG)
    c["ml"].update(val_months=6, min_train_months=6)
    return c


def high_low(data):
    p = data["prices"]
    rng = np.random.default_rng(1)
    spread = pd.DataFrame(rng.uniform(0.002, 0.02, p.shape), index=p.index, columns=p.columns)
    return p * (1 + spread), p * (1 - spread)


@pytest.fixture(scope="module")
def toy():
    return make_toy_data()


@pytest.fixture(scope="module")
def panel(toy):
    h, lo = high_low(toy)
    return vol_panel(toy, STOCKS, "SPY", h, lo)


@pytest.fixture(scope="module")
def dates(panel):
    return panel.index.get_level_values("date").unique().sort_values()[14:-1]


def test_target_is_next_month_realized_volatility(toy, panel):
    """target at t = log std of the daily returns after t up to the next month-end."""
    me = toy["returns_monthly"].index
    t = panel.index.get_level_values("date").unique()[5]
    s = me[me > t][0]
    r = toy["returns_daily"]
    expected = np.log(r.loc[(r.index > t) & (r.index <= s), STOCKS].std() + 1e-8)
    np.testing.assert_allclose(panel.loc[t, "target"].values, expected.values)
    assert (panel.loc[t, "target_date"] == s).all()


def test_features_use_only_past_data(toy):
    """Scrambling every input after t leaves the features at t (and before) unchanged."""
    h, lo = high_low(toy)
    t = toy["returns_monthly"].index[30]
    a = vol_panel(toy, STOCKS, "SPY", h, lo)
    moved = {k: perturb_after(v, t) if k in ("returns_daily", "prices") else v
             for k, v in toy.items()}
    b = vol_panel(moved, STOCKS, "SPY", perturb_after(h, t), perturb_after(lo, t))
    ia = a.index.get_level_values("date") <= t
    pd.testing.assert_frame_equal(a.loc[ia, VOL_FEATURES], b.loc[ia, VOL_FEATURES])
    # Targets realized after t do move, so the test is not vacuous.
    assert not a.loc[a.index.get_level_values("date") == t, "target"].equals(
        b.loc[b.index.get_level_values("date") == t, "target"])


def test_vol_forecasts_unchanged_by_future_data(toy, panel, dates):
    """End to end: forecasts made at or before t ignore data after t."""
    cfg = toy_cfg()
    t = dates[len(dates) // 2]
    h, lo = high_low(toy)
    moved = {k: perturb_after(v, t) if k in ("returns_daily", "prices") else v
             for k, v in toy.items()}
    pb = vol_panel(moved, STOCKS, "SPY", perturb_after(h, t), perturb_after(lo, t))
    a = vol_forecasts(toy, STOCKS, dates, cfg, panel=panel)
    b = vol_forecasts(moved, STOCKS, dates, cfg, panel=pb)
    pd.testing.assert_frame_equal(a.loc[:t], b.loc[:t])
    assert not a.loc[t:].iloc[1:].equals(b.loc[t:].iloc[1:])


def test_dr_covariance_is_psd_with_forecast_diagonal(toy):
    t = toy["returns_monthly"].index[30]
    lv = pd.Series(np.log(np.linspace(0.01, 0.03, len(STOCKS))), index=STOCKS)
    S = dr_covariance(toy["returns_daily"], t, STOCKS, lv, 252, 21)
    assert np.linalg.eigvalsh(S.values).min() > -1e-12
    np.testing.assert_allclose(np.diag(S.values), np.exp(2 * lv.values) * 21)
    np.testing.assert_allclose(S.values, S.values.T)


def test_dr_covariance_uses_only_past(toy):
    t = toy["returns_monthly"].index[30]
    lv = pd.Series(np.log(0.02), index=STOCKS)
    assert_uses_only_past(lambda r, t_: dr_covariance(r, t_, STOCKS, lv, 252, 21),
                          toy["returns_daily"], t)


def test_bl_uses_ml_sigma_only_when_given(toy, dates):
    """vol_forecasts=None keeps the sample Σ (Phases 2–5 unchanged); otherwise Σ = D·R·D."""
    cfg = toy_cfg()
    cfg["universe"] = {s: "X" for s in STOCKS}
    t = dates[-3]
    hist = history_at(toy, t)
    fc = pd.DataFrame(0.005, index=dates, columns=STOCKS)
    vol = pd.DataFrame(np.log(0.015), index=dates, columns=STOCKS)
    w, dpm = cfg["backtest"]["cov_window_days"], cfg["data"]["trading_days_per_year"] / 12
    a = bl_inputs(hist, fc, None, STOCKS, cfg, "A", 0.05)
    b = bl_inputs(hist, fc, None, STOCKS, cfg, "A", 0.05, vol_forecasts=vol)
    pd.testing.assert_frame_equal(a["Sigma"], sample_covariance(hist.returns_daily[STOCKS], t, w, dpm))
    pd.testing.assert_frame_equal(b["Sigma"], dr_covariance(hist.returns_daily, t, STOCKS, vol.loc[t], w, dpm))


def test_har_recovers_a_linear_relation(panel, dates):
    p = panel.copy()
    p["target"] = 0.1 + 0.5 * p["lv_5"] + 0.3 * p["lv_21"] + 0.1 * p["lv_63"]
    f = walk_forward(HARModel(), p, dates, 6, 6).forecasts
    t = dates[-1]
    expected = p.loc[t, "target"].reindex(STOCKS)
    np.testing.assert_allclose(f.loc[t, STOCKS].values, expected.values, atol=1e-8)


def test_diebold_mariano():
    rng = np.random.default_rng(0)
    idx = pd.date_range("2016-01-31", periods=120, freq="ME")
    base = pd.Series(rng.uniform(0.5, 1.5, 120), index=idx)
    same = diebold_mariano(base, base)
    assert same["mean_diff"] == 0
    better = diebold_mariano(base * 0.8, base)
    assert better["dm"] < -3 and better["p_value"] < 0.01
    worse = diebold_mariano(base, base * 0.8)
    assert worse["dm"] == pytest.approx(-better["dm"])

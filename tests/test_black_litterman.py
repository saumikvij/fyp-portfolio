"""Phase 4 tests: Black–Litterman theorem checks (T6, Cor. 6.1), Ω, and no look-ahead."""

import numpy as np
import pandas as pd
import pytest

from src.backtest import black_litterman_strategy, history_at, run_backtest
from src.black_litterman import (bl_inputs, forecast_errors, implied_prior, market_weights,
                                 omega_a, omega_b, posterior_covariance, posterior_mean,
                                 posterior_mean_bayes, posterior_mean_gls)
from src.data import load_config
from src.optimization import mean_variance_closed_form
from tests.pit import perturb_after
from tests.toy import make_toy_data

CFG = load_config()
TAU = CFG["black_litterman"]["tau"]
SEEDS = [0, 1, 2]


def problem(seed, n=8, k=None):
    """Random Σ (factor model), Π, views Q with diagonal Ω, and a pick matrix P."""
    rng = np.random.default_rng(seed)
    F = rng.normal(0, 0.03, (n, 3))
    S = F @ F.T + np.diag(rng.uniform(0.002, 0.006, n))
    Pi = rng.normal(0.006, 0.003, n)
    k = n if k is None else k
    P = np.eye(n) if k == n else rng.normal(0, 1, (k, n))
    Q = rng.normal(0.01, 0.02, k)
    Om = rng.uniform(0.001, 0.01, k)
    return Pi, S, P, Q, Om


# --------------------------------------------------------------------------- T6


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("k", [None, 3])          # absolute views on all assets; 3 mixed views
def test_T6_bayesian_and_gls_derivations_agree(seed, k):
    """T6: the normal–normal Bayesian update and GLS give the same μ_BL (and the
    update form used in the strategies matches both)."""
    Pi, S, P, Q, Om = problem(seed, k=k)
    bayes = posterior_mean_bayes(Pi, S, TAU, Q, Om, P)
    gls = posterior_mean_gls(Pi, S, TAU, Q, Om, P)
    upd = posterior_mean(Pi, S, TAU, Q, Om, P)
    np.testing.assert_allclose(bayes, gls, rtol=1e-10, atol=1e-14)
    np.testing.assert_allclose(upd, bayes, rtol=1e-10, atol=1e-14)


@pytest.mark.parametrize("seed", SEEDS)
def test_T6_posterior_covariance(seed):
    """M = [(τΣ)⁻¹ + PᵀΩ⁻¹P]⁻¹ is symmetric PD and τΣ − M is PSD (views only reduce
    uncertainty); the update form equals the precision form."""
    _, S, P, _, Om = problem(seed, k=3)
    M = posterior_covariance(S, TAU, Om, P)
    precision_form = np.linalg.inv(np.linalg.inv(TAU * S) + P.T @ np.diag(1 / Om) @ P)
    np.testing.assert_allclose(M, precision_form, rtol=1e-8, atol=1e-14)
    np.testing.assert_allclose(M, M.T, atol=1e-15)
    assert np.linalg.eigvalsh(M).min() > 0
    assert np.linalg.eigvalsh(TAU * S - M).min() > -1e-14


def test_T6_scalar_case_is_precision_weighted_average():
    """With diagonal Σ and P = I each μᵢ is a precision-weighted average of Πᵢ and Qᵢ."""
    s2 = np.array([0.004, 0.01])
    Pi, Q, Om = np.array([0.005, 0.008]), np.array([0.02, -0.01]), np.array([0.002, 0.0005])
    w_view = (TAU * s2) / (TAU * s2 + Om)
    np.testing.assert_allclose(posterior_mean(Pi, np.diag(s2), TAU, Q, Om),
                               (1 - w_view) * Pi + w_view * Q)


# --------------------------------------------------------------------------- Cor. 6.1


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("k", [None, 3])
def test_cor61_huge_omega_returns_prior(seed, k):
    Pi, S, P, Q, Om = problem(seed, k=k)
    for scale in (1e6, 1e10):
        mu = posterior_mean(Pi, S, TAU, Q, Om * scale, P)
        np.testing.assert_allclose(mu, Pi, atol=1e-3 / scale ** 0.5)
    np.testing.assert_allclose(posterior_mean(Pi, S, TAU, Q, Om * 1e12, P), Pi, atol=1e-9)


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("k", [None, 3])
def test_cor61_tiny_omega_makes_views_hold_exactly(seed, k):
    Pi, S, P, Q, Om = problem(seed, k=k)
    for scale in (1e-6, 1e-10):
        mu = posterior_mean(Pi, S, TAU, Q, Om * scale, P)
        np.testing.assert_allclose(P @ mu, Q, atol=1e-5)
    # Ω = 0 exactly is allowed in the update form: Pμ = Q.
    np.testing.assert_allclose(P @ posterior_mean(Pi, S, TAU, Q, np.zeros(len(Q)), P), Q,
                               atol=1e-12)


# --------------------------------------------------------------------------- prior


@pytest.mark.parametrize("seed", SEEDS)
def test_reverse_optimization_recovers_market_weights(seed):
    """At μ = Π = δΣw_mkt, unconstrained mean-variance with γ = δ returns w_mkt."""
    _, S, _, _, _ = problem(seed)
    names = [f"S{i}" for i in range(len(S))]
    Sigma = pd.DataFrame(S, index=names, columns=names)
    w = pd.Series(np.random.default_rng(seed).dirichlet(np.ones(len(S))), index=names)
    Pi = implied_prior(Sigma, w, 2.5)
    np.testing.assert_allclose(mean_variance_closed_form(Pi, Sigma, 2.5), w, atol=1e-12)


def test_market_weights():
    w = market_weights(pd.Series({"A": 10.0, "B": 20.0}), pd.Series({"A": 3.0, "B": 1.0}))
    np.testing.assert_allclose(w, [0.6, 0.4])
    with pytest.raises(ValueError):
        market_weights(pd.Series({"A": 10.0, "C": 5.0}), pd.Series({"A": 3.0}))


# --------------------------------------------------------------------------- Ω


def test_forecast_errors_use_only_realized_months():
    me = pd.date_range("2020-01-31", periods=4, freq="ME")
    f = pd.DataFrame({"A": [0.01, 0.02, 0.03, 0.04]}, index=me)
    ex = pd.DataFrame({"A": [0.0, 0.05, -0.02, 0.10]}, index=me).loc[:me[2]]  # known at me[2]
    e = forecast_errors(f, ex)
    assert list(e.index) == [me[0], me[1]]          # forecasts at me[2], me[3] not realized
    np.testing.assert_allclose(e["A"], [0.05 - 0.01, -0.02 - 0.02])


def test_omega_a_shrinkage_formula():
    var = pd.Series({"A": 0.004, "B": 0.009})
    assert omega_a(pd.DataFrame(columns=["A", "B"]), var, 24, 12).equals(var)   # no errors
    e = pd.DataFrame({"A": [0.1, -0.1, 0.05], "B": [0.0, np.nan, 0.02]})
    om = omega_a(e, var, 24, 12)
    assert om["A"] == pytest.approx((0.01 + 0.01 + 0.0025 + 12 * 0.004) / (3 + 12))
    assert om["B"] == pytest.approx((0.0 + 0.0004 + 12 * 0.009) / (2 + 12))
    # Only the last ``window`` errors count.
    assert omega_a(e, var, 1, 12)["A"] == pytest.approx((0.0025 + 12 * 0.004) / 13)


def test_omega_b_raw_and_calibrated():
    v = pd.Series({"A": 1e-6, "B": 3e-6})
    oa = pd.Series({"A": 0.004, "B": 0.008})
    np.testing.assert_allclose(omega_b(v), v + 1e-12)
    cal = omega_b(v, oa)
    assert cal.mean() == pytest.approx(oa.mean())
    assert cal["B"] / cal["A"] == pytest.approx(3.0, rel=1e-5)    # keeps relative disagreement
    flat = omega_b(pd.Series({"A": 0.0, "B": 0.0}), oa)           # copies agree exactly
    np.testing.assert_allclose(flat, oa.mean())


# --------------------------------------------------------------------------- look-ahead


@pytest.fixture(scope="module")
def toy():
    return make_toy_data()


def toy_forecasts(toy, seed=0):
    stocks = [c for c in toy["prices"].columns if c != "SPY"]
    me = toy["returns_monthly"].index
    rng = np.random.default_rng(seed)
    f = pd.DataFrame(rng.normal(0.01, 0.02, (len(me), len(stocks))), index=me, columns=stocks)
    v = pd.DataFrame(rng.uniform(1e-6, 1e-4, f.shape), index=me, columns=stocks)
    return stocks, f, v


@pytest.mark.parametrize("kind", ["A", "B", "B-cal"])
def test_bl_inputs_unchanged_by_future_data(toy, kind):
    """Scramble prices, returns, rf and all forecasts after t: Π, Q, Ω, μ_BL at t equal."""
    stocks, f, v = toy_forecasts(toy)
    t = toy["returns_monthly"].index[30]
    moved = {k: (perturb_after(x, t) if k not in ("sectors", "shares_outstanding") else x)
             for k, x in toy.items()}
    a = bl_inputs(history_at(toy, t), f, v, stocks, CFG, kind, TAU)
    b = bl_inputs(history_at(moved, t), perturb_after(f, t), perturb_after(v, t), stocks,
                  CFG, kind, TAU)
    for key in ("Pi", "Q", "Omega", "mu_bl", "w_mkt"):
        pd.testing.assert_series_equal(a[key], b[key])


def test_s6_backtest_unchanged_by_future_data_and_respects_constraints(toy):
    stocks, f, v = toy_forecasts(toy)
    dates = toy["returns_monthly"].index[20:40]
    t = dates[10]
    moved = {k: (perturb_after(x, t) if k not in ("sectors", "shares_outstanding") else x)
             for k, x in toy.items()}
    cfg = {**CFG, "backtest": {**CFG["backtest"], "w_max": 0.4},
           "optimization": {**CFG["optimization"], "sector_max": 0.6}}   # toy: 2 sectors
    wt = CFG["optimization"]["weight_tol"]
    a = run_backtest("a", black_litterman_strategy(f, v, stocks, cfg, 5.0, "A", TAU),
                     toy, dates, wt)
    b = run_backtest("b", black_litterman_strategy(perturb_after(f, t), perturb_after(v, t),
                                                   stocks, cfg, 5.0, "A", TAU),
                     moved, dates, wt)
    pd.testing.assert_frame_equal(a.weights.loc[:t], b.weights.loc[:t])
    assert (a.weights >= 0).all().all() and (a.weights <= 0.4 + wt).all().all()
    by_sector = a.weights.T.groupby(toy["sectors"]["sector"]).sum().T
    assert (by_sector <= 0.6 + wt).all().all()                          # the sector rule


def test_views_false_uses_prior_only(toy):
    """views=False is the Ω → ∞ limit: same weights as Ω scaled by 1e12."""
    stocks, f, v = toy_forecasts(toy)
    dates = toy["returns_monthly"].index[25:28]
    cfg = {**CFG, "backtest": {**CFG["backtest"], "w_max": 0.4},
           "optimization": {**CFG["optimization"], "sector_max": 0.6}}   # toy: 2 sectors
    wt = CFG["optimization"]["weight_tol"]
    prior = run_backtest("p", black_litterman_strategy(f, v, stocks, cfg, 5.0, "A", TAU,
                                                       views=False), toy, dates, wt)
    huge = run_backtest("h", black_litterman_strategy(f, v, stocks, cfg, 5.0, "A", TAU,
                                                      omega_scale=1e12), toy, dates, wt)
    np.testing.assert_allclose(prior.weights.values, huge.weights.values, atol=1e-6)

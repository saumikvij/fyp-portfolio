"""Phase 1 theorem-checking tests (plan Section 7): T1, T2, Cor. 2.1, T3, T4.

Each test checks a result from the report numerically on random well-conditioned
problems (a small factor model), and the solver's constrained solution where relevant.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.optimize import minimize

from src.data import load_config, load_processed
from src.estimators import historical_mean, sample_covariance
from src.optimization import (Constraints, frontier_closed_form, frontier_constants,
                              frontier_return_range, frontier_variance, gmv_closed_form,
                              kkt_residuals, mean_variance, mean_variance_closed_form,
                              min_variance, portfolio_stats, tangency_closed_form,
                              two_fund_portfolios, efficient_frontier)

CFG = load_config()
TOL = CFG["optimization"]["kkt_tol"]
WTOL = CFG["optimization"]["weight_tol"]
SEEDS = [0, 1, 2]
UNCONSTRAINED = Constraints(long_only=False, w_max=None)


def make_problem(seed: int, n: int = 10):
    """Random monthly μ and positive definite Σ from a 3-factor model."""
    rng = np.random.default_rng(seed)
    tickers = [f"S{i}" for i in range(n)]
    F = rng.normal(0, 0.03, (n, 3))
    Sigma = F @ F.T + np.diag(rng.uniform(0.002, 0.006, n))
    mu = rng.normal(0.008, 0.006, n)
    sectors = pd.Series(["X", "Y", "Z", "W"] * (n // 4) + ["X"] * (n % 4), index=tickers)
    return (pd.Series(mu, index=tickers), pd.DataFrame(Sigma, index=tickers, columns=tickers),
            sectors)


def utility(w, mu, Sigma, gamma):
    return float(mu @ w - gamma / 2 * w @ Sigma.values @ w)


# --------------------------------------------------------------------------- T1


@pytest.mark.parametrize("seed", SEEDS)
def test_T1_unique_optimum_strong_concavity(seed):
    """If w* is optimal, U(w*) − U(w) ≥ (γ/2)(w − w*)ᵀΣ(w − w*) > 0 for every other
    feasible w: the optimum is unique when Σ ≻ 0."""
    mu, Sigma, _ = make_problem(seed)
    gamma, cons = 5.0, Constraints(long_only=True, w_max=0.3)
    w_star = mean_variance(mu, Sigma, gamma, cons).weights.values
    rng = np.random.default_rng(seed + 100)
    u_star = utility(w_star, mu, Sigma, gamma)
    for _ in range(200):
        w = rng.dirichlet(np.ones(len(mu)))
        if w.max() > 0.3:
            continue
        d = w - w_star
        gap = u_star - utility(w, mu, Sigma, gamma)
        assert gap >= gamma / 2 * d @ Sigma.values @ d - 1e-8


# --------------------------------------------------------------------------- T2


@pytest.mark.parametrize("seed", SEEDS)
def test_T2_closed_form_frontier_matches_solver(seed):
    mu, Sigma, _ = make_problem(seed)
    consts = frontier_constants(mu, Sigma)
    for m in np.linspace(mu.min(), mu.max(), 7):
        w_cf = frontier_closed_form(mu, Sigma, m)
        w_num = efficient_frontier(mu, Sigma, [m], UNCONSTRAINED).iloc[0][mu.index]
        np.testing.assert_allclose(w_cf.values, w_num.values.astype(float), atol=1e-6)
        stats = portfolio_stats(w_cf, mu, Sigma)
        assert stats["mean"] == pytest.approx(m)
        assert stats["variance"] == pytest.approx(frontier_variance(m, consts), rel=1e-9)


@pytest.mark.parametrize("seed", SEEDS)
def test_T2_frontier_is_parabola_in_variance_mean_space(seed):
    """Numerical (solver) frontier points lie exactly on σ² = (Am² − 2Bm + C)/D."""
    mu, Sigma, _ = make_problem(seed)
    targets = np.linspace(mu.min(), mu.max(), 9)
    ef = efficient_frontier(mu, Sigma, targets, UNCONSTRAINED)
    coef = np.polyfit(ef["mean"], ef["variance"], 2)
    A, B, C, D = frontier_constants(mu, Sigma)
    np.testing.assert_allclose(coef, [A / D, -2 * B / D, C / D], rtol=1e-5)


@pytest.mark.parametrize("seed", SEEDS)
def test_T2_gmv_is_vertex(seed):
    mu, Sigma, _ = make_problem(seed)
    A, B, _, _ = frontier_constants(mu, Sigma)
    w = gmv_closed_form(Sigma)
    stats = portfolio_stats(w, mu, Sigma)
    assert stats["variance"] == pytest.approx(1 / A)
    assert stats["mean"] == pytest.approx(B / A)
    w_num = min_variance(Sigma, UNCONSTRAINED).weights
    np.testing.assert_allclose(w.values, w_num.values, atol=1e-6)


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("gamma", [2.0, 5.0, 10.0])
def test_mean_variance_closed_form_matches_solver_and_lies_on_frontier(seed, gamma):
    mu, Sigma, _ = make_problem(seed)
    w_cf = mean_variance_closed_form(mu, Sigma, gamma)
    w_num = mean_variance(mu, Sigma, gamma, UNCONSTRAINED).weights
    np.testing.assert_allclose(w_cf.values, w_num.values, atol=1e-6)
    stats = portfolio_stats(w_cf, mu, Sigma)
    assert stats["variance"] == pytest.approx(
        frontier_variance(stats["mean"], frontier_constants(mu, Sigma)), rel=1e-9)


# --------------------------------------------------------------------------- Cor. 2.1


@pytest.mark.parametrize("seed", SEEDS)
def test_cor21_two_fund_separation(seed):
    """Any frontier portfolio is the predicted affine combination of two others."""
    mu, Sigma, _ = make_problem(seed)
    m1, m2, m3 = 0.004, 0.012, 0.020
    ef = efficient_frontier(mu, Sigma, [m1, m2, m3], UNCONSTRAINED)[mu.index].astype(float)
    alpha = (m3 - m2) / (m1 - m2)
    predicted = alpha * ef.loc[m1] + (1 - alpha) * ef.loc[m2]
    np.testing.assert_allclose(ef.loc[m3].values, predicted.values, atol=1e-6)
    g, h = two_fund_portfolios(mu, Sigma)
    assert g.sum() == pytest.approx(1) and h.sum() == pytest.approx(0, abs=1e-9)


# --------------------------------------------------------------------------- T3


@pytest.mark.parametrize("seed", SEEDS)
def test_T3_tangency_matches_numerical_sharpe_max(seed):
    mu, Sigma, _ = make_problem(seed)
    A, B, _, _ = frontier_constants(mu, Sigma)
    rf = 0.5 * B / A                      # strictly below the GMV mean

    def neg_sharpe(w):
        return -(mu.values @ w - rf) / np.sqrt(w @ Sigma.values @ w)

    n = len(mu)
    opt = minimize(neg_sharpe, np.ones(n) / n, method="SLSQP", options={"ftol": 1e-14,
                   "maxiter": 1000}, constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1}])
    w_t = tangency_closed_form(mu, Sigma, rf)
    assert -opt.fun == pytest.approx(-neg_sharpe(w_t.values), rel=1e-7)
    np.testing.assert_allclose(opt.x, w_t.values, atol=1e-4)


@pytest.mark.parametrize("seed", SEEDS)
def test_T3_tangency_on_frontier_and_best_sharpe(seed):
    mu, Sigma, _ = make_problem(seed)
    consts = frontier_constants(mu, Sigma)
    rf = 0.5 * consts.B / consts.A
    s = portfolio_stats(tangency_closed_form(mu, Sigma, rf), mu, Sigma)
    assert s["variance"] == pytest.approx(frontier_variance(s["mean"], consts), rel=1e-9)
    sharpe_t = (s["mean"] - rf) / s["vol"]
    m = np.linspace(consts.B / consts.A, 0.1, 500)
    assert np.all((m - rf) / np.sqrt(frontier_variance(m, consts)) <= sharpe_t + 1e-12)


def test_T3_no_tangency_when_rf_above_gmv_mean():
    mu, Sigma, _ = make_problem(0)
    A, B, _, _ = frontier_constants(mu, Sigma)
    with pytest.raises(ValueError, match="No tangency"):
        tangency_closed_form(mu, Sigma, 1.5 * B / A)


# --------------------------------------------------------------------------- T4


CONSTRAINT_CASES = {
    "long_only": Constraints(long_only=True),
    "capped": Constraints(long_only=True, w_max=0.2),
    "capped_sectors": None,  # built per problem below (needs the sector labels)
}


def constraints_for(case, sectors):
    if case == "capped_sectors":
        return Constraints(long_only=True, w_max=0.2, sectors=sectors,
                           sector_bounds={"X": (0.1, 0.3), "Y": (0.0, 0.0)})
    return CONSTRAINT_CASES[case]


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("gamma", [2.0, 5.0, 10.0])
@pytest.mark.parametrize("case", list(CONSTRAINT_CASES))
def test_T4_kkt_hold_at_solver_solution(seed, gamma, case):
    mu, Sigma, sectors = make_problem(seed)
    cons = constraints_for(case, sectors)
    res = mean_variance(mu, Sigma, gamma, cons)
    for name, value in kkt_residuals(res, mu, Sigma, gamma, cons).items():
        assert value < TOL, (name, value)


@pytest.mark.parametrize("seed", SEEDS)
def test_T4_multiplier_interpretation(seed):
    """Held assets share marginal utility ν; zero-weight assets are at or below ν;
    capped assets are at or above ν."""
    mu, Sigma, _ = make_problem(seed)
    gamma, cons = 5.0, Constraints(long_only=True, w_max=0.2)
    res = mean_variance(mu, Sigma, gamma, cons)
    w = res.weights
    mu_marg = mu - gamma * Sigma.values @ w.values
    # Interior-point solutions sit within ~1e-6 of an active bound, hence WTOL.
    zero, capped = w < WTOL, w > 0.2 - WTOL
    interior = ~zero & ~capped
    assert interior.any() and zero.any()
    np.testing.assert_allclose(mu_marg[interior], res.nu, atol=WTOL)
    assert (mu_marg[zero] <= res.nu + TOL).all()
    assert (mu_marg[capped] >= res.nu - TOL).all()
    # Exact identity from stationarity: marginal utility = ν − λ + η.
    np.testing.assert_allclose(mu_marg, res.nu - res.lam + res.eta, atol=TOL)


def test_sector_bounds_respected():
    mu, Sigma, sectors = make_problem(0)
    cons = constraints_for("capped_sectors", sectors)
    w = mean_variance(mu, Sigma, 5.0, cons).weights
    by_sector = w.groupby(sectors).sum()
    assert by_sector["Y"] == pytest.approx(0, abs=1e-7)
    assert 0.1 - 1e-7 <= by_sector["X"] <= 0.3 + 1e-7
    assert (w <= 0.2 + 1e-7).all() and (w >= -1e-7).all()


def test_infeasible_constraints_raise():
    mu, Sigma, _ = make_problem(0)
    with pytest.raises(ValueError, match="infeasible"):
        mean_variance(mu, Sigma, 5.0, Constraints(long_only=True, w_max=0.05))  # 10 x 5% < 1


def test_unknown_sector_raises():
    mu, Sigma, sectors = make_problem(0)
    cons = Constraints(sectors=sectors, sector_bounds={"Nope": (0, 0)})
    with pytest.raises(ValueError, match="Unknown sectors"):
        mean_variance(mu, Sigma, 5.0, cons)


def test_constrained_frontier_within_return_range_and_above_unconstrained():
    mu, Sigma, _ = make_problem(1)
    cons = Constraints(long_only=True, w_max=0.3)
    lo, hi = frontier_return_range(mu, cons)
    targets = np.linspace(lo, hi, 8)
    ef = efficient_frontier(mu, Sigma, targets, cons)
    np.testing.assert_allclose(ef["mean"], targets, atol=1e-7)
    # Adding constraints can only raise the minimum variance at each target.
    assert (ef["variance"] >= frontier_variance(targets, frontier_constants(mu, Sigma))
            - 1e-10).all()


# --------------------------------------------------------------------------- real data

HAVE_DATA = (Path(CFG["data"]["processed_dir"]) / "prices.parquet").exists()


@pytest.mark.data
@pytest.mark.skipif(not HAVE_DATA, reason="run `python -m src.data` first")
@pytest.mark.parametrize("level", ["low", "medium", "high"])
def test_T4_kkt_on_real_data(level):
    d = load_processed(CFG)
    stocks = list(CFG["universe"])
    t = pd.Timestamp(CFG["frontier_plot"]["as_of"])
    Sigma = sample_covariance(d["returns_daily"][stocks], t,
                              CFG["backtest"]["cov_window_days"],
                              CFG["data"]["trading_days_per_year"] / 12)
    mu = historical_mean(d["returns_monthly"][stocks], t)
    gamma = CFG["risk_tolerance"][level]
    cons = Constraints(long_only=True, w_max=CFG["backtest"]["w_max"],
                       sectors=d["sectors"]["sector"], sector_bounds={"Energy": (0.0, 0.0)})
    res = mean_variance(mu, Sigma, gamma, cons)
    for name, value in kkt_residuals(res, mu, Sigma, gamma, cons).items():
        assert value < TOL, (name, value)

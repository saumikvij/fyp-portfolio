"""Phase 2 tests: metrics, and the theorem checks for T7 (plan Section 7).

T7: CVaR is a coherent risk measure (in particular subadditive); VaR is not.
"""

import numpy as np
import pandas as pd
import pytest

from src.data import load_config
from src.metrics import (annualized_return, annualized_vol, cvar_historical,
                         cvar_tail_average, max_drawdown, performance_summary,
                         sharpe_confidence_interval, sharpe_ratio, var_historical,
                         weight_stability)

CFG = load_config()
ALPHA = CFG["tail_risk"]["alpha"]


def monthly(values):
    return pd.Series(values, index=pd.date_range("2016-01-31", periods=len(values), freq="ME"))


# --------------------------------------------------------------------------- basics


def test_annualized_return_and_vol():
    r = monthly([0.01] * 12)
    assert annualized_return(r) == pytest.approx(1.01 ** 12 - 1)
    assert annualized_vol(r) == pytest.approx(0.0)
    r2 = monthly([0.02, -0.01] * 6)
    assert annualized_vol(r2) == pytest.approx(r2.std(ddof=1) * np.sqrt(12))


def test_sharpe_ratio_uses_excess_returns():
    r = monthly([0.02, 0.00, 0.01, 0.03])
    rf = monthly([0.001] * 4)
    ex = r - rf
    assert sharpe_ratio(r, rf) == pytest.approx(np.sqrt(12) * ex.mean() / ex.std(ddof=1))


def test_sharpe_ci_formula():
    rng = np.random.default_rng(0)
    r, rf = monthly(rng.normal(0.01, 0.04, 120)), monthly(np.zeros(120))
    sr, lo, hi = sharpe_confidence_interval(r, rf, 0.95)
    sr_m = r.mean() / r.std(ddof=1)
    se = np.sqrt((1 + sr_m ** 2 / 2) / 120) * np.sqrt(12)
    assert sr == pytest.approx(sharpe_ratio(r, rf))
    assert (hi - lo) / 2 == pytest.approx(1.959964 * se, rel=1e-5)


def test_sharpe_ci_coverage_iid_normal():
    """Monte Carlo: the Lo (2002) interval covers the true Sharpe about 95% of the time."""
    rng = np.random.default_rng(1)
    mu, sd, T, sims = 0.008, 0.04, 120, 2000
    true_sr = np.sqrt(12) * mu / sd
    zeros = monthly(np.zeros(T))
    hits = 0
    for _ in range(sims):
        _, lo, hi = sharpe_confidence_interval(monthly(rng.normal(mu, sd, T)), zeros, 0.95)
        hits += lo <= true_sr <= hi
    assert abs(hits / sims - 0.95) < 0.02


def test_max_drawdown():
    r = monthly([0.10, -0.20, 0.05, -0.10, 0.50])
    # Wealth: 1, 1.1, 0.88, 0.924, 0.8316, 1.2474 -> trough 0.8316 from peak 1.1.
    assert max_drawdown(r) == pytest.approx(1 - 0.8316 / 1.1)
    assert max_drawdown(monthly([0.01, 0.02])) == 0.0
    assert max_drawdown(monthly([-0.5])) == pytest.approx(0.5)   # loss from the start


def test_weight_stability():
    w = pd.DataFrame({"A": [0.5, 0.5, 0.2], "B": [0.5, 0.5, 0.8]})
    assert weight_stability(w) == pytest.approx((0 + 0.6) / 2)


# --------------------------------------------------------------------------- VaR / CVaR


def test_var_cvar_small_example():
    # 20 outcomes; the worst 5% is one outcome (loss 10).
    r = np.array([-0.10, -0.05] + [0.01] * 18)
    assert var_historical(r, 0.95) == pytest.approx(0.05)
    assert cvar_historical(r, 0.95) == pytest.approx(0.10)


def _ru_minimum(r, alpha):
    """min over ζ of ζ + E[(L − ζ)⁺]/(1 − α). The objective is convex and piecewise
    linear with kinks at the sample losses, so the minimum is at one of them."""
    loss = -np.asarray(r)
    return min(z + np.mean(np.maximum(loss - z, 0)) / (1 - alpha) for z in loss)


@pytest.mark.parametrize("seed", range(5))
@pytest.mark.parametrize("n", [100, 199, 1000])
@pytest.mark.parametrize("alpha", [0.90, 0.95, 0.99])
def test_cvar_tail_average_equals_rockafellar_uryasev(seed, n, alpha):
    """Plan Section 7: CVaR as a tail average = the Rockafellar–Uryasev minimum,
    including sample sizes where (1 − α)N is not an integer."""
    r = np.random.default_rng(seed).standard_t(4, n) * 0.05
    ru = _ru_minimum(r, alpha)
    assert cvar_tail_average(r, alpha) == pytest.approx(ru, rel=1e-10)
    assert cvar_historical(r, alpha) == pytest.approx(ru, rel=1e-10)
    assert cvar_historical(r, alpha) >= var_historical(r, alpha)


# --------------------------------------------------------------------------- T7


def test_T7_var_not_subadditive_counterexample():
    """Two independent loans, each losing 100 with probability 4% (else 0).

    Alone, P(loss > 0) = 4% < 5%, so VaR_95 = 0 for each. Together,
    P(at least one default) = 1 − 0.96² = 7.84% > 5%, so VaR_95(A + B) = 100 > 0 + 0.
    The 10 000 scenarios below reproduce these probabilities exactly.
    """
    counts = {(0, 0): 9216, (100, 0): 384, (0, 100): 384, (100, 100): 16}
    loss_a = np.concatenate([[a] * c for (a, _), c in counts.items()]).astype(float)
    loss_b = np.concatenate([[b] * c for (_, b), c in counts.items()]).astype(float)
    r_a, r_b = -loss_a, -loss_b
    assert var_historical(r_a, 0.95) == 0.0
    assert var_historical(r_b, 0.95) == 0.0
    assert var_historical(r_a + r_b, 0.95) == 100.0            # VaR(A+B) > VaR(A)+VaR(B)
    # CVaR is subadditive on the same example.
    assert cvar_historical(r_a + r_b, 0.95) <= (cvar_historical(r_a, 0.95)
                                                + cvar_historical(r_b, 0.95))


@pytest.mark.parametrize("seed", range(20))
def test_T7_cvar_subadditive_on_random_samples(seed):
    """CVaR(A + B) ≤ CVaR(A) + CVaR(B) for correlated, heavy-tailed, skewed samples."""
    rng = np.random.default_rng(seed)
    n = 500
    z = rng.standard_t(3, (n, 2)) @ np.array([[1.0, 0.0], [rng.uniform(-1, 1), 1.0]])
    r_a = 0.05 * z[:, 0] - 0.2 * (rng.random(n) < 0.03)   # add rare jumps
    r_b = 0.03 * z[:, 1]
    for alpha in (0.90, 0.95, 0.99):
        assert cvar_historical(r_a + r_b, alpha) <= (cvar_historical(r_a, alpha)
                                                     + cvar_historical(r_b, alpha) + 1e-12)


@pytest.mark.parametrize("seed", range(5))
def test_T7_cvar_other_coherence_axioms(seed):
    """Translation invariance, positive homogeneity and monotonicity (Artzner et al.)."""
    rng = np.random.default_rng(seed)
    r = rng.standard_t(4, 400) * 0.04
    c, lam = 0.01, 3.0
    # Adding a sure gain c lowers the loss by exactly c.
    assert cvar_historical(r + c, ALPHA) == pytest.approx(cvar_historical(r, ALPHA) - c)
    assert cvar_historical(lam * r, ALPHA) == pytest.approx(lam * cvar_historical(r, ALPHA))
    better = r + np.abs(rng.normal(0, 0.01, 400))           # never worse in any scenario
    assert cvar_historical(better, ALPHA) <= cvar_historical(r, ALPHA)


def test_performance_summary_keys():
    rng = np.random.default_rng(2)
    r, rf = monthly(rng.normal(0.01, 0.04, 60)), monthly(np.full(60, 0.001))
    s = performance_summary(r, rf, turnover=monthly(np.full(60, 0.1)))
    assert {"ann_return", "ann_vol", "sharpe", "sharpe_lo", "sharpe_hi", "max_drawdown",
            "var_m", "cvar_m", "avg_turnover"} <= set(s)
    assert s["sharpe_lo"] < s["sharpe"] < s["sharpe_hi"]

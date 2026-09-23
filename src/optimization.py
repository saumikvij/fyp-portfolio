"""Mean-variance optimization (plan Section 2.2): closed forms and the constrained QP.

Main problem, with risk aversion γ > 0:

    maximize  μᵀw − (γ/2) wᵀΣw   subject to  1ᵀw = 1, 0 ≤ wᵢ ≤ w_max, sector bounds

solved as the equivalent minimization of (γ/2) wᵀΣw − μᵀw. Closed forms use equality
constraints only (short sales allowed) and use the plan's notation
A = 1ᵀΣ⁻¹1, B = 1ᵀΣ⁻¹μ, C = μᵀΣ⁻¹μ, D = AC − B².

All functions take μ as a Series and Σ as a DataFrame indexed by ticker and return
weights as a Series with the same index. Linear systems are solved, never inverted.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import NamedTuple

import cvxpy as cp
import numpy as np
import pandas as pd


# --------------------------------------------------------------------------- closed forms


class FrontierConstants(NamedTuple):
    A: float  # 1ᵀΣ⁻¹1
    B: float  # 1ᵀΣ⁻¹μ
    C: float  # μᵀΣ⁻¹μ
    D: float  # AC − B²


def frontier_constants(mu: pd.Series, Sigma: pd.DataFrame) -> FrontierConstants:
    """The scalars A, B, C, D of the closed-form frontier.

    Parameters
    ----------
    mu : expected returns (n).
    Sigma : covariance (n x n), positive definite.

    Returns
    -------
    FrontierConstants(A, B, C, D). D > 0 unless μ is a multiple of 1 (then the
    frontier degenerates to a single point).

    Theorem: T2 (constants of the frontier parabola).
    """
    ones = np.ones(len(mu))
    x1 = np.linalg.solve(Sigma.values, ones)         # Σ⁻¹1
    xm = np.linalg.solve(Sigma.values, mu.values)    # Σ⁻¹μ
    A, B, C = ones @ x1, ones @ xm, mu.values @ xm
    return FrontierConstants(A, B, C, A * C - B * B)


def gmv_closed_form(Sigma: pd.DataFrame) -> pd.Series:
    """Global minimum-variance portfolio with only 1ᵀw = 1: w = Σ⁻¹1 / A.

    Parameters
    ----------
    Sigma : covariance (n x n), positive definite.

    Returns
    -------
    Series of weights summing to 1 (may be negative).

    Theorem: T2 special case (vertex of the frontier parabola, variance 1/A).
    """
    x1 = np.linalg.solve(Sigma.values, np.ones(len(Sigma)))
    return pd.Series(x1 / x1.sum(), index=Sigma.index)


def two_fund_portfolios(mu: pd.Series, Sigma: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """Vectors g, h with frontier weights w*(m) = g + m·h for every target return m.

    From the Lagrange conditions: w*(m) = [(C − Bm)Σ⁻¹1 + (Am − B)Σ⁻¹μ] / D, so
    g = (CΣ⁻¹1 − BΣ⁻¹μ)/D and h = (AΣ⁻¹μ − BΣ⁻¹1)/D. Note 1ᵀg = 1, 1ᵀh = 0.

    Parameters
    ----------
    mu : expected returns (n).
    Sigma : covariance (n x n), positive definite.

    Returns
    -------
    (g, h) as Series.

    Theorem: T2 and Corollary 2.1 (two-fund separation: w*(m) is affine in m).
    """
    A, B, C, D = frontier_constants(mu, Sigma)
    if D <= 0:
        raise ValueError("Degenerate frontier: μ is (numerically) a multiple of 1")
    x1 = np.linalg.solve(Sigma.values, np.ones(len(mu)))
    xm = np.linalg.solve(Sigma.values, mu.values)
    g = (C * x1 - B * xm) / D
    h = (A * xm - B * x1) / D
    return pd.Series(g, index=mu.index), pd.Series(h, index=mu.index)


def frontier_closed_form(mu: pd.Series, Sigma: pd.DataFrame, target: float) -> pd.Series:
    """Minimum-variance portfolio with expected return ``target`` (short sales allowed).

    Solves min wᵀΣw s.t. μᵀw = target, 1ᵀw = 1.

    Parameters
    ----------
    mu : expected returns (n).
    Sigma : covariance (n x n), positive definite.
    target : required expected return m.

    Returns
    -------
    Series of frontier weights w*(m) = g + m·h.

    Theorem: T2.
    """
    g, h = two_fund_portfolios(mu, Sigma)
    return g + target * h


def frontier_variance(target: float | np.ndarray, consts: FrontierConstants):
    """Variance of the frontier portfolio: σ²(m) = (A m² − 2B m + C) / D.

    Parameters
    ----------
    target : target return(s) m.
    consts : output of ``frontier_constants``.

    Returns
    -------
    σ²(m), a parabola in m, with minimum 1/A at m = B/A (the GMV portfolio).

    Theorem: T2.
    """
    A, B, C, D = consts
    return (A * np.square(target) - 2 * B * target + C) / D


def tangency_closed_form(mu: pd.Series, Sigma: pd.DataFrame, rf: float) -> pd.Series:
    """Maximum-Sharpe (tangency) portfolio: w ∝ Σ⁻¹(μ − r_f 1), scaled so 1ᵀw = 1.

    Parameters
    ----------
    mu : expected returns (n).
    Sigma : covariance (n x n), positive definite.
    rf : risk-free rate, same period as μ.

    Returns
    -------
    Series of weights summing to 1 (may be negative).

    Raises
    ------
    ValueError if 1ᵀΣ⁻¹(μ − r_f 1) ≤ 0, i.e. r_f ≥ B/A (the GMV mean): the ray from
    r_f then touches the lower branch and there is no maximum-Sharpe portfolio.

    Theorem: T3.
    """
    z = np.linalg.solve(Sigma.values, mu.values - rf)
    if z.sum() <= 0:
        raise ValueError("No tangency portfolio: r_f is not below the GMV mean B/A")
    return pd.Series(z / z.sum(), index=mu.index)


def mean_variance_closed_form(mu: pd.Series, Sigma: pd.DataFrame, gamma: float) -> pd.Series:
    """Solution of max μᵀw − (γ/2)wᵀΣw s.t. 1ᵀw = 1 (no bounds).

    w* = Σ⁻¹1/A + (1/γ)(Σ⁻¹μ − (B/A)Σ⁻¹1): the GMV portfolio plus a zero-cost
    tilt towards μ that shrinks as γ grows.

    Parameters
    ----------
    mu : expected returns (n).
    Sigma : covariance (n x n), positive definite.
    gamma : risk aversion γ > 0.

    Returns
    -------
    Series of weights summing to 1.

    Theorem: T1 (unique optimum), T2 (it lies on the frontier).
    """
    x1 = np.linalg.solve(Sigma.values, np.ones(len(mu)))
    xm = np.linalg.solve(Sigma.values, mu.values)
    A, B = x1.sum(), xm.sum()
    return pd.Series(x1 / A + (xm - (B / A) * x1) / gamma, index=mu.index)


# --------------------------------------------------------------------------- constrained QP


@dataclass
class Constraints:
    """Constraints of the main problem besides the budget 1ᵀw = 1.

    Attributes
    ----------
    long_only : impose wᵢ ≥ 0.
    w_max : upper bound on every weight, or None.
    sectors : Series ticker -> sector name (needed only with ``sector_bounds``).
    sector_bounds : {sector: (lower, upper)} on the total weight of that sector,
        e.g. {"Energy": (0.0, 0.0)} excludes energy; sectors not listed are free.
    """

    long_only: bool = True
    w_max: float | None = None
    sectors: pd.Series | None = None
    sector_bounds: dict[str, tuple[float, float]] = field(default_factory=dict)

    def sector_matrix(self, index: pd.Index) -> tuple[list[str], np.ndarray]:
        """Rows of 0/1 sector membership for the bounded sectors, in ``index`` order."""
        if not self.sector_bounds:
            return [], np.zeros((0, len(index)))
        if self.sectors is None:
            raise ValueError("sector_bounds given without sectors")
        s = self.sectors.reindex(index)
        if s.isna().any():
            raise ValueError(f"No sector for {s[s.isna()].index.tolist()}")
        names = list(self.sector_bounds)
        unknown = set(names) - set(s)
        if unknown:
            raise ValueError(f"Unknown sectors in sector_bounds: {sorted(unknown)}")
        return names, np.array([(s == name).values.astype(float) for name in names])


@dataclass
class MVResult:
    """Solution of the constrained problem, with the multipliers needed for T4.

    Lagrangian convention (checked against cvxpy):
        ∇f(w) + ν·1 − λ + η + Sᵀ(ρ_up − ρ_low) = 0,
    with f(w) = (γ/2)wᵀΣw − μᵀw, λ ≥ 0 for w ≥ 0, η ≥ 0 for w ≤ w_max, and
    ρ_up, ρ_low ≥ 0 for the sector bounds S w ≤ u and S w ≥ l.
    Economic meaning: stationarity reads μᵢ − γ(Σw)ᵢ = ν − λᵢ + ηᵢ (+ sector terms).
    So every asset held strictly between its bounds has the same marginal utility ν;
    an asset held at zero has marginal utility ν − λᵢ ≤ ν (not worth buying), and
    one at the cap has ν + ηᵢ ≥ ν (would be bought more if the cap allowed).
    """

    weights: pd.Series
    status: str
    objective: float          # value of μᵀw − (γ/2)wᵀΣw (the maximization form)
    nu: float                 # budget multiplier
    lam: pd.Series            # multipliers of w ≥ 0 (zeros if not long-only)
    eta: pd.Series            # multipliers of w ≤ w_max (zeros if no cap)
    sector_names: list[str]
    rho_up: np.ndarray
    rho_low: np.ndarray


def mean_variance(mu: pd.Series, Sigma: pd.DataFrame, gamma: float,
                  constraints: Constraints, solver: str = "CLARABEL",
                  target_return: float | None = None) -> MVResult:
    """Solve max μᵀw − (γ/2)wᵀΣw s.t. 1ᵀw = 1 and ``constraints`` (a convex QP).

    Parameters
    ----------
    mu : expected returns (n). Pass zeros for the minimum-variance problem.
    Sigma : covariance (n x n), positive definite.
    gamma : risk aversion γ > 0.
    constraints : bounds and sector limits.
    solver : cvxpy solver name.
    target_return : if given, add μᵀw = target_return; the −μᵀw term is then
        constant, so the problem becomes min wᵀΣw at that return (a frontier point).

    Returns
    -------
    MVResult with weights and all multipliers.

    Raises
    ------
    ValueError if the problem is infeasible or the solver fails.

    Theorem: T1 (unique optimum since Σ ≻ 0 makes the objective strictly concave),
    T4 (the multipliers returned satisfy the KKT conditions, see ``kkt_residuals``).
    """
    n = len(mu)
    S_val = Sigma.loc[mu.index, mu.index].values
    w = cp.Variable(n)
    budget = cp.sum(w) == 1
    cons = [budget]
    lower = upper = None
    if constraints.long_only:
        lower = w >= 0
        cons.append(lower)
    if constraints.w_max is not None:
        upper = w <= constraints.w_max
        cons.append(upper)
    names, S = constraints.sector_matrix(mu.index)
    sec_up = sec_low = None
    if names:
        lo = np.array([constraints.sector_bounds[s][0] for s in names])
        hi = np.array([constraints.sector_bounds[s][1] for s in names])
        sec_up, sec_low = S @ w <= hi, S @ w >= lo
        cons += [sec_up, sec_low]
    if target_return is not None:
        cons.append(mu.values @ w == target_return)

    objective = (gamma / 2) * cp.quad_form(w, cp.psd_wrap(S_val)) - mu.values @ w
    prob = cp.Problem(cp.Minimize(objective), cons)
    prob.solve(solver=solver)
    if prob.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
        raise ValueError(f"Mean-variance problem {prob.status}")

    zeros = pd.Series(0.0, index=mu.index)
    k = len(names)
    return MVResult(
        weights=pd.Series(w.value, index=mu.index),
        status=prob.status,
        objective=-prob.value,
        nu=float(budget.dual_value),
        lam=pd.Series(lower.dual_value, index=mu.index) if lower is not None else zeros,
        eta=pd.Series(upper.dual_value, index=mu.index) if upper is not None else zeros,
        sector_names=names,
        rho_up=np.asarray(sec_up.dual_value) if k else np.zeros(0),
        rho_low=np.asarray(sec_low.dual_value) if k else np.zeros(0),
    )


def min_variance(Sigma: pd.DataFrame, constraints: Constraints,
                 solver: str = "CLARABEL") -> MVResult:
    """Minimum-variance portfolio under ``constraints`` (strategy S3).

    Parameters
    ----------
    Sigma : covariance (n x n), positive definite.
    constraints : bounds and sector limits.
    solver : cvxpy solver name.

    Returns
    -------
    MVResult (μ = 0, γ = 1, so the objective is −½ wᵀΣw).

    Theorem: T1, T4 (special case μ = 0); equals ``gmv_closed_form`` without bounds.
    """
    return mean_variance(pd.Series(0.0, index=Sigma.index), Sigma, 1.0, constraints, solver)


def frontier_return_range(mu: pd.Series, constraints: Constraints,
                          solver: str = "CLARABEL") -> tuple[float, float]:
    """Lowest and highest expected return reachable under ``constraints`` (two LPs).

    Parameters
    ----------
    mu : expected returns (n).
    constraints : must bound the weights (long-only or a cap), else the range is
        unbounded.
    solver : cvxpy solver name.

    Returns
    -------
    (min μᵀw, max μᵀw) over feasible w.

    Theorem: none (helper for the constrained frontier).
    """
    w = cp.Variable(len(mu))
    cons = [cp.sum(w) == 1]
    if constraints.long_only:
        cons.append(w >= 0)
    if constraints.w_max is not None:
        cons.append(w <= constraints.w_max)
    names, S = constraints.sector_matrix(mu.index)
    if names:
        cons += [S @ w <= [constraints.sector_bounds[s][1] for s in names],
                 S @ w >= [constraints.sector_bounds[s][0] for s in names]]
    out = []
    for sense in (cp.Minimize, cp.Maximize):
        prob = cp.Problem(sense(mu.values @ w), cons)
        prob.solve(solver=solver)
        if prob.status != cp.OPTIMAL:
            raise ValueError(f"Return range problem {prob.status}")
        out.append(float(prob.value))
    return out[0], out[1]


def efficient_frontier(mu: pd.Series, Sigma: pd.DataFrame, targets: np.ndarray,
                       constraints: Constraints, solver: str = "CLARABEL") -> pd.DataFrame:
    """Numerical frontier: minimum variance for each target return under constraints.

    Parameters
    ----------
    mu : expected returns (n).
    Sigma : covariance (n x n), positive definite.
    targets : target returns m (must be reachable, see ``frontier_return_range``).
    constraints : bounds and sector limits.
    solver : cvxpy solver name.

    Returns
    -------
    DataFrame indexed by target with columns ``mean``, ``variance``, ``vol`` and one
    weight column per ticker.

    Theorem: T2 (matches the closed form when there are no bounds).
    """
    rows = []
    for m in targets:
        # With μᵀw fixed at m the −μᵀw term is constant, so this minimizes wᵀΣw.
        res = mean_variance(mu, Sigma, 2.0, constraints, solver, target_return=m)
        w = res.weights
        var = float(w @ Sigma.values @ w)
        rows.append({"target": m, "mean": float(mu @ w), "variance": var,
                     "vol": np.sqrt(var), **w.to_dict()})
    return pd.DataFrame(rows).set_index("target")


# --------------------------------------------------------------------------- KKT (T4)


def kkt_residuals(res: MVResult, mu: pd.Series, Sigma: pd.DataFrame, gamma: float,
                  constraints: Constraints) -> dict[str, float]:
    """Largest violation of each KKT condition at a solution of ``mean_variance``.

    For the convex problem min f(w) = (γ/2)wᵀΣw − μᵀw under linear constraints, the
    KKT conditions are necessary (Slater holds when a strictly feasible w exists)
    and sufficient (f convex), so all residuals ≈ 0 certifies optimality.

    Parameters
    ----------
    res : output of ``mean_variance`` for the same inputs.
    mu, Sigma, gamma, constraints : the problem that was solved.

    Returns
    -------
    dict with keys ``stationarity``, ``primal``, ``dual``, ``complementarity``;
    each is the max absolute violation (0 at an exact solution).

    Theorem: T4.
    """
    w = res.weights.values
    Sv = Sigma.loc[mu.index, mu.index].values
    names, S = constraints.sector_matrix(mu.index)
    grad = gamma * Sv @ w - mu.values
    stat = grad + res.nu - res.lam.values + res.eta.values + S.T @ (res.rho_up - res.rho_low)

    primal = [abs(w.sum() - 1)]
    comp = [0.0]
    if constraints.long_only:
        primal.append(max(0.0, -w.min()))
        comp.append(np.max(np.abs(res.lam.values * w)))
    if constraints.w_max is not None:
        primal.append(max(0.0, (w - constraints.w_max).max()))
        comp.append(np.max(np.abs(res.eta.values * (constraints.w_max - w))))
    if names:
        sw = S @ w
        lo = np.array([constraints.sector_bounds[s][0] for s in names])
        hi = np.array([constraints.sector_bounds[s][1] for s in names])
        primal += [max(0.0, (sw - hi).max()), max(0.0, (lo - sw).max())]
        comp += [np.max(np.abs(res.rho_up * (hi - sw))), np.max(np.abs(res.rho_low * (sw - lo)))]

    duals = np.concatenate([res.lam.values, res.eta.values, res.rho_up, res.rho_low])
    return {
        "stationarity": float(np.max(np.abs(stat))),
        "primal": float(max(primal)),
        "dual": float(max(0.0, -duals.min())) if duals.size else 0.0,
        "complementarity": float(max(comp)),
    }


# --------------------------------------------------------------------------- helpers


def portfolio_stats(w: pd.Series, mu: pd.Series, Sigma: pd.DataFrame) -> dict[str, float]:
    """Ex-ante mean, variance and volatility of a portfolio.

    Parameters
    ----------
    w : weights (n).
    mu : expected returns (n).
    Sigma : covariance (n x n).

    Returns
    -------
    dict with ``mean`` = μᵀw, ``variance`` = wᵀΣw, ``vol`` = √(wᵀΣw), in the units
    of μ and Σ (monthly here).

    Theorem: none (Section 2.1 definitions).
    """
    w = w.reindex(mu.index)
    var = float(w.values @ Sigma.loc[mu.index, mu.index].values @ w.values)
    return {"mean": float(mu @ w), "variance": var, "vol": float(np.sqrt(var))}

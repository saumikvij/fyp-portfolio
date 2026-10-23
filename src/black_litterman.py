"""Black–Litterman: market prior, ML views, view uncertainty and posterior (plan 2.4, 4.4).

Notation (monthly units, excess returns):
    Π = δ Σ w_mkt                 equilibrium prior (reverse optimization)
    μ ~ N(Π, τΣ)                  prior uncertainty about expected returns
    Q = P μ + η,  η ~ N(0, Ω)     views; here P = I and Q = ML forecasts
    μ_BL = [(τΣ)⁻¹ + PᵀΩ⁻¹P]⁻¹ [(τΣ)⁻¹Π + PᵀΩ⁻¹Q]           (T6)

The posterior mean is implemented three ways that T6 says must agree: the Bayesian
precision form, generalized least squares on the stacked system, and the update form
Π + τΣPᵀ(PτΣPᵀ + Ω)⁻¹(Q − PΠ), which is used in practice because it stays stable
when Ω is small (Cor. 6.1).

View uncertainty Ω = diag(σ²ᵢ):
    Ω-A  mean squared past forecast error of stock i (only errors already realized)
    Ω-B  variance of the forecasts of the bootstrap copies of the model (Phase 3)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.estimators import sample_covariance
from src.risk_model import dr_covariance


# --------------------------------------------------------------------------- prior


def market_weights(close: pd.Series, shares: pd.Series) -> pd.Series:
    """Approximate market-cap weights: close price × shares outstanding, normalized.

    Parameters
    ----------
    close : split-adjusted (not dividend-adjusted) close prices at t, by ticker.
    shares : shares outstanding (all classes), by ticker; a current snapshot, which
        is the approximation acknowledged in plan Section 3 (D0.9).

    Returns
    -------
    Weights summing to 1, indexed like ``close``.

    Theorem: none (input to the prior, plan 2.4).
    """
    cap = close * shares.reindex(close.index)
    if cap.isna().any() or (cap <= 0).any():
        raise ValueError("Missing or non-positive market cap")
    return cap / cap.sum()


def implied_prior(Sigma: pd.DataFrame, w_mkt: pd.Series, delta: float) -> pd.Series:
    """Equilibrium excess returns by reverse optimization: Π = δ Σ w_mkt.

    Π are the expected returns under which the market portfolio is the optimal
    mean-variance portfolio for risk aversion δ (with only 1ᵀw = 1, the solution of
    max μᵀw − (δ/2)wᵀΣw at μ = Π is exactly w_mkt; tested).

    Parameters
    ----------
    Sigma : covariance (n x n).
    w_mkt : market weights (n).
    delta : market risk aversion δ.

    Returns
    -------
    Π as a Series (n).

    Theorem: input to T6 (prior mean).
    """
    w = w_mkt.reindex(Sigma.index)
    return pd.Series(delta * Sigma.values @ w.values, index=Sigma.index)


# --------------------------------------------------------------------------- posterior (T6)


def _arrays(Pi, Sigma, P, Q, Omega):
    n = len(Pi)
    P = np.eye(n) if P is None else np.asarray(P, dtype=float)
    Om = np.asarray(Omega, dtype=float)
    Om = np.diag(Om) if Om.ndim == 1 else Om
    return (np.asarray(Pi, dtype=float), np.asarray(Sigma, dtype=float), P,
            np.asarray(Q, dtype=float), Om)


def posterior_mean(Pi, Sigma, tau: float, Q, Omega, P=None) -> np.ndarray:
    """Black–Litterman posterior mean, update form (used by the strategies).

    μ_BL = Π + τΣPᵀ(PτΣPᵀ + Ω)⁻¹(Q − PΠ). Algebraically equal to the precision form
    of T6 (Woodbury identity) but defined even when Ω is singular, e.g. Ω = 0.

    Parameters
    ----------
    Pi : prior mean Π (n).
    Sigma : covariance Σ (n x n).
    tau : scale τ of the prior uncertainty.
    Q : view values (k).
    Omega : view covariance, (k x k) or its diagonal (k).
    P : pick matrix (k x n); None means P = I (an absolute view on every asset).

    Returns
    -------
    μ_BL as an array (n).

    Theorem: T6 (posterior mean); Cor. 6.1 (limits as Ω → ∞ and Ω → 0).
    """
    Pi, S, P, Q, Om = _arrays(Pi, Sigma, P, Q, Omega)
    tS = tau * S
    A = P @ tS @ P.T + Om
    return Pi + tS @ P.T @ np.linalg.solve(A, Q - P @ Pi)


def posterior_mean_bayes(Pi, Sigma, tau: float, Q, Omega, P=None) -> np.ndarray:
    """Posterior mean in precision form: [(τΣ)⁻¹ + PᵀΩ⁻¹P]⁻¹[(τΣ)⁻¹Π + PᵀΩ⁻¹Q].

    The normal–normal Bayesian update: prior μ ~ N(Π, τΣ), likelihood Q | μ ~ N(Pμ, Ω).
    Requires Ω invertible.

    Parameters
    ----------
    Same as ``posterior_mean``.

    Returns
    -------
    μ_BL as an array (n).

    Theorem: T6 (Bayesian derivation).
    """
    Pi, S, P, Q, Om = _arrays(Pi, Sigma, P, Q, Omega)
    prior_prec = np.linalg.inv(tau * S)
    view_prec = P.T @ np.linalg.solve(Om, P)
    rhs = prior_prec @ Pi + P.T @ np.linalg.solve(Om, Q)
    return np.linalg.solve(prior_prec + view_prec, rhs)


def posterior_mean_gls(Pi, Sigma, tau: float, Q, Omega, P=None) -> np.ndarray:
    """Posterior mean as generalized least squares on the stacked system.

    Treat the prior and the views as noisy observations of μ:
        [Π; Q] = [I; P] μ + [ε; η],   Cov([ε; η]) = blockdiag(τΣ, Ω),
    and solve μ̂ = (XᵀV⁻¹X)⁻¹XᵀV⁻¹y with X = [I; P], y = [Π; Q], V = blockdiag(τΣ, Ω).

    Parameters
    ----------
    Same as ``posterior_mean``.

    Returns
    -------
    μ_BL as an array (n).

    Theorem: T6 (GLS derivation; shown equal to the Bayesian one).
    """
    Pi, S, P, Q, Om = _arrays(Pi, Sigma, P, Q, Omega)
    n, k = len(Pi), len(Q)
    X = np.vstack([np.eye(n), P])
    y = np.concatenate([Pi, Q])
    V = np.zeros((n + k, n + k))
    V[:n, :n], V[n:, n:] = tau * S, Om
    Vinv_X = np.linalg.solve(V, X)
    return np.linalg.solve(X.T @ Vinv_X, Vinv_X.T @ y)


def posterior_covariance(Sigma, tau: float, Omega, P=None) -> np.ndarray:
    """Posterior covariance of μ: M = [(τΣ)⁻¹ + PᵀΩ⁻¹P]⁻¹ = τΣ − τΣPᵀ(PτΣPᵀ+Ω)⁻¹PτΣ.

    Parameters
    ----------
    Sigma, tau, Omega, P : as in ``posterior_mean``.

    Returns
    -------
    M (n x n), symmetric positive definite and ⪯ τΣ (views can only reduce
    uncertainty).

    Theorem: T6 (posterior variance).
    """
    S = np.asarray(Sigma, dtype=float)
    P = np.eye(len(S)) if P is None else np.asarray(P, dtype=float)
    Om = np.asarray(Omega, dtype=float)
    Om = np.diag(Om) if Om.ndim == 1 else Om
    tS = tau * S
    return tS - tS @ P.T @ np.linalg.solve(P @ tS @ P.T + Om, P @ tS)


# --------------------------------------------------------------------------- view uncertainty


def forecast_errors(forecasts: pd.DataFrame, excess_returns: pd.DataFrame) -> pd.DataFrame:
    """Realized forecast errors e = (realized excess return) − forecast.

    The forecast made at t' is for the month ending at the next month-end s'. Only
    forecasts whose month s' is present in ``excess_returns`` get an error, so passing
    returns sliced to <= t gives exactly the errors known at t.

    Parameters
    ----------
    forecasts : forecast date x ticker.
    excess_returns : monthly excess returns (month-end x ticker), e.g. sliced to <= t.

    Returns
    -------
    DataFrame indexed by forecast date t' (only realized ones) x ticker.

    Theorem: none (input to Ω-A).
    """
    me = excess_returns.index
    rows = {}
    for tp in forecasts.index:
        nxt = me[me > tp]
        if len(nxt):
            rows[tp] = excess_returns.loc[nxt[0], forecasts.columns] - forecasts.loc[tp]
    return pd.DataFrame(rows).T if rows else pd.DataFrame(columns=forecasts.columns)


def omega_a(errors: pd.DataFrame, return_var: pd.Series, window: int,
            prior_months: int) -> pd.Series:
    """Ω-A: mean squared past forecast error, shrunk toward the return variance.

    σ²ᵢ = (Σ_{last ``window`` errors} e²ᵢ + k·varᵢ) / (nᵢ + k), with k = ``prior_months``.
    varᵢ (stock i's return variance) is the error variance of a forecast with no
    skill, so it is the natural starting value; it dominates in the first months of
    the backtest, when few errors are known, and fades as errors accumulate.

    Parameters
    ----------
    errors : realized errors known at t (rows = forecast dates, all before t).
    return_var : each stock's monthly return variance at t (diagonal of Σ).
    window : number of most recent errors used.
    prior_months : k, weight of the return-variance starting value in months.

    Returns
    -------
    Ω-A diagonal (Series by ticker), strictly positive.

    Theorem: none (plan 2.4 / 4.4, Ω-A).
    """
    recent = errors.iloc[-window:].astype(float)
    sq = (recent ** 2).reindex(columns=return_var.index)
    n = sq.notna().sum()
    total = sq.sum().fillna(0.0)
    return ((total + prior_months * return_var) / (n + prior_months)).astype(float)


def omega_b(ensemble_var: pd.Series, omega_a_level: pd.Series | None = None,
            floor: float = 1e-12) -> pd.Series:
    """Ω-B: variance of the bootstrap copies' forecasts, raw or level-calibrated.

    Raw Ω-B measures how much the model's *estimate* would change with different
    training samples, which is typically far smaller than its forecast *error*, so
    raw Ω-B makes the views almost certain. With ``omega_a_level`` the values are
    rescaled so their cross-sectional mean equals that of Ω-A at the same date: the
    relative disagreement across stocks is kept and the overall level is taken from
    realized errors (D4.4).

    Parameters
    ----------
    ensemble_var : copies' forecast variance at t, by ticker.
    omega_a_level : Ω-A at t; None returns raw Ω-B (plus ``floor``).
    floor : added to every value so Ω stays positive when copies agree exactly.

    Returns
    -------
    Ω-B diagonal (Series by ticker).

    Theorem: none (plan 4.4, Ω-B).
    """
    v = ensemble_var + floor
    if omega_a_level is None:
        return v
    return v * (omega_a_level.reindex(v.index).mean() / v.mean())


# --------------------------------------------------------------------------- per-date inputs


def bl_inputs(hist, forecasts: pd.DataFrame, ensemble_var: pd.DataFrame | None,
              stocks: list[str], cfg: dict, omega_kind: str, tau: float,
              omega_scale: float = 1.0, vol_forecasts: pd.DataFrame | None = None) -> dict:
    """Everything Black–Litterman needs at rebalance date ``hist.t``, from data <= t.

    Parameters
    ----------
    hist : ``src.backtest.History`` (all tables sliced to <= t, plus close and shares).
    forecasts : ML forecasts (forecast date x ticker); row t is used for Q and rows
        before t (with realized outcomes) for Ω-A.
    ensemble_var : copies' forecast variance (date x ticker), needed for Ω-B.
    stocks : the universe.
    cfg : config (covariance window, δ, Ω-A settings).
    omega_kind : "A", "B" (raw) or "B-cal" (level-calibrated to Ω-A).
    tau : τ.
    omega_scale : multiplies Ω (ablation).
    vol_forecasts : ML log-volatility forecasts (date x ticker, ``src.risk_model``); if
        given, Σ = D·R·D from row t instead of the sample covariance (D6).

    Returns
    -------
    dict with Sigma, w_mkt, Pi, Q, Omega, mu_bl (all Series / DataFrame by ticker).

    Theorem: T6 (posterior used by strategy S6).
    """
    bl = cfg["black_litterman"]
    t = hist.t
    window, dpm = cfg["backtest"]["cov_window_days"], cfg["data"]["trading_days_per_year"] / 12
    if vol_forecasts is None:
        Sigma = sample_covariance(hist.returns_daily[stocks], t, window, dpm)
    else:
        Sigma = dr_covariance(hist.returns_daily, t, stocks, vol_forecasts.loc[t], window, dpm)
    w_mkt = market_weights(hist.close.loc[t, stocks], hist.shares[stocks])
    Pi = implied_prior(Sigma, w_mkt, bl["delta"])
    Q = forecasts.loc[t, stocks]

    excess = hist.returns_monthly[stocks].sub(hist.rf_monthly, axis=0)
    errors = forecast_errors(forecasts.loc[forecasts.index < t, stocks], excess)
    om_a = omega_a(errors, pd.Series(np.diag(Sigma.values), index=stocks),
                   bl["omega_a_window"], bl["omega_a_prior_months"])
    if omega_kind == "A":
        Omega = om_a
    elif omega_kind == "B":
        Omega = omega_b(ensemble_var.loc[t, stocks])
    elif omega_kind == "B-cal":
        Omega = omega_b(ensemble_var.loc[t, stocks], om_a)
    else:
        raise ValueError(f"Unknown omega kind {omega_kind!r}")
    Omega = Omega * omega_scale
    mu = posterior_mean(Pi.values, Sigma.values, tau, Q.values, Omega.values)
    return {"Sigma": Sigma, "w_mkt": w_mkt, "Pi": Pi, "Q": Q, "Omega": Omega,
            "mu_bl": pd.Series(mu, index=stocks)}

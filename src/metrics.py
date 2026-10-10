"""Performance and risk metrics for the backtest (plan Section 6, "Metrics").

Returns are simple returns per period (monthly unless stated). Losses are L = −r.
Tail risk follows plan Section 2.5: VaR_α is the α-quantile of the loss and CVaR_α
the expected loss in the worst (1 − α) share of outcomes, both estimated
historically from realized returns.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm


def annualized_return(r: pd.Series, periods_per_year: int = 12) -> float:
    """Geometric average return per year: (∏(1 + r))^(periods/N) − 1.

    Parameters
    ----------
    r : simple returns per period.
    periods_per_year : 12 for monthly, 252 for daily.

    Returns
    -------
    Annualized compound return.

    Theorem: none.
    """
    return float((1 + r).prod() ** (periods_per_year / len(r)) - 1)


def annualized_vol(r: pd.Series, periods_per_year: int = 12) -> float:
    """Sample standard deviation × √periods_per_year.

    Parameters
    ----------
    r : simple returns per period.
    periods_per_year : 12 for monthly, 252 for daily.

    Returns
    -------
    Annualized volatility.

    Theorem: none.
    """
    return float(r.std(ddof=1) * np.sqrt(periods_per_year))


def sharpe_ratio(r: pd.Series, rf: pd.Series, periods_per_year: int = 12) -> float:
    """Annualized Sharpe ratio of excess returns: √q · mean(r − r_f) / sd(r − r_f).

    Parameters
    ----------
    r : portfolio returns per period.
    rf : risk-free return for the same periods (aligned on the index).
    periods_per_year : q, 12 for monthly.

    Returns
    -------
    Annualized Sharpe ratio.

    Theorem: none (definition).
    """
    ex = (r - rf.reindex(r.index)).dropna()
    return float(np.sqrt(periods_per_year) * ex.mean() / ex.std(ddof=1))


def sharpe_confidence_interval(r: pd.Series, rf: pd.Series, level: float = 0.95,
                               periods_per_year: int = 12) -> tuple[float, float, float]:
    """Sharpe ratio with a confidence interval from Lo (2002), i.i.d. case.

    For T periods with per-period Sharpe SR, the asymptotic standard error is
    SE(SR) = √((1 + SR²/2) / T); annualizing multiplies SR and SE by √q.
    The formula is cited from Lo (2002); its proof is optional item O1.

    Parameters
    ----------
    r : portfolio returns per period.
    rf : risk-free returns for the same periods.
    level : confidence level, e.g. 0.95.
    periods_per_year : q.

    Returns
    -------
    (annualized Sharpe, lower bound, upper bound).

    Theorem: O1 (cited, not proved).
    """
    ex = (r - rf.reindex(r.index)).dropna()
    sr = ex.mean() / ex.std(ddof=1)
    se = np.sqrt((1 + 0.5 * sr ** 2) / len(ex))
    z = norm.ppf(0.5 + level / 2)
    k = np.sqrt(periods_per_year)
    return float(k * sr), float(k * (sr - z * se)), float(k * (sr + z * se))


def max_drawdown(r: pd.Series) -> float:
    """Largest peak-to-trough fall of cumulative wealth, as a positive fraction.

    Parameters
    ----------
    r : simple returns per period, in time order.

    Returns
    -------
    max over t of 1 − W_t / max_{s≤t} W_s, with W_0 = 1 before the first return.

    Theorem: none.
    """
    wealth = np.concatenate([[1.0], np.cumprod(1 + r.values)])
    return float(1 - (wealth / np.maximum.accumulate(wealth)).min())


def var_historical(r: pd.Series | np.ndarray, alpha: float = 0.95) -> float:
    """Historical Value-at-Risk: the α-quantile of the loss L = −r.

    Uses the lower quantile VaR_α = min{ℓ : F̂_L(ℓ) ≥ α} of the empirical
    distribution, the definition under which the Rockafellar–Uryasev minimum is
    attained at ζ = VaR_α.

    Parameters
    ----------
    r : returns (one per scenario / period).
    alpha : confidence level, e.g. 0.95.

    Returns
    -------
    VaR_α as a loss (positive = a loss).

    Theorem: T7 (VaR is the risk measure shown not to be subadditive).
    """
    return float(np.quantile(-np.asarray(r, dtype=float), alpha, method="inverted_cdf"))


def cvar_historical(r: pd.Series | np.ndarray, alpha: float = 0.95) -> float:
    """Historical CVaR via the Rockafellar–Uryasev formula evaluated at ζ = VaR_α.

    CVaR_α(L) = min_ζ { ζ + E[(L − ζ)⁺] / (1 − α) }, and the minimum is attained at
    ζ = VaR_α, so CVaR_α = VaR_α + E[(L − VaR_α)⁺] / (1 − α).

    Parameters
    ----------
    r : returns (one per scenario / period, equally likely).
    alpha : confidence level.

    Returns
    -------
    CVaR_α as a loss (positive = a loss); always ≥ VaR_α.

    Theorem: T7 (CVaR is coherent; Rockafellar–Uryasev representation).
    """
    loss = -np.asarray(r, dtype=float)
    v = var_historical(r, alpha)
    return float(v + np.mean(np.maximum(loss - v, 0.0)) / (1 - alpha))


def cvar_tail_average(r: pd.Series | np.ndarray, alpha: float = 0.95) -> float:
    """CVaR computed directly as the average of the worst (1 − α) share of losses.

    With N equally likely outcomes, k = (1 − α)N outcomes form the tail; when k is
    not an integer the boundary outcome gets fractional weight k − ⌊k⌋. Used to check
    ``cvar_historical`` (plan Section 7: tail average = Rockafellar–Uryasev).

    Parameters
    ----------
    r : returns (one per scenario / period).
    alpha : confidence level.

    Returns
    -------
    CVaR_α as a loss.

    Theorem: T7 (definition of CVaR as the expected tail loss).
    """
    loss = np.sort(-np.asarray(r, dtype=float))[::-1]
    k = (1 - alpha) * len(loss)
    whole = int(np.floor(k + 1e-9))
    frac = k - whole if k - whole > 1e-9 else 0.0
    total = loss[:whole].sum() + (frac * loss[whole] if frac else 0.0)
    return float(total / k)


def weight_stability(weights: pd.DataFrame) -> float:
    """Average L1 change between consecutive target portfolios, Σᵢ|w_{t,i} − w_{t−1,i}|.

    Unlike turnover it ignores price drift, so it measures only how much the
    optimizer changes its mind from month to month (linked to T5).

    Parameters
    ----------
    weights : target weights (rebalance date x asset).

    Returns
    -------
    Mean over rebalances after the first; 0 for a constant portfolio.

    Theorem: T5 (empirical side: unstable μ estimates give unstable weights).
    """
    return float(weights.diff().abs().sum(axis=1).iloc[1:].mean())


def performance_summary(r: pd.Series, rf: pd.Series, *, alpha: float = 0.95,
                        ci_level: float = 0.95, turnover: pd.Series | None = None,
                        weights: pd.DataFrame | None = None,
                        daily: pd.Series | None = None) -> dict[str, float]:
    """All plan metrics for one strategy.

    Parameters
    ----------
    r : monthly net returns.
    rf : monthly risk-free returns (aligned on the index).
    alpha : tail level for VaR / CVaR.
    ci_level : Sharpe confidence level.
    turnover : monthly turnover, optional; the first month (building the portfolio
        from cash) is left out of the average.
    weights : target weights over time, optional (weight stability).
    daily : daily net returns, optional (daily VaR / CVaR as a check).

    Returns
    -------
    dict: ann_return, ann_vol, sharpe, sharpe_lo, sharpe_hi, max_drawdown,
    var_m, cvar_m (monthly, as losses), and when given avg_turnover,
    weight_stability, var_d, cvar_d, max_drawdown_d (drawdown on the daily path,
    which also catches falls that recover within a month).

    Theorem: T7 (CVaR reported alongside variance); O1 (Sharpe CI, cited).
    """
    sr, lo, hi = sharpe_confidence_interval(r, rf, ci_level)
    out = {
        "ann_return": annualized_return(r),
        "ann_vol": annualized_vol(r),
        "sharpe": sr, "sharpe_lo": lo, "sharpe_hi": hi,
        "max_drawdown": max_drawdown(r),
        "var_m": var_historical(r, alpha),
        "cvar_m": cvar_historical(r, alpha),
    }
    if turnover is not None:
        out["avg_turnover"] = float(turnover.iloc[1:].mean())
    if weights is not None:
        out["weight_stability"] = weight_stability(weights)
    if daily is not None:
        out["var_d"] = var_historical(daily, alpha)
        out["cvar_d"] = cvar_historical(daily, alpha)
        out["max_drawdown_d"] = max_drawdown(daily)
    return out


# --------------------------------------------------------------------------- Sharpe differences


def sharpe_difference_memmel(r1: pd.Series, r2: pd.Series, rf: pd.Series,
                             periods_per_year: int = 12) -> dict[str, float]:
    """Test H0: SR1 = SR2 with the Jobson–Korkie statistic, Memmel (2003) correction.

    Uses the correlation ρ between the two strategies' excess returns, so it accounts
    for their dependence (unlike comparing two separate confidence intervals):
        θ = [2(1 − ρ) + ½(SR1² + SR2² − 2·SR1·SR2·ρ²)] / T,   z = (SR1 − SR2) / √θ,
    with per-period Sharpe ratios. Assumes i.i.d. normal returns; see
    ``sharpe_difference_bootstrap`` for a version without that assumption.

    Parameters
    ----------
    r1, r2 : returns of the two strategies over the same periods.
    rf : risk-free returns for those periods.
    periods_per_year : q, used to annualize the reported difference.

    Returns
    -------
    dict with ``diff`` (annualized SR1 − SR2), ``z``, ``p_value`` (two-sided), ``rho``.

    Theorem: none (statistical test; Jobson & Korkie 1981, Memmel 2003).
    """
    e1 = (r1 - rf.reindex(r1.index)).values
    e2 = (r2 - rf.reindex(r2.index)).values
    T = len(e1)
    s1, s2 = e1.mean() / e1.std(ddof=1), e2.mean() / e2.std(ddof=1)
    rho = np.corrcoef(e1, e2)[0, 1]
    theta = (2 * (1 - rho) + 0.5 * (s1 ** 2 + s2 ** 2 - 2 * s1 * s2 * rho ** 2)) / T
    z = (s1 - s2) / np.sqrt(theta)
    return {"diff": float(np.sqrt(periods_per_year) * (s1 - s2)), "z": float(z),
            "p_value": float(2 * norm.sf(abs(z))), "rho": float(rho)}


def sharpe_difference_bootstrap(r1: pd.Series, r2: pd.Series, rf: pd.Series, *,
                                block: int, n_boot: int, seed: int, level: float = 0.95,
                                periods_per_year: int = 12) -> dict[str, float]:
    """Paired circular block bootstrap test of H0: SR1 = SR2.

    Each bootstrap sample draws the *same* blocks of consecutive months for both
    strategies, which keeps their cross-dependence and short-run autocorrelation
    (in the spirit of Ledoit & Wolf 2008). The p-value compares |Δ* − Δ̂| with |Δ̂|,
    i.e. the bootstrap distribution re-centred under H0.

    Parameters
    ----------
    r1, r2 : returns of the two strategies over the same periods.
    rf : risk-free returns for those periods.
    block : block length in periods.
    n_boot : number of bootstrap samples.
    seed : random seed.
    level : confidence level of the percentile interval for the difference.
    periods_per_year : q, to annualize.

    Returns
    -------
    dict with ``diff`` (annualized SR1 − SR2), ``ci_lo``, ``ci_hi``, ``p_value``.

    Theorem: none (statistical test).
    """
    e = np.column_stack([(r1 - rf.reindex(r1.index)).values,
                         (r2 - rf.reindex(r2.index)).values])
    T = len(e)
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(T / block))
    starts = rng.integers(0, T, size=(n_boot, n_blocks))
    idx = ((starts[:, :, None] + np.arange(block)) % T).reshape(n_boot, -1)[:, :T]
    sample = e[idx]                                           # (n_boot, T, 2)
    sr = sample.mean(axis=1) / sample.std(axis=1, ddof=1)
    k = np.sqrt(periods_per_year)
    d_boot = k * (sr[:, 0] - sr[:, 1])
    d_hat = k * (e[:, 0].mean() / e[:, 0].std(ddof=1) - e[:, 1].mean() / e[:, 1].std(ddof=1))
    lo, hi = np.quantile(d_boot, [0.5 - level / 2, 0.5 + level / 2])
    p = float(np.mean(np.abs(d_boot - d_hat) >= abs(d_hat)))
    return {"diff": float(d_hat), "ci_lo": float(lo), "ci_hi": float(hi), "p_value": p}


def holm_adjust(p_values: list[float] | np.ndarray) -> np.ndarray:
    """Holm (1979) step-down adjustment for a family of tests.

    Controls the probability of any false rejection in the family at level α without
    assuming the tests are independent (they are not: the strategies are correlated).

    Parameters
    ----------
    p_values : raw p-values of the family.

    Returns
    -------
    Adjusted p-values, same order; reject when adjusted p < α.

    Theorem: none (multiple-testing correction).
    """
    p = np.asarray(p_values, dtype=float)
    m = len(p)
    order = np.argsort(p)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * p[i]))
        adj[i] = running
    return adj

"""EXPERIMENT (branch experiment/model-search): ways to improve how Black–Litterman uses ML
forecasts (DECISIONS BL1), rather than new forecasting models.

1. IC-scaled views (Grinold & Kahn: alpha = IC·σ·z). ML forecasts are used only for their
   ranking: z = normal score of the month's forecast ranks, σ = each stock's volatility,
   IC = the model's trailing rank IC (past months only, shrunk toward 0, floored at 0).
   Views are centred on the prior, Q = Π + IC·σ·z, with He–Litterman Ω = τ·diag(Σ).
   When the model has had no skill lately (IC ≤ 0), the views vanish.
2. Strong-signal views: as 1, but views only on the top and bottom 20% of the ranking
   (P = selected rows of I), in the spirit of confidence-gated views.
3. One relative view (ranking-based views; Yao, Li & Li 2024): P = +1/k on the k stocks
   ranked best, −1/k on the k ranked worst (k = 20% of stocks). Q = average realized
   top-minus-bottom spread of the model over the last ``window`` months, Ω = its squared
   standard error. A single, well-measured view instead of n noisy ones.
4. Tracking-error-targeted confidence (Shu & Mulvey 2024; "active risk targeting"):
   Ω is multiplied by the factor c that makes the BL portfolio's predicted tracking error
   against the prior-only portfolio equal a target (found by bisection on log c), so the
   ML can move the portfolio by a fixed, chosen amount of risk each month.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm, spearmanr

from src.backtest import History, _constraints, _sigma
from src.black_litterman import (forecast_errors, implied_prior, market_weights, omega_a,
                                 posterior_mean)
from src.optimization import mean_variance


def _realized_next(hist: History, stocks: list[str]) -> pd.DataFrame:
    """Excess return of the period after each date (row t' → return realized at t'+1),
    known only for t' < t."""
    ex = hist.returns_monthly[stocks].sub(hist.rf_monthly, axis=0)
    return ex.shift(-1)


def trailing_ic(f: pd.DataFrame, hist: History, stocks: list[str], window: int,
                prior: int) -> float:
    """Trailing rank IC of past forecasts, shrunk toward 0 with ``prior`` pseudo-periods,
    floored at 0: max(0, Σ IC / (n + prior))."""
    nxt = _realized_next(hist, stocks)
    past = f.loc[f.index < hist.t, stocks].index.intersection(nxt.dropna(how="all").index)
    past = past[-window:]
    ics = [spearmanr(f.loc[d, stocks], nxt.loc[d, stocks], nan_policy="omit")[0] for d in past]
    ics = [x for x in ics if np.isfinite(x)]
    return max(0.0, float(np.sum(ics)) / (len(ics) + prior))


def normal_scores(x: pd.Series) -> pd.Series:
    """Cross-sectional ranks → standard-normal quantiles, rescaled to unit std."""
    z = pd.Series(norm.ppf((x.rank() - 0.5) / x.notna().sum()), index=x.index)
    sd = z.std(ddof=0)
    return z / sd if sd > 0 else z * 0.0


def spread_history(f: pd.DataFrame, hist: History, stocks: list[str], frac: float,
                   window: int) -> pd.Series:
    """Realized top-minus-bottom spread of the model's past rankings (past only)."""
    nxt = _realized_next(hist, stocks)
    k = max(1, int(round(frac * len(stocks))))
    out = {}
    for d in f.loc[f.index < hist.t].index.intersection(nxt.dropna(how="all").index)[-window:]:
        order = f.loc[d, stocks].sort_values()
        out[d] = nxt.loc[d, order.index[-k:]].mean() - nxt.loc[d, order.index[:k]].mean()
    return pd.Series(out, dtype=float)


def _te(w1: pd.Series, w0: pd.Series, Sigma: pd.DataFrame) -> float:
    d = (w1 - w0).values
    return float(np.sqrt(max(d @ Sigma.values @ d, 0.0)))


def improved_bl_strategy(forecasts: pd.DataFrame, stocks: list[str], cfg: dict, gamma: float,
                         tau: float, method: str, te_target: float | None = None,
                         ic_window: int = 36, ic_prior: int = 12, frac: float = 0.2):
    """Black–Litterman strategy with one of the view constructions above.

    method : "omega_a" (Phase 4 views and Ω-A), "ic" (1), "ic_strong" (2), "relative" (3).
    te_target : if given (per period, e.g. 0.02/√12), Ω is rescaled each period to hit
        this tracking error against the prior-only portfolio (4).
    """
    bl = cfg["black_litterman"]
    solver = cfg["optimization"]["solver"]

    def views(hist, Sigma, Pi):
        t = hist.t
        sig = np.sqrt(np.diag(Sigma.values))
        if method == "omega_a":
            ex = hist.returns_monthly[stocks].sub(hist.rf_monthly, axis=0)
            err = forecast_errors(forecasts.loc[forecasts.index < t, stocks], ex)
            om = omega_a(err, pd.Series(sig ** 2, index=stocks), bl["omega_a_window"],
                         bl["omega_a_prior_months"]).values
            return np.eye(len(stocks)), forecasts.loc[t, stocks].values, om
        if method in ("ic", "ic_strong"):
            ic = trailing_ic(forecasts, hist, stocks, ic_window, ic_prior)
            if ic == 0.0:
                return None
            z = normal_scores(forecasts.loc[t, stocks]).values
            if not np.any(z):                        # all forecasts equal: no ranking
                return None
            keep = np.ones(len(stocks), bool)
            if method == "ic_strong":
                lo, hi = np.quantile(z, [frac, 1 - frac])
                keep = (z <= lo) | (z >= hi)
            P = np.eye(len(stocks))[keep]
            return P, (Pi.values + ic * sig * z)[keep], (tau * sig ** 2)[keep]
        if method == "relative":
            sp = spread_history(forecasts, hist, stocks, frac, ic_window)
            if len(sp) < ic_prior:
                return None
            k = max(1, int(round(frac * len(stocks))))
            order = forecasts.loc[t, stocks].sort_values()
            if order.iloc[0] == order.iloc[-1]:      # all forecasts equal: no ranking
                return None
            p = pd.Series(0.0, index=stocks)
            p[order.index[-k:]] = 1.0 / k
            p[order.index[:k]] = -1.0 / k
            return (p.values[None, :], np.array([sp.mean()]),
                    np.array([sp.var(ddof=1) / len(sp)]))
        raise ValueError(method)

    def strategy(hist: History) -> pd.Series:
        t = hist.t
        Sigma = _sigma(hist, stocks, cfg)
        w_mkt = market_weights(hist.close.loc[t, stocks], hist.shares[stocks])
        Pi = implied_prior(Sigma, w_mkt, bl["delta"])
        cons = _constraints(hist, cfg)
        v = views(hist, Sigma, Pi)
        if v is None:
            return mean_variance(Pi, Sigma, gamma, cons, solver).weights
        P, Q, om = v

        def solve(scale):
            mu = posterior_mean(Pi.values, Sigma.values, tau, Q, om * scale, P)
            return mean_variance(pd.Series(mu, index=stocks), Sigma, gamma, cons, solver).weights

        if te_target is None:
            return solve(1.0)
        w0 = mean_variance(Pi, Sigma, gamma, cons, solver).weights
        lo, hi = np.log(1e-3), np.log(1e4)          # small scale = more trust = larger TE
        w_lo = solve(np.exp(lo))
        if _te(w_lo, w0, Sigma) <= te_target:
            return w_lo
        for _ in range(25):
            mid = (lo + hi) / 2
            if _te(solve(np.exp(mid)), w0, Sigma) > te_target:
                lo = mid
            else:
                hi = mid
        return solve(np.exp(hi))
    return strategy

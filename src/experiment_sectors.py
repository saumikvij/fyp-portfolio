"""EXPERIMENT (branch experiment/model-search): industry rotation, MS15 strategy 2.

Rapach, Strauss, Tu & Zhou (2019): forecast each sector's next-month return from last
month's returns of all sectors. Here: 11 GICS sectors, equal-weighted portfolios of
the wide-universe stocks, one elastic net per sector, walk-forward with yearly
retraining and chronological validation (same procedure as every other model).

The forecasts enter Black–Litterman as sector views on the 51-stock portfolio: row s
of P averages the 51-universe stocks in sector s, Q_s is the sector forecast, and the
posterior uses the general-P form of T6.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.backtest import History, _constraints, _sigma
from src.black_litterman import (forecast_errors, implied_prior, market_weights, omega_a,
                                 posterior_mean)
from src.optimization import mean_variance


def sector_returns(wide: dict) -> pd.DataFrame:
    """Monthly excess returns of equal-weighted sector portfolios (month-end x sector)."""
    rm = wide["returns_monthly"]
    sec = wide["universe"]["sector"]
    stocks = [s for s in sec.index if s in rm.columns]
    ew = rm[stocks].T.groupby(sec.loc[stocks]).mean().T
    return ew.sub(wide["rf_monthly"]["rf"].reindex(ew.index), axis=0)


def sector_panels(sret: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """One panel per sector: features = last month's return of every sector (known at t),
    target = this sector's return over the next month, with its realization date."""
    me = sret.index
    nxt = pd.Series(list(me[1:]) + [pd.NaT], index=me)
    feats = sret.add_prefix("lag_")
    out = {}
    for s in sret.columns:
        p = feats.copy()
        p["target"] = sret[s].shift(-1)
        p["target_date"] = nxt.values
        p.index = pd.MultiIndex.from_arrays([me, [s] * len(me)], names=["date", "ticker"])
        out[s] = p
    return out


def sector_matrix(stocks: list[str], sectors: pd.Series, names: list[str]) -> np.ndarray:
    """P (sectors x stocks): row s = 1/n_s on the stocks of sector s."""
    P = np.zeros((len(names), len(stocks)))
    for i, s in enumerate(names):
        m = (sectors.reindex(stocks) == s).values
        P[i, m] = 1.0 / m.sum()
    return P


def sector_bl_strategy(sector_fc: pd.DataFrame, sret: pd.DataFrame, stocks: list[str],
                       cfg: dict, gamma: float, tau: float):
    """Black–Litterman with sector views (general P, T6) on the given stocks.

    Ω-A per sector = mean squared past forecast error (last ``omega_a_window`` realized
    months), shrunk toward the sector portfolio's variance, as for stock views (D4.4).
    """
    bl = cfg["black_litterman"]

    def strategy(hist: History) -> pd.Series:
        t = hist.t
        Sigma = _sigma(hist, stocks, cfg)
        w_mkt = market_weights(hist.close.loc[t, stocks], hist.shares[stocks])
        Pi = implied_prior(Sigma, w_mkt, bl["delta"])
        names = list(sector_fc.columns)
        P = sector_matrix(stocks, hist.sectors, names)
        keep = P.sum(axis=1) > 0                     # sectors present among the stocks
        P, names = P[keep], [n for n, k in zip(names, keep) if k]
        Q = sector_fc.loc[t, names].values
        errors = forecast_errors(sector_fc.loc[sector_fc.index < t, names], sret.loc[:t, names])
        var = pd.Series(np.diag(P @ Sigma.values @ P.T), index=names)
        Om = omega_a(errors, var, bl["omega_a_window"], bl["omega_a_prior_months"]).values
        mu = posterior_mean(Pi.values, Sigma.values, tau, Q, Om, P)
        return mean_variance(pd.Series(mu, index=stocks), Sigma, gamma, _constraints(hist, cfg),
                             cfg["optimization"]["solver"]).weights
    return strategy

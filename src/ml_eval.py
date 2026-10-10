"""Forecast evaluation (plan Section 4.5).

All functions take forecasts and realized targets as DataFrames indexed by forecast
date t (rows) and ticker (columns): ``forecast.loc[t, i]`` is the prediction made at
t of stock i's excess return over the next month, and ``realized.loc[t, i]`` is that
return once realized.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.inspection import permutation_importance

from src.features import FEATURES


def _align(forecast: pd.DataFrame, realized: pd.DataFrame):
    f, y = forecast.align(realized, join="inner")
    mask = f.notna() & y.notna()
    return f.where(mask), y.where(mask)


def r2_oos(forecast: pd.DataFrame, realized: pd.DataFrame,
           benchmark: pd.DataFrame | None = None) -> float:
    """Out-of-sample R², pooled over stocks and months.

    R²_oos = 1 − Σ(y − ŷ)² / Σ(y − b)², where b = 0 by default (Gu, Kelly & Xiu 2020)
    or another model's forecast (then it measures improvement over that model).

    Parameters
    ----------
    forecast, realized : date x ticker.
    benchmark : optional benchmark forecast, date x ticker; None means b = 0.

    Returns
    -------
    R²_oos (can be negative: the model is worse than the benchmark).

    Theorem: none.
    """
    f, y = _align(forecast, realized)
    b = 0.0 if benchmark is None else benchmark.reindex_like(y)
    sse = ((y - f) ** 2).sum().sum()
    sse_b = ((y - b) ** 2).where(y.notna()).sum().sum()
    return float(1 - sse / sse_b)


def rank_ic(forecast: pd.DataFrame, realized: pd.DataFrame) -> pd.Series:
    """Monthly Spearman rank correlation between forecasts and realized returns.

    Parameters
    ----------
    forecast, realized : date x ticker.

    Returns
    -------
    Series indexed by date (NaN when a month's forecasts are all equal).

    Theorem: none.
    """
    f, y = _align(forecast, realized)
    return f.rank(axis=1).corrwith(y.rank(axis=1), axis=1)


def newey_west_se(x: pd.Series, lags: int | None = None) -> float:
    """Newey–West (1987) standard error of the mean of a time series.

    Allows for autocorrelation up to ``lags`` periods with Bartlett weights. The
    default lag is ⌊4 (T/100)^(2/9)⌋, the usual rule of thumb.

    Parameters
    ----------
    x : series in time order (NaNs dropped).
    lags : number of lags; 0 gives the ordinary standard error (with ddof = 0).

    Returns
    -------
    Standard error of the sample mean.

    Theorem: none.
    """
    v = x.dropna().values
    T = len(v)
    if lags is None:
        lags = int(np.floor(4 * (T / 100) ** (2 / 9)))
    d = v - v.mean()
    s = d @ d / T
    for k in range(1, lags + 1):
        s += 2 * (1 - k / (lags + 1)) * (d[k:] @ d[:-k]) / T
    return float(np.sqrt(s / T))


def summarize_series(x: pd.Series, periods_per_year: int = 12) -> dict[str, float]:
    """Mean, sd, t-statistic (mean / (sd / √T)), Newey–West t-statistic, share > 0.

    Parameters
    ----------
    x : monthly series (e.g. rank IC or quintile spread).
    periods_per_year : unused; kept for symmetry with the other metrics.

    Returns
    -------
    dict with mean, sd, t_stat, t_stat_nw, share_positive, n.

    Theorem: none.
    """
    x = x.dropna()
    return {"mean": float(x.mean()), "sd": float(x.std(ddof=1)),
            "t_stat": float(x.mean() / (x.std(ddof=1) / np.sqrt(len(x)))),
            "t_stat_nw": float(x.mean() / newey_west_se(x)),
            "share_positive": float((x > 0).mean()), "n": int(len(x))}


def hit_rate(forecast: pd.DataFrame, realized: pd.DataFrame) -> dict[str, float]:
    """Share of stock-months where the forecast has the sign of the realized return.

    Returned with the base rate (share of positive realized returns), since a model
    that always predicts "up" scores that much for free.

    Parameters
    ----------
    forecast, realized : date x ticker.

    Returns
    -------
    dict with ``hit_rate`` and ``base_rate_up``.

    Theorem: none.
    """
    f, y = _align(forecast, realized)
    valid = f.notna()
    return {"hit_rate": float((np.sign(f) == np.sign(y))[valid].sum().sum() / valid.sum().sum()),
            "base_rate_up": float((y > 0)[valid].sum().sum() / valid.sum().sum())}


def quantile_spread(forecast: pd.DataFrame, realized: pd.DataFrame,
                    q: int) -> pd.DataFrame:
    """Each month, sort stocks into ``q`` groups by forecast; equal-weight group returns.

    Parameters
    ----------
    forecast, realized : date x ticker.
    q : number of groups (5 = quintiles; deciles are too thin with 51 stocks).

    Returns
    -------
    DataFrame (date x group 1..q, plus ``spread`` = top − bottom group), monthly.

    Theorem: none (economic test, plan Section 4.5).
    """
    f, y = _align(forecast, realized)
    rows = {}
    for t in f.index:
        ft, yt = f.loc[t].dropna(), y.loc[t].dropna()
        if ft.nunique() < q:
            continue
        groups = pd.qcut(ft.rank(method="first"), q, labels=range(1, q + 1))
        rows[t] = yt.groupby(groups, observed=True).mean()
    out = pd.DataFrame(rows).T
    out["spread"] = out[q] - out[1]
    return out


class _Predictor(RegressorMixin, BaseEstimator):
    """Minimal scikit-learn regressor wrapper so ``permutation_importance`` calls the
    model's own ``predict_array`` (which undoes any target scaling)."""

    def __init__(self, model):
        self.model = model

    def fit(self, X, y):          # required by scikit-learn's scorer checks; unused
        return self

    def predict(self, X):
        return self.model.predict_array(X)


def permutation_importance_oos(models: dict, panel: pd.DataFrame,
                               forecast_dates: pd.DatetimeIndex, n_repeats: int,
                               seed: int) -> pd.DataFrame:
    """Permutation importance on out-of-sample data, model by model.

    For each retrain, the model fitted then is scored on the months it actually
    forecast (until the next retrain); shuffling one feature at a time, the increase
    in mean squared error is that feature's importance. Results are averaged over
    blocks weighted by their number of rows.

    Parameters
    ----------
    models : retrain date -> fitted model with ``predict_array`` (any model in
        ``src.models`` except the historical mean).
    panel : rank-normalized features + targets, indexed by (date, ticker).
    forecast_dates : all forecast dates.
    n_repeats : shuffles per feature (scikit-learn ``permutation_importance``).
    seed : random seed.

    Returns
    -------
    DataFrame indexed by feature with ``importance`` (MSE increase) and ``sd``,
    sorted from most to least important.

    Theorem: none.
    """
    starts = sorted(models)
    dates = panel.index.get_level_values("date")
    imp, sd, weight = np.zeros(len(FEATURES)), np.zeros(len(FEATURES)), 0
    for k, start in enumerate(starts):
        end = starts[k + 1] if k + 1 < len(starts) else forecast_dates[-1] + pd.Timedelta(days=1)
        block = panel[(dates >= start) & (dates < end) & panel["target"].notna()]
        if block.empty:
            continue
        res = permutation_importance(_Predictor(models[start]), block[FEATURES].values,
                                     block["target"].values, n_repeats=n_repeats,
                                     random_state=seed, scoring="neg_mean_squared_error")
        imp += res.importances_mean * len(block)
        sd += res.importances_std * len(block)
        weight += len(block)
    return pd.DataFrame({"importance": imp / weight, "sd": sd / weight},
                        index=FEATURES).sort_values("importance", ascending=False)

"""ML risk model: forecast each stock's next-month volatility and build Σ from it (D6).

Target: log of the stock's realized daily volatility over the next month. Inputs, all
known at the month-end t and kept as levels (not ranked, since the level of risk is
what is forecast): log volatility over the last 5, 21, 63 and 252 trading days, log
high–low range volatility over 21 days (Parkinson 1980), and log market (SPY) volatility
over 21 days. Model: the Phase 3 elastic net, with the same grid, walk-forward,
yearly retraining and chronological validation. Baselines: trailing 63-day volatility
and a HAR-style regression on the 5/21/63-day volatilities (Corsi 2009).

The forecasts give the covariance Σ = D·R·D, with D = diag(forecast monthly
volatilities) and R the trailing 252-day correlation matrix. Σ is positive semidefinite
because R is (and D is diagonal), and diag(Σ) equals the forecast variances. Black–
Litterman uses this Σ everywhere it uses a covariance: the prior Π = δΣw_mkt, the
posterior (T6), Ω-A's shrinkage target, and the mean-variance step.

Developed and tested on the experiment branch ``experiment/model-search`` (MS15.3,
MS15.5, FN2); see DECISIONS.md D6.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.models import ElasticNetModel, walk_forward

VOL_FEATURES = ["lv_5", "lv_21", "lv_63", "lv_252", "lrange_21", "lmkt_21"]
HAR_FEATURES = ["lv_5", "lv_21", "lv_63"]


class HARModel:
    """Baseline: HAR-style least squares of next-month log volatility on the 1-week,
    1-month and 3-month log volatilities (Corsi 2009), refit yearly, no tuning."""

    name = "HAR"
    tuned = False

    def __init__(self, retrain_every: int = 12):
        self.retrain_every = retrain_every

    def fit(self, train: pd.DataFrame) -> None:
        X = np.column_stack([np.ones(len(train)), train[HAR_FEATURES].values])
        self.coef_ = np.linalg.lstsq(X, train["target"].values, rcond=None)[0]

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.column_stack([np.ones(len(X)), X[HAR_FEATURES].values]) @ self.coef_

    def clone(self, seed: int) -> "HARModel":
        return HARModel(self.retrain_every)


def load_high_low(raw_path: str | Path, calendar: pd.DatetimeIndex,
                  stocks: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split-adjusted daily High and Low prices on the trading calendar.

    Parameters
    ----------
    raw_path : the raw Yahoo download (``data/raw/prices_raw.parquet``).
    calendar : trading days (index of ``prices``).
    stocks : columns to keep.

    Returns
    -------
    (high, low), each date x stock, scaled by Adj Close / Close (the same split and
    dividend adjustment as the prices) and forward-filled for at most 5 days.

    Theorem: none.
    """
    raw = pd.read_parquet(raw_path)
    wide = {c: raw[c].unstack("Ticker").reindex(calendar)[stocks]
            for c in ("High", "Low", "Close", "Adj Close")}
    k = wide["Adj Close"] / wide["Close"]
    return ((wide["High"] * k).ffill(limit=5), (wide["Low"] * k).ffill(limit=5))


def vol_panel(data: dict, stocks: list[str], bench: str,
              high: pd.DataFrame, low: pd.DataFrame) -> pd.DataFrame:
    """Volatility features and the next-month log-volatility target at every month-end.

    Parameters
    ----------
    data : output of ``src.data.load_processed``.
    stocks : the universe.
    bench : market column (SPY).
    high, low : from ``load_high_low``.

    Returns
    -------
    DataFrame indexed by (date, ticker) with ``VOL_FEATURES``, ``target`` (log daily
    volatility over the trading days after t up to the next month-end) and
    ``target_date`` (that month-end). Features use only days <= t; months with less
    than a year of history are skipped.

    Theorem: none (no look-ahead by construction, tested).
    """
    r = data["returns_daily"]
    me = data["returns_monthly"].index
    nxt = pd.Series(list(me[1:]) + [pd.NaT], index=me)
    frames = {}
    for t in me:
        past = r.loc[:t]
        if len(past) < 253:
            continue
        f = pd.DataFrame(index=stocks)
        for w in (5, 21, 63, 252):
            f[f"lv_{w}"] = np.log(past[stocks].iloc[-w:].std() + 1e-8)
        h, lo = high.loc[:t, stocks], low.loc[:t, stocks]
        f["lrange_21"] = np.log(np.sqrt((np.log(h / lo) ** 2).iloc[-21:].mean()
                                        / (4 * np.log(2))) + 1e-8)
        f["lmkt_21"] = float(np.log(past[bench].iloc[-21:].std() + 1e-8))
        s = nxt[t]
        if pd.notna(s):
            fut = r.loc[(r.index > t) & (r.index <= s), stocks]
            f["target"] = np.log(fut.std() + 1e-8)
        else:
            f["target"] = np.nan
        f["target_date"] = s
        frames[t] = f
    return pd.concat(frames, names=["date", "ticker"])


def vol_forecasts(data: dict, stocks: list[str], dates: pd.DatetimeIndex, cfg: dict,
                  raw_path: str | Path | None = None, model=None,
                  panel: pd.DataFrame | None = None) -> pd.DataFrame:
    """Walk-forward forecasts of next-month log daily volatility (elastic net by default).

    Parameters
    ----------
    data : output of ``src.data.load_processed``.
    stocks : the universe.
    dates : forecast dates (rebalance dates).
    cfg : config (``ml`` settings: elastic-net grid, validation and training months).
    raw_path : raw download with High/Low; default ``<raw_dir>/prices_raw.parquet``.
    model : any Phase 3-style model (default: the elastic net on ``VOL_FEATURES``).
    panel : a precomputed ``vol_panel`` (built from ``raw_path`` if None).

    Returns
    -------
    DataFrame date x stock of forecast log daily volatility; row t uses data <= t only.

    Theorem: none (Phase 3 walk-forward, plan Section 4.3).
    """
    ml = cfg["ml"]
    if panel is None:
        raw_path = raw_path or Path(cfg["data"]["raw_dir"]) / "prices_raw.parquet"
        high, low = load_high_low(raw_path, data["prices"].index, stocks)
        panel = vol_panel(data, stocks, cfg["data"]["benchmark"], high, low)
    if model is None:
        e = ml["elastic_net"]
        model = ElasticNetModel(e["alphas"], e["l1_ratios"], e["max_iter"], cfg["seed"],
                                retrain_every=ml["retrain_every_months"], features=VOL_FEATURES)
    return walk_forward(model, panel, dates, ml["val_months"], ml["min_train_months"]).forecasts


def dr_covariance(returns_daily: pd.DataFrame, t: pd.Timestamp, stocks: list[str],
                  log_vol_forecast: pd.Series, window: int,
                  days_per_month: float) -> pd.DataFrame:
    """Σ = D·R·D from volatility forecasts and the trailing correlation.

    Parameters
    ----------
    returns_daily : daily returns (any rows; only those <= t are used).
    t : estimation date.
    stocks : the universe.
    log_vol_forecast : forecast log daily volatility per stock, made at t.
    window : trading days for the correlation matrix R.
    days_per_month : converts daily to monthly volatility (√ scaling).

    Returns
    -------
    Monthly covariance (stock x stock), positive semidefinite, with diagonal equal to
    the forecast monthly variances.

    Theorem: none (Σ = D R D is PSD since R is PSD and D diagonal).
    """
    x = returns_daily.loc[:t, stocks].iloc[-window:]
    R = x.corr().values
    d = np.exp(log_vol_forecast.reindex(stocks).values) * np.sqrt(days_per_month)
    return pd.DataFrame(R * np.outer(d, d), index=stocks, columns=stocks)

"""EXPERIMENT (branch experiment/model-search): round-3 model styles (DECISIONS MS7).

``ClassifierModel`` predicts whether a stock beats the month's median, then turns the
probability into a return forecast: forecast = a + k·(p − ½), where a is the average
training target and k the least-squares slope of the training target on (p − ½), both
fitted on the training rows only. Tuning and walk-forward are the same as for every
other model (validation MSE of the resulting return forecasts).
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from itertools import product

import numpy as np
import pandas as pd

from src.features import FEATURES


def _beats_median(df: pd.DataFrame) -> np.ndarray:
    med = df.groupby(level="date")["target"].transform("median")
    return (df["target"] > med).astype(int).values


@dataclass
class ClassifierModel:
    """Any scikit-learn classifier with ``predict_proba``, used as a return forecaster."""

    name: str
    factory: object
    grid: dict
    seed: int
    retrain_every: int = 12
    features: list[str] = field(default_factory=lambda: list(FEATURES))
    tuned: bool = True
    params_: dict = field(default_factory=dict)

    def _fit(self, params: dict, train: pd.DataFrame):
        clf = self.factory(**params, seed=self.seed).fit(train[self.features].values,
                                                         _beats_median(train))
        p = clf.predict_proba(train[self.features].values)[:, 1] - 0.5
        y = train["target"].values
        a = float(y.mean())
        k = float(p @ (y - a) / (p @ p)) if p @ p > 0 else 0.0
        return clf, a, k

    def _predict(self, fitted, X: np.ndarray) -> np.ndarray:
        clf, a, k = fitted
        return a + k * (clf.predict_proba(X)[:, 1] - 0.5)

    def tune(self, tune_set: pd.DataFrame, val_set: pd.DataFrame) -> dict:
        keys, best = list(self.grid), None
        for values in product(*(self.grid[k] for k in keys)):
            params = dict(zip(keys, values))
            pred = self._predict(self._fit(params, tune_set), val_set[self.features].values)
            mse = float(np.mean((pred - val_set["target"].values) ** 2))
            if best is None or mse < best["val_mse"]:
                best = {**params, "val_mse": mse}
        return best

    def fit(self, train: pd.DataFrame) -> None:
        self.fitted_ = self._fit(self.params_, train)

    def predict_array(self, X: np.ndarray) -> np.ndarray:
        return self._predict(self.fitted_, X)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.predict_array(X[self.features].values)

    def clone(self, seed: int) -> "ClassifierModel":
        c = copy.deepcopy(self)
        c.seed = seed
        return c


def shuffle_targets(panel: pd.DataFrame, seed: int) -> pd.DataFrame:
    """Placebo: shuffle each month's targets across stocks (structure kept, link removed)."""
    rng = np.random.default_rng(seed)
    out = panel.copy()
    for d, idx in out.groupby(level="date").indices.items():
        vals = out["target"].values[idx]
        out.iloc[idx, out.columns.get_loc("target")] = rng.permutation(vals)
    return out


class SubsampledRegressor:
    """Fit any regressor on a seeded random subsample of at most ``n`` rows.

    Used only where an exact solver does not scale (quantile regression on the wide
    universe, DECISIONS WU3); prediction is unchanged.
    """

    def __init__(self, base, n: int, seed: int):
        self.base, self.n, self.seed = base, n, seed

    def fit(self, X, y):
        if len(y) > self.n:
            idx = np.random.default_rng(self.seed).choice(len(y), self.n, replace=False)
            X, y = X[idx], y[idx]
        self.base.fit(X, y)
        return self

    def predict(self, X):
        return self.base.predict(X)

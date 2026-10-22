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


@dataclass
class IPCAModel:
    """Instrumented PCA (Kelly, Pruitt & Su 2019) as a return forecaster (MS14).

    r_{i,t+1} = z_{i,t}ᵀ Γ f_{t+1} + ε, z = signals + constant. Alternating least squares:
    f_t = (ΓᵀZ_tᵀZ_tΓ)⁻¹ ΓᵀZ_tᵀ r_t for each month, then vec(Γ) from
    Σ_t (Z_tᵀZ_t ⊗ f_t f_tᵀ) vec(Γ) = Σ_t (Z_tᵀ r_t ⊗ f_t). Forecast = z ᵀ Γ λ with λ the
    mean factor return in training. K (number of factors) is tuned on validation MSE.
    """

    grid: dict
    seed: int
    retrain_every: int = 12
    name: str = "IPCA"
    features: list[str] = field(default_factory=lambda: list(FEATURES))
    tuned: bool = True
    params_: dict = field(default_factory=dict)
    max_iter: int = 200
    tol: float = 1e-6

    def _design(self, df: pd.DataFrame) -> np.ndarray:
        return np.column_stack([df[self.features].values, np.ones(len(df))])

    def _fit(self, K: int, train: pd.DataFrame):
        Z = self._design(train)
        r = train["target"].values
        groups = list(train.groupby(level="date").indices.values())
        L = Z.shape[1]
        # Initialise Γ from the characteristic-managed portfolios X_t = Z_tᵀ r_t / N_t.
        X = np.column_stack([Z[g].T @ r[g] / len(g) for g in groups])
        U, _, _ = np.linalg.svd(X, full_matrices=False)
        G = U[:, :K]
        F = np.zeros((len(groups), K))
        for _ in range(self.max_iter):
            for j, g in enumerate(groups):
                ZG = Z[g] @ G
                F[j] = np.linalg.lstsq(ZG, r[g], rcond=None)[0]
            A = np.zeros((L * K, L * K))
            b = np.zeros(L * K)
            for j, g in enumerate(groups):
                ZZ = Z[g].T @ Z[g]
                A += np.kron(ZZ, np.outer(F[j], F[j]))
                b += np.kron(Z[g].T @ r[g], F[j])
            G_new = np.linalg.lstsq(A, b, rcond=None)[0].reshape(L, K)
            # Normalise: Γ orthonormal (factors rescaled accordingly).
            Q, R = np.linalg.qr(G_new)
            G_new, F = Q, F @ R.T
            done = np.max(np.abs(G_new - G)) < self.tol
            G = G_new
            if done:
                break
        return G, F.mean(axis=0)

    def _predict(self, fitted, df: pd.DataFrame) -> np.ndarray:
        G, lam = fitted
        return self._design(df) @ G @ lam

    def tune(self, tune_set: pd.DataFrame, val_set: pd.DataFrame) -> dict:
        best = None
        for K in self.grid["K"]:
            pred = self._predict(self._fit(K, tune_set), val_set)
            mse = float(np.mean((pred - val_set["target"].values) ** 2))
            if best is None or mse < best["val_mse"]:
                best = {"K": K, "val_mse": mse}
        return best

    def fit(self, train: pd.DataFrame) -> None:
        self.fitted_ = self._fit(self.params_["K"], train)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self._predict(self.fitted_, X)

    def predict_array(self, X: np.ndarray) -> np.ndarray:
        G, lam = self.fitted_
        return np.column_stack([X, np.ones(len(X))]) @ G @ lam

    def clone(self, seed: int) -> "IPCAModel":
        c = copy.deepcopy(self)
        c.seed = seed
        return c


@dataclass
class RankerModel:
    """Learning to rank (LambdaMART via LightGBM ``lambdarank``), MS15 strategy 1.

    Trained to order stocks within each month; labels are next-month return quintiles
    (0–4). The score becomes a return forecast a + k·score, with a = mean training
    target and k = least-squares slope of the training target on the score.
    """

    grid: dict
    seed: int
    retrain_every: int = 12
    name: str = "LTR"
    features: list[str] = field(default_factory=lambda: list(FEATURES))
    tuned: bool = True
    params_: dict = field(default_factory=dict)
    n_levels: int = 5

    def _labels_groups(self, df: pd.DataFrame):
        df = df.sort_index(level="date", sort_remaining=False)
        q = df.groupby(level="date")["target"].transform(
            lambda x: pd.qcut(x.rank(method="first"), self.n_levels, labels=False))
        groups = df.groupby(level="date", sort=False).size().values
        return df, q.astype(int).values, groups

    def _fit(self, params: dict, train: pd.DataFrame):
        import lightgbm as lgb
        df, y, groups = self._labels_groups(train)
        m = lgb.LGBMRanker(objective="lambdarank", learning_rate=0.05, min_child_samples=50,
                           subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                           random_state=self.seed, deterministic=True, force_row_wise=True,
                           n_jobs=1, verbose=-1, **params)
        m.fit(df[self.features].values, y, group=groups)
        s = m.predict(df[self.features].values)
        s = (s - s.mean()) / (s.std() or 1.0)
        tgt = df["target"].values
        a = float(tgt.mean())
        k = float(s @ (tgt - a) / (s @ s)) if s @ s > 0 else 0.0
        return m, a, k, s.mean(), s.std()

    def _predict(self, fitted, X: np.ndarray) -> np.ndarray:
        m, a, k, _, _ = fitted
        s = m.predict(X)
        s = (s - s.mean()) / (s.std() or 1.0)            # standardize within the month
        return a + k * s

    def tune(self, tune_set: pd.DataFrame, val_set: pd.DataFrame) -> dict:
        keys, best = list(self.grid), None
        for values in product(*(self.grid[k] for k in keys)):
            params = dict(zip(keys, values))
            f = self._fit(params, tune_set)
            pred = np.concatenate([self._predict(f, g[self.features].values)
                                   for _, g in val_set.groupby(level="date")])
            truth = np.concatenate([g["target"].values for _, g in val_set.groupby(level="date")])
            mse = float(np.mean((pred - truth) ** 2))
            if best is None or mse < best["val_mse"]:
                best = {**params, "val_mse": mse}
        return best

    def fit(self, train: pd.DataFrame) -> None:
        self.fitted_ = self._fit(self.params_, train)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self._predict(self.fitted_, X[self.features].values)

    def clone(self, seed: int) -> "RankerModel":
        c = copy.deepcopy(self)
        c.seed = seed
        return c

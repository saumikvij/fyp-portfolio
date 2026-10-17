"""Return-forecasting models and the chronological walk-forward procedure (plan 4.2–4.4).

Procedure at each forecast date t (a month-end):
1. Available data = panel rows whose target is realized by t (``target_date <= t``).
   Nothing realized after t can enter training.
2. On a retrain date (the first forecast date, then every ``retrain_every`` months),
   split the available feature months chronologically: the last ``val_months`` are
   the validation set, everything earlier is the tuning set. Each hyperparameter
   setting is fit on the tuning set and scored by mean squared error on validation
   (LightGBM and the MLP also use the validation months for early stopping); the best
   setting is then refit on all available rows (expanding window).
3. Optionally (Ω-B), fit ``n_copies`` extra copies of the chosen model, each on a
   bootstrap sample of the available months with its own seed.
4. Forecast next-month excess returns from the features dated t; with copies, also
   record the variance of the copies' forecasts for each stock.
Every retrain is recorded in a log (dates of each split, sizes, chosen parameters)
so the procedure can be audited and reported.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from itertools import product

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNet
from sklearn.neural_network import MLPRegressor

from src.features import FEATURES


def _mse(pred: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean((pred - y) ** 2))


class HistoricalMeanModel:
    """Baseline (no ML): each stock's average past excess return.

    Refit every month on all realized targets (expanding window). Stocks without
    history fall back to the cross-sectional average.
    """

    name = "Historical mean"
    retrain_every = 1
    tuned = False

    def fit(self, train: pd.DataFrame) -> None:
        self.means_ = train.groupby(level="ticker")["target"].mean()
        self.overall_ = float(train["target"].mean())

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        tickers = X.index.get_level_values("ticker")
        return self.means_.reindex(tickers).fillna(self.overall_).values

    def clone(self, seed: int) -> "HistoricalMeanModel":
        return HistoricalMeanModel()


@dataclass
class ElasticNetModel:
    """Elastic net on the rank-normalized features (ridge and lasso as special cases).

    Hyperparameters α (penalty strength) and l1_ratio (lasso share) are chosen on the
    chronological validation split; ``seed`` fixes the coordinate-descent order.
    """

    alphas: list[float]
    l1_ratios: list[float]
    max_iter: int
    seed: int
    retrain_every: int = 12
    name: str = "Elastic net"
    features: list[str] = field(default_factory=lambda: list(FEATURES))
    tuned: bool = True
    params_: dict = field(default_factory=dict)

    def _make(self, alpha: float, l1_ratio: float) -> ElasticNet:
        return ElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=self.max_iter,
                          random_state=self.seed, selection="cyclic")

    def tune(self, tune_set: pd.DataFrame, val_set: pd.DataFrame) -> dict:
        """Grid search on (α, l1_ratio) by validation MSE; returns the best setting."""
        best = None
        for a, l1 in product(self.alphas, self.l1_ratios):
            m = self._make(a, l1).fit(tune_set[self.features].values, tune_set["target"].values)
            mse = _mse(m.predict(val_set[self.features].values), val_set["target"].values)
            if best is None or mse < best["val_mse"]:
                best = {"alpha": a, "l1_ratio": l1, "val_mse": mse}
        return best

    def fit(self, train: pd.DataFrame) -> None:
        self.model_ = self._make(self.params_["alpha"], self.params_["l1_ratio"])
        self.model_.fit(train[self.features].values, train["target"].values)

    def predict_array(self, X: np.ndarray) -> np.ndarray:
        return self.model_.predict(X)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.predict_array(X[self.features].values)

    def coefficients(self) -> dict:
        return dict(zip(self.features, self.model_.coef_))

    def clone(self, seed: int) -> "ElasticNetModel":
        c = copy.deepcopy(self)
        c.seed = seed
        return c


@dataclass
class LightGBMModel:
    """Gradient-boosted trees (LightGBM) with early stopping on the validation months.

    Grid over ``num_leaves`` x ``min_child_samples``; for each setting the number of
    trees is chosen by early stopping on validation MSE. The final refit on all
    available months uses the chosen number of trees.
    """

    num_leaves: list[int]
    min_child_samples: list[int]
    learning_rate: float
    max_trees: int
    early_stopping_rounds: int
    feature_fraction: float
    bagging_fraction: float
    bagging_freq: int
    lambda_l2: float
    seed: int
    retrain_every: int = 12
    name: str = "LightGBM"
    features: list[str] = field(default_factory=lambda: list(FEATURES))
    tuned: bool = True
    params_: dict = field(default_factory=dict)

    def _make(self, num_leaves: int, min_child_samples: int, n_trees: int):
        return lgb.LGBMRegressor(
            n_estimators=n_trees, learning_rate=self.learning_rate, num_leaves=num_leaves,
            min_child_samples=min_child_samples, colsample_bytree=self.feature_fraction,
            subsample=self.bagging_fraction, subsample_freq=self.bagging_freq,
            reg_lambda=self.lambda_l2, random_state=self.seed, deterministic=True,
            force_row_wise=True, n_jobs=1, verbose=-1)

    def tune(self, tune_set: pd.DataFrame, val_set: pd.DataFrame) -> dict:
        """Grid search; early stopping on the validation months picks the tree count."""
        Xt, yt = tune_set[self.features].values, tune_set["target"].values
        Xv, yv = val_set[self.features].values, val_set["target"].values
        best = None
        for nl, mcs in product(self.num_leaves, self.min_child_samples):
            m = self._make(nl, mcs, self.max_trees)
            m.fit(Xt, yt, eval_X=(Xv,), eval_y=(yv,), eval_metric="l2",
                  callbacks=[lgb.early_stopping(self.early_stopping_rounds, verbose=False)])
            n_trees = max(int(m.best_iteration_ or self.max_trees), 1)
            mse = _mse(m.predict(Xv, num_iteration=n_trees), yv)
            if best is None or mse < best["val_mse"]:
                best = {"num_leaves": nl, "min_child_samples": mcs, "n_trees": n_trees,
                        "val_mse": mse}
        return best

    def fit(self, train: pd.DataFrame) -> None:
        p = self.params_
        self.model_ = self._make(p["num_leaves"], p["min_child_samples"], p["n_trees"])
        self.model_.fit(train[self.features].values, train["target"].values)

    def predict_array(self, X: np.ndarray) -> np.ndarray:
        return self.model_.predict(X)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.predict_array(X[self.features].values)

    def clone(self, seed: int) -> "LightGBMModel":
        c = copy.deepcopy(self)
        c.seed = seed
        return c


@dataclass
class MLPModel:
    """Small feed-forward network (scikit-learn MLPRegressor, 2–3 hidden layers).

    Early stopping uses the *chronological* validation months: the network is trained
    one epoch at a time and the epoch count with the lowest validation MSE is kept
    (scikit-learn's own early stopping would hold out a random split, which the plan
    rules out). Targets are scaled to unit variance for training and scaled back.
    """

    hidden_layers: list[list[int]]
    alphas: list[float]
    learning_rate_init: float
    batch_size: int
    max_epochs: int
    patience: int
    seed: int
    retrain_every: int = 12
    name: str = "MLP"
    features: list[str] = field(default_factory=lambda: list(FEATURES))
    tuned: bool = True
    params_: dict = field(default_factory=dict)

    def _make(self, hidden: tuple[int, ...], alpha: float, n_rows: int) -> MLPRegressor:
        return MLPRegressor(hidden_layer_sizes=hidden, alpha=alpha, solver="adam",
                            learning_rate_init=self.learning_rate_init,
                            batch_size=min(self.batch_size, n_rows), shuffle=True,
                            random_state=self.seed)

    def tune(self, tune_set: pd.DataFrame, val_set: pd.DataFrame) -> dict:
        """Grid over (layers, α); epochs chosen by chronological early stopping."""
        Xt, Xv = tune_set[self.features].values, val_set[self.features].values
        scale = float(tune_set["target"].std())
        yt, yv = tune_set["target"].values / scale, val_set["target"].values
        best = None
        for hidden, alpha in product(self.hidden_layers, self.alphas):
            m = self._make(tuple(hidden), alpha, len(Xt))
            best_ep, best_mse, since = 0, np.inf, 0
            for ep in range(1, self.max_epochs + 1):
                m.partial_fit(Xt, yt)
                mse = _mse(m.predict(Xv) * scale, yv)
                if mse < best_mse:
                    best_ep, best_mse, since = ep, mse, 0
                else:
                    since += 1
                    if since >= self.patience:
                        break
            if best is None or best_mse < best["val_mse"]:
                best = {"hidden": tuple(hidden), "alpha": alpha, "epochs": best_ep,
                        "val_mse": best_mse}
        return best

    def fit(self, train: pd.DataFrame) -> None:
        p = self.params_
        self.scale_ = float(train["target"].std())
        self.model_ = self._make(tuple(p["hidden"]), p["alpha"], len(train))
        X, y = train[self.features].values, train["target"].values / self.scale_
        for _ in range(p["epochs"]):
            self.model_.partial_fit(X, y)

    def predict_array(self, X: np.ndarray) -> np.ndarray:
        return self.model_.predict(X) * self.scale_

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.predict_array(X[self.features].values)

    def clone(self, seed: int) -> "MLPModel":
        c = copy.deepcopy(self)
        c.seed = seed
        return c


@dataclass
class RandomForestModel:
    """Random forest (scikit-learn): many deep trees on bootstrap samples, averaged.

    Grid over ``max_depth`` x ``min_samples_leaf`` chosen by validation MSE on the
    chronological validation months; ``max_features`` fixed. Added for the model search
    (DECISIONS MS2).
    """

    n_estimators: int
    max_depth: list[int]
    min_samples_leaf: list[int]
    max_features: float
    seed: int
    retrain_every: int = 12
    name: str = "Random forest"
    features: list[str] = field(default_factory=lambda: list(FEATURES))
    tuned: bool = True
    params_: dict = field(default_factory=dict)

    def _make(self, depth: int, leaf: int) -> RandomForestRegressor:
        return RandomForestRegressor(n_estimators=self.n_estimators, max_depth=depth,
                                     min_samples_leaf=leaf, max_features=self.max_features,
                                     random_state=self.seed, n_jobs=-1)

    def tune(self, tune_set: pd.DataFrame, val_set: pd.DataFrame) -> dict:
        """Grid search on (max_depth, min_samples_leaf) by validation MSE."""
        best = None
        for depth, leaf in product(self.max_depth, self.min_samples_leaf):
            m = self._make(depth, leaf).fit(tune_set[self.features].values,
                                            tune_set["target"].values)
            mse = _mse(m.predict(val_set[self.features].values), val_set["target"].values)
            if best is None or mse < best["val_mse"]:
                best = {"max_depth": depth, "min_samples_leaf": leaf, "val_mse": mse}
        return best

    def fit(self, train: pd.DataFrame) -> None:
        p = self.params_
        self.model_ = self._make(p["max_depth"], p["min_samples_leaf"])
        self.model_.fit(train[self.features].values, train["target"].values)

    def predict_array(self, X: np.ndarray) -> np.ndarray:
        return self.model_.predict(X)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.predict_array(X[self.features].values)

    def clone(self, seed: int) -> "RandomForestModel":
        c = copy.deepcopy(self)
        c.seed = seed
        return c


@dataclass
class GridModel:
    """Any scikit-learn regressor with a small hyperparameter grid (model search, MS4).

    ``factory(**params, seed=seed)`` builds the estimator; ``grid`` maps each parameter
    to its candidate values. Tuned by validation MSE on the chronological validation
    months, like the other models.
    """

    name: str
    factory: object
    grid: dict
    seed: int
    retrain_every: int = 12
    features: list[str] = field(default_factory=lambda: list(FEATURES))
    tuned: bool = True
    params_: dict = field(default_factory=dict)

    def tune(self, tune_set: pd.DataFrame, val_set: pd.DataFrame) -> dict:
        """Try every combination in ``grid``; keep the lowest validation MSE."""
        keys, best = list(self.grid), None
        for values in product(*(self.grid[k] for k in keys)):
            params = dict(zip(keys, values))
            m = self.factory(**params, seed=self.seed).fit(tune_set[self.features].values,
                                                           tune_set["target"].values)
            mse = _mse(m.predict(val_set[self.features].values), val_set["target"].values)
            if best is None or mse < best["val_mse"]:
                best = {**params, "val_mse": mse}
        return best

    def fit(self, train: pd.DataFrame) -> None:
        self.model_ = self.factory(**self.params_, seed=self.seed)
        self.model_.fit(train[self.features].values, train["target"].values)

    def predict_array(self, X: np.ndarray) -> np.ndarray:
        return self.model_.predict(X)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.predict_array(X[self.features].values)

    def clone(self, seed: int) -> "GridModel":
        c = copy.deepcopy(self)
        c.seed = seed
        return c


@dataclass
class WalkForwardResult:
    """Forecasts and, with copies, their disagreement (both forecast date x ticker),
    plus the retrain log and the fitted models."""

    forecasts: pd.DataFrame
    log: pd.DataFrame
    models: dict
    ensemble_var: pd.DataFrame | None = None


def _bootstrap_months(avail: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Resample whole feature months with replacement (keeps each cross-section)."""
    months = avail.index.get_level_values("date").unique()
    pick = rng.choice(len(months), size=len(months), replace=True)
    groups = dict(tuple(avail.groupby(level="date")))
    return pd.concat([groups[months[i]] for i in pick])


def walk_forward(model, panel: pd.DataFrame, forecast_dates: pd.DatetimeIndex,
                 val_months: int, min_train_months: int, n_copies: int = 0,
                 seed: int = 0, train_window_months: int | None = None,
                 target_mode: str = "raw") -> WalkForwardResult:
    """Run the chronological train / validate / forecast loop described above.

    Parameters
    ----------
    model : a model from this module (``fit``, ``predict``, ``clone``,
        ``retrain_every``, ``tuned`` and, if tuned, ``tune`` and ``params_``).
    panel : rank-normalized features + ``target`` + ``target_date``, indexed by
        (date, ticker).
    forecast_dates : month-ends at which forecasts are made.
    val_months : number of most recent feature months used for validation.
    min_train_months : minimum feature months in the tuning set.
    n_copies : number of bootstrap copies for Ω-B (0 = none).
    seed : base seed for the copies (copy k uses seed + k + 1).
    train_window_months : None = expanding window; otherwise only the most recent
        this many available feature months are used (rolling window, MS2).
    target_mode : "raw" = learn next-month excess returns; "demeaned" = learn excess
        returns minus each month's cross-sectional average (which stocks beat the
        others). Demeaned forecasts are returned as levels by adding ``level`` = the
        average realized excess return in the training window (known at t).

    Returns
    -------
    WalkForwardResult. ``log`` has one row per retrain with: retrain_date,
    train_start / train_end (first and last feature month fitted), tune_start /
    tune_end and val_start / val_end (the tuning split), last_target_date (latest
    realized target used), n_rows, the chosen hyperparameters with their validation
    MSE, and coefficients when the model has them. ``ensemble_var`` holds the variance
    of the copies' forecasts (Ω-B), or None without copies.

    Raises
    ------
    ValueError if a retrain has fewer than ``min_train_months`` tuning months.

    Theorem: none (plan Section 4.3–4.4; no look-ahead, tested).
    """
    dates = panel.index.get_level_values("date")
    forecasts, ens_var, log, models = {}, {}, [], {}
    copies: list = []
    last_retrain = None
    for t in forecast_dates:
        due = last_retrain is None or (t.to_period("M") - last_retrain.to_period("M")).n \
            >= model.retrain_every
        if due:
            avail = panel[(panel["target_date"] <= t) & panel["target"].notna()]
            months = avail.index.get_level_values("date").unique().sort_values()
            if train_window_months is not None:
                months = months[-train_window_months:]
                avail = avail[avail.index.get_level_values("date").isin(months)]
            level = 0.0
            if target_mode == "demeaned":
                level = float(avail["target"].mean())
                avail = avail.copy()
                avail["target"] -= avail.groupby(level="date")["target"].transform("mean")
            elif target_mode != "raw":
                raise ValueError(f"Unknown target_mode {target_mode!r}")
            row = {"retrain_date": t, "train_start": months[0], "train_end": months[-1],
                   "last_target_date": avail["target_date"].max(), "n_rows": len(avail),
                   "level": level}
            if model.tuned:
                val_m, tune_m = months[-val_months:], months[:-val_months]
                if len(tune_m) < min_train_months:
                    raise ValueError(f"Only {len(tune_m)} tuning months at {t.date()}")
                av_dates = avail.index.get_level_values("date")
                best = model.tune(avail[av_dates.isin(tune_m)], avail[av_dates.isin(val_m)])
                model.params_ = {k: v for k, v in best.items() if k != "val_mse"}
                row.update(tune_start=tune_m[0], tune_end=tune_m[-1], val_start=val_m[0],
                           val_end=val_m[-1], **best)
            model.fit(avail)
            if hasattr(model, "coefficients"):
                row.update({f"coef_{f}": c for f, c in model.coefficients().items()})
            models[t] = copy.deepcopy(model)
            if n_copies:
                rng = np.random.default_rng(seed + t.year * 100 + t.month)
                copies = []
                for k in range(n_copies):
                    c = model.clone(seed + k + 1)
                    c.fit(_bootstrap_months(avail, rng))
                    copies.append(c)
            log.append(row)
            last_retrain = t
        X_t = panel[dates == t]
        tickers = X_t.index.get_level_values("ticker")
        forecasts[t] = pd.Series(model.predict(X_t) + level, index=tickers)
        if n_copies:
            preds = np.vstack([c.predict(X_t) for c in copies])
            ens_var[t] = pd.Series(preds.var(axis=0, ddof=1), index=tickers)
    return WalkForwardResult(
        pd.DataFrame(forecasts).T.rename_axis("date"), pd.DataFrame(log), models,
        pd.DataFrame(ens_var).T.rename_axis("date") if n_copies else None)

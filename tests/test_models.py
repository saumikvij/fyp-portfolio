"""Phase 3 tests: models, the chronological procedure, leakage, and forecast metrics."""

import numpy as np
import pandas as pd
import pytest

from src.data import load_config
from src.features import (FEATURE_SETS, FEATURES, add_targets, build_feature_panel,
                          rank_normalize)
from src.ml_eval import (hit_rate, newey_west_se, quantile_spread, r2_oos, rank_ic,
                         summarize_series)
from src.models import (ElasticNetModel, HistoricalMeanModel, LightGBMModel, MLPModel,
                        RandomForestModel, walk_forward)
from tests.pit import perturb_after
from tests.toy import make_toy_data

CFG = load_config()
W, RSI = CFG["ml"]["windows"], CFG["ml"]["rsi_days"]
EN = CFG["ml"]["elastic_net"]
VAL, MIN_TRAIN = 6, 6          # small splits for the ~4-year toy data


def toy_panel(data):
    stocks = [c for c in data["prices"].columns if c != "SPY"]
    panel = rank_normalize(build_feature_panel(data, stocks, "SPY", W, RSI))
    return add_targets(panel, data["returns_monthly"][stocks], data["rf_monthly"]["rf"])


def enet():
    return ElasticNetModel(EN["alphas"], EN["l1_ratios"], EN["max_iter"], CFG["seed"],
                           retrain_every=12)


def lgbm(max_trees=300):
    g = CFG["ml"]["lightgbm"]
    return LightGBMModel(g["num_leaves"], [20], g["learning_rate"] * 5, max_trees, 30,
                         g["feature_fraction"], g["bagging_fraction"], g["bagging_freq"],
                         g["lambda_l2"], CFG["seed"])


def mlp(max_epochs=30):
    m = CFG["ml"]["mlp"]
    return MLPModel(m["hidden_layers"][:1], m["alphas"][:1], m["learning_rate_init"],
                    m["batch_size"], max_epochs, 5, CFG["seed"])


TUNED = [enet, lgbm, mlp]


@pytest.fixture(scope="module")
def toy():
    return make_toy_data()


@pytest.fixture(scope="module")
def panel(toy):
    return toy_panel(toy)


@pytest.fixture(scope="module")
def dates(panel):
    return panel.index.get_level_values("date").unique().sort_values()[MIN_TRAIN + VAL + 2:]


# --------------------------------------------------------------------------- leakage


@pytest.mark.parametrize("model_fn", TUNED)
def test_leakage_training_targets_realized_by_retrain_date(panel, dates, model_fn):
    """Plan Section 7: no target realized after t is used by the model used at t."""
    log = walk_forward(model_fn(), panel, dates, VAL, MIN_TRAIN).log
    assert len(log) >= 2
    assert (log["last_target_date"] <= log["retrain_date"]).all()
    assert (log["train_end"] < log["retrain_date"]).all()


@pytest.mark.parametrize("model_fn", TUNED)
def test_leakage_validation_strictly_before_forecast_and_after_tuning(panel, dates, model_fn):
    """Plan Section 7: the validation split is strictly earlier than the forecast date,
    and strictly later than the data the candidates are fitted on."""
    log = walk_forward(model_fn(), panel, dates, VAL, MIN_TRAIN).log
    assert (log["val_end"] < log["retrain_date"]).all()
    assert (log["tune_end"] < log["val_start"]).all()
    assert (log["val_end"] == log["train_end"]).all()   # final refit adds the val months


def test_retrain_schedule(panel, dates):
    res = walk_forward(enet(), panel, dates, VAL, MIN_TRAIN)
    gaps = res.log["retrain_date"].dt.to_period("M").diff().dropna().apply(lambda d: d.n)
    assert (gaps == 12).all()
    assert list(res.forecasts.index) == list(dates)


@pytest.mark.parametrize("model_fn", TUNED + [HistoricalMeanModel])
def test_forecasts_unchanged_by_future_data(toy, dates, model_fn):
    """End to end: scramble all raw data after t, rebuild features and targets,
    re-run the walk-forward (with Ω-B copies): every forecast and every ensemble
    variance made at or before t is identical."""
    t = dates[len(dates) // 2]
    moved = {k: (perturb_after(v, t) if k not in ("sectors", "shares_outstanding") else v) for k, v in toy.items()}
    a = walk_forward(model_fn(), toy_panel(toy), dates, VAL, MIN_TRAIN, n_copies=3, seed=1)
    b = walk_forward(model_fn(), toy_panel(moved), dates, VAL, MIN_TRAIN, n_copies=3, seed=1)
    pd.testing.assert_frame_equal(a.forecasts.loc[:t], b.forecasts.loc[:t])
    pd.testing.assert_frame_equal(a.ensemble_var.loc[:t], b.ensemble_var.loc[:t])
    assert not a.forecasts.loc[t:].iloc[1:].equals(b.forecasts.loc[t:].iloc[1:])


def test_too_little_training_data_raises(panel):
    early = panel.index.get_level_values("date").unique().sort_values()[[4]]
    with pytest.raises(ValueError, match="tuning months"):
        walk_forward(enet(), panel, early, VAL, MIN_TRAIN)


# --------------------------------------------------------------------------- models


def test_historical_mean_is_average_of_realized_targets(panel, dates):
    t = dates[5]
    f = walk_forward(HistoricalMeanModel(), panel, dates, VAL, MIN_TRAIN).forecasts.loc[t]
    realized = panel[panel["target_date"] <= t]
    np.testing.assert_allclose(f.values, realized.groupby(level="ticker")["target"].mean()
                               .reindex(f.index).values)


def test_tuning_picks_lowest_validation_mse(panel):
    m = enet()
    d = panel.dropna(subset=["target"])
    months = d.index.get_level_values("date").unique()
    tune = d[d.index.get_level_values("date").isin(months[:-VAL])]
    val = d[d.index.get_level_values("date").isin(months[-VAL:])]
    best = m.tune(tune, val)
    for a in EN["alphas"]:
        for l1 in EN["l1_ratios"]:
            fit = m._make(a, l1).fit(tune[FEATURES].values, tune["target"].values)
            mse = np.mean((fit.predict(val[FEATURES].values) - val["target"].values) ** 2)
            assert best["val_mse"] <= mse + 1e-15


def test_elastic_net_recovers_a_planted_signal():
    """Targets driven by one feature: the model finds it and forecasts rank stocks well."""
    rng = np.random.default_rng(0)
    dates = pd.date_range("2012-01-31", periods=80, freq="ME")
    idx = pd.MultiIndex.from_product([dates, [f"S{i}" for i in range(40)]],
                                     names=["date", "ticker"])
    X = pd.DataFrame(rng.uniform(-0.5, 0.5, (len(idx), len(FEATURES))), index=idx,
                     columns=FEATURES)
    X["target"] = 0.04 * X["mom_12_1"] + rng.normal(0, 0.02, len(idx))
    nxt = dict(zip(dates[:-1], dates[1:]))
    X["target_date"] = [nxt.get(d, pd.NaT) for d in idx.get_level_values("date")]
    fdates = dates[40:-1]
    res = walk_forward(enet(), X, fdates, 12, 12)
    coefs = res.log.filter(like="coef_")
    assert (coefs["coef_mom_12_1"] > 0.02).all()
    realized = X.loc[fdates, "target"].unstack()
    # Best achievable: signal sd 0.04/√12 ≈ 0.0115 vs noise 0.02 -> corr ≈ 0.50, R² ≈ 0.25.
    signal_sd = 0.04 / np.sqrt(12)
    best_corr = signal_sd / np.hypot(signal_sd, 0.02)
    assert rank_ic(res.forecasts, realized).mean() > 0.85 * best_corr
    assert r2_oos(res.forecasts, realized) > 0.8 * best_corr ** 2


# --------------------------------------------------------------------------- metrics


def grid(values, dates=("2020-01-31", "2020-02-29")):
    return pd.DataFrame(values, index=pd.to_datetime(list(dates)), columns=list("ABCDE"))


def test_r2_oos():
    y = grid([[0.1, -0.2, 0.05, 0.0, 0.3], [0.02, 0.01, -0.1, 0.2, -0.05]])
    assert r2_oos(y, y) == 1.0
    assert r2_oos(y * 0, y) == pytest.approx(0.0)
    assert r2_oos(y * 0.5, y, benchmark=y * 0.5) == pytest.approx(0.0)
    assert r2_oos(y * 0.5, y) == pytest.approx(1 - 0.25)


def test_rank_ic_and_hit_rate():
    y = grid([[0.1, -0.2, 0.05, 0.0, 0.3], [0.02, 0.01, -0.1, 0.2, -0.05]])
    np.testing.assert_allclose(rank_ic(y, y), 1.0)
    np.testing.assert_allclose(rank_ic(-y, y), -1.0)
    h = hit_rate(y.where(y != 0, 0.01), y)
    assert h["base_rate_up"] == pytest.approx(6 / 10)


def test_quantile_spread_perfect_forecast():
    rng = np.random.default_rng(1)
    y = pd.DataFrame(rng.normal(0, 0.05, (12, 50)),
                     index=pd.date_range("2020-01-31", periods=12, freq="ME"))
    q = quantile_spread(y, y, 5)
    assert (q["spread"] > 0).all()
    assert (q[[1, 2, 3, 4, 5]].diff(axis=1).iloc[:, 1:] > 0).all().all()   # monotone


# --------------------------------------------------------------------------- step 2 models


def planted_panel(signal, n_months=90, n_stocks=40, seed=0):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2012-01-31", periods=n_months, freq="ME")
    idx = pd.MultiIndex.from_product([dates, [f"S{i}" for i in range(n_stocks)]],
                                     names=["date", "ticker"])
    X = pd.DataFrame(rng.uniform(-0.5, 0.5, (len(idx), len(FEATURES))), index=idx,
                     columns=FEATURES)
    X["target"] = signal(X) + rng.normal(0, 0.02, len(idx))
    nxt = dict(zip(dates[:-1], dates[1:]))
    X["target_date"] = [nxt.get(d, pd.NaT) for d in idx.get_level_values("date")]
    return X, dates[48:-1]


def test_nonlinear_models_find_a_nonlinear_signal_a_linear_model_cannot():
    """Target depends on |x| (U-shape): no linear trend, so the elastic net finds ~0
    rank IC, while LightGBM and the MLP pick it up."""
    X, fdates = planted_panel(lambda d: 0.08 * (d["vol_1m"].abs() - 0.25))
    realized = X.loc[fdates, "target"].unstack()
    ic = {}
    for name, fn in [("enet", enet), ("lgbm", lambda: lgbm(500)), ("mlp", lambda: mlp(80))]:
        ic[name] = rank_ic(walk_forward(fn(), X, fdates, 12, 12).forecasts, realized).mean()
    assert abs(np.nan_to_num(ic["enet"])) < 0.1
    assert ic["lgbm"] > 0.3 and ic["mlp"] > 0.3


@pytest.mark.parametrize("model_fn", TUNED)
def test_runs_are_reproducible(panel, dates, model_fn):
    a = walk_forward(model_fn(), panel, dates, VAL, MIN_TRAIN, n_copies=2, seed=3)
    b = walk_forward(model_fn(), panel, dates, VAL, MIN_TRAIN, n_copies=2, seed=3)
    pd.testing.assert_frame_equal(a.forecasts, b.forecasts)
    pd.testing.assert_frame_equal(a.ensemble_var, b.ensemble_var)


def test_ensemble_variance_shape_and_sign(panel, dates):
    res = walk_forward(lgbm(), panel, dates, VAL, MIN_TRAIN, n_copies=4, seed=0)
    assert res.ensemble_var.shape == res.forecasts.shape
    assert (res.ensemble_var >= 0).all().all() and (res.ensemble_var > 0).any().any()


def test_mlp_never_uses_random_validation_split(panel, dates):
    res = walk_forward(mlp(), panel, dates, VAL, MIN_TRAIN)
    for m in res.models.values():
        assert m.model_.early_stopping is False          # sklearn's random hold-out is off
        assert 1 <= m.params_["epochs"] <= 30           # chosen on chronological validation


def test_newey_west():
    rng = np.random.default_rng(2)
    x = pd.Series(rng.normal(0, 1, 200))
    assert newey_west_se(x, lags=0) == pytest.approx(x.std(ddof=0) / np.sqrt(200))
    # Positively autocorrelated series: Newey-West SE exceeds the naive SE.
    ar = pd.Series(np.convolve(rng.normal(0, 1, 205), np.ones(6), "valid"))
    assert newey_west_se(ar) > ar.std(ddof=1) / np.sqrt(len(ar)) * 1.3
    s = summarize_series(ar + 0.5)
    assert abs(s["t_stat_nw"]) < abs(s["t_stat"])


# --------------------------------------------------------------------------- model search options


def rforest():
    return RandomForestModel(60, [4], [20], 0.5, CFG["seed"])


@pytest.mark.parametrize("window", [8, 14])
def test_rolling_window_uses_only_recent_months(panel, dates, window):
    m = HistoricalMeanModel()
    m.tuned, m.retrain_every = False, 12
    log = walk_forward(m, panel, dates, VAL, MIN_TRAIN, train_window_months=window).log
    for _, row in log.iterrows():
        months = pd.period_range(row["train_start"], row["train_end"], freq="M")
        assert len(months) <= window
        assert row["last_target_date"] <= row["retrain_date"]


def test_rolling_window_with_tuning_and_leakage(panel, dates):
    log = walk_forward(enet(), panel, dates, VAL, MIN_TRAIN, train_window_months=14).log
    assert (log["last_target_date"] <= log["retrain_date"]).all()
    assert (log["tune_end"] < log["val_start"]).all() and (log["val_end"] < log["retrain_date"]).all()


def test_demeaned_target_level_is_past_average(panel, dates):
    res = walk_forward(enet(), panel, dates, VAL, MIN_TRAIN, target_mode="demeaned")
    for _, row in res.log.iterrows():
        realized = panel[(panel["target_date"] <= row["retrain_date"]) & panel["target"].notna()]
        assert row["level"] == pytest.approx(realized["target"].mean())


@pytest.mark.parametrize("mode", ["raw", "demeaned"])
@pytest.mark.parametrize("window", [None, 14])
def test_search_options_unchanged_by_future_data(toy, dates, mode, window):
    t = dates[len(dates) // 2]
    moved = {k: (perturb_after(v, t) if k not in ("sectors", "shares_outstanding") else v)
             for k, v in toy.items()}
    a = walk_forward(enet(), toy_panel(toy), dates, VAL, MIN_TRAIN, train_window_months=window,
                     target_mode=mode).forecasts
    b = walk_forward(enet(), toy_panel(moved), dates, VAL, MIN_TRAIN, train_window_months=window,
                     target_mode=mode).forecasts
    pd.testing.assert_frame_equal(a.loc[:t], b.loc[:t])


def test_random_forest_finds_nonlinear_signal_and_has_no_leakage(panel, dates):
    X, fdates = planted_panel(lambda d: 0.08 * (d["vol_1m"].abs() - 0.25))
    realized = X.loc[fdates, "target"].unstack()
    assert rank_ic(walk_forward(rforest(), X, fdates, 12, 12).forecasts, realized).mean() > 0.3
    log = walk_forward(rforest(), panel, dates, VAL, MIN_TRAIN).log
    assert (log["last_target_date"] <= log["retrain_date"]).all()


def test_extended_features_point_in_time(toy):
    stocks = [c for c in toy["prices"].columns if c != "SPY"]
    t = toy["prices_monthly"].index[24]
    moved = {k: (perturb_after(v, t) if k not in ("sectors", "shares_outstanding") else v)
             for k, v in toy.items()}
    a = build_feature_panel(toy, stocks, "SPY", W, RSI, "extended")
    b = build_feature_panel(moved, stocks, "SPY", W, RSI, "extended")
    da, db = a.index.get_level_values("date"), b.index.get_level_values("date")
    pd.testing.assert_frame_equal(a[da <= t], b[db <= t])
    assert list(a.columns) == FEATURE_SETS["extended"]
    # Sector-relative momentum averages to zero within each sector and month.
    sec = toy["sectors"]["sector"]
    g = a["mom_sector_rel"].groupby([a.index.get_level_values("date"),
                                     sec.reindex(a.index.get_level_values("ticker")).values])
    np.testing.assert_allclose(g.mean().values, 0, atol=1e-12)


def test_grid_model_tunes_and_has_no_leakage(panel, dates, toy):
    from sklearn.neighbors import KNeighborsRegressor
    from src.models import GridModel

    def knn():
        return GridModel("KNN", lambda n_neighbors, seed: KNeighborsRegressor(n_neighbors),
                         {"n_neighbors": [3, 10]}, CFG["seed"])

    res = walk_forward(knn(), panel, dates, VAL, MIN_TRAIN)
    assert set(res.log["n_neighbors"]) <= {3, 10}
    assert (res.log["last_target_date"] <= res.log["retrain_date"]).all()
    t = dates[len(dates) // 2]
    moved = {k: (perturb_after(v, t) if k not in ("sectors", "shares_outstanding") else v)
             for k, v in toy.items()}
    b = walk_forward(knn(), toy_panel(moved), dates, VAL, MIN_TRAIN).forecasts
    pd.testing.assert_frame_equal(res.forecasts.loc[:t], b.loc[:t])


def test_classifier_model_and_shuffle(panel, dates):
    from sklearn.linear_model import LogisticRegression
    from src.experiment_models import ClassifierModel, shuffle_targets

    clf = ClassifierModel("Logistic", lambda C, seed: LogisticRegression(C=C, max_iter=500),
                          {"C": [0.1, 1.0]}, CFG["seed"])
    res = walk_forward(clf, panel, dates, VAL, MIN_TRAIN)
    assert (res.log["last_target_date"] <= res.log["retrain_date"]).all()
    assert res.forecasts.notna().all().all()
    sh = shuffle_targets(panel, 0)
    g0 = panel.groupby(level="date")["target"].apply(lambda x: sorted(x.dropna()))
    g1 = sh.groupby(level="date")["target"].apply(lambda x: sorted(x.dropna()))
    assert (g0 == g1).all()                                   # same values each month
    assert not np.allclose(panel["target"].fillna(0), sh["target"].fillna(0))


def test_ipca_recovers_planted_factor_premium():
    """Returns = (loading from signals) × factor with positive mean: IPCA forecasts rank well."""
    from src.experiment_models import IPCAModel
    rng = np.random.default_rng(3)
    dates = pd.date_range("2012-01-31", periods=90, freq="ME")
    idx = pd.MultiIndex.from_product([dates, [f"S{i}" for i in range(40)]], names=["date", "ticker"])
    X = pd.DataFrame(rng.uniform(-0.5, 0.5, (len(idx), len(FEATURES))), index=idx, columns=FEATURES)
    f = pd.Series(rng.normal(0.02, 0.03, len(dates)), index=dates)
    X["target"] = X["mom_12_1"].values * f.reindex(X.index.get_level_values("date")).values \
        + rng.normal(0, 0.02, len(idx))
    nxt = dict(zip(dates[:-1], dates[1:]))
    X["target_date"] = [nxt.get(d, pd.NaT) for d in idx.get_level_values("date")]
    fdates = dates[48:-1]
    res = walk_forward(IPCAModel({"K": [1, 2]}, CFG["seed"]), X, fdates, 12, 12)
    assert rank_ic(res.forecasts, X.loc[fdates, "target"].unstack()).mean() > 0.2
    assert (res.log["last_target_date"] <= res.log["retrain_date"]).all()


def test_ranker_recovers_planted_ranking_signal():
    from src.experiment_models import RankerModel
    X, fdates = planted_panel(lambda d: 0.04 * d["mom_12_1"])
    res = walk_forward(RankerModel({"num_leaves": [7], "n_estimators": [100]}, CFG["seed"]),
                       X, fdates, 12, 12)
    assert rank_ic(res.forecasts, X.loc[fdates, "target"].unstack()).mean() > 0.3
    assert (res.log["last_target_date"] <= res.log["retrain_date"]).all()

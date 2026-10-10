"""Phase 3: return forecasts from three ML models against simple baselines, and S5.

Run from the repo root:  python -m notebooks.phase3_forecasting

Models: elastic net, LightGBM and a small MLP, all trained with the same chronological
walk-forward procedure (src/models.py), each with ``ml.ensemble.n_copies`` bootstrap
copies whose disagreement is Ω-B for Phase 4.

Writes to ``phase3.table_dir``:
- phase3_training_log.csv      every retrain of every model: tuning / validation /
                               training months, chosen hyperparameters, validation MSE
- phase3_forecast_eval.csv     R²_oos, rank IC, hit rate, quintile spread per model
- phase3_importance.csv        out-of-sample permutation importance per model
- phase3_omega_b_check.csv     does ensemble disagreement track forecast errors?
- phase3_s5_metrics.csv        backtest of S5 for each model and γ, with S1 and S4
- phase3_s5_sharpe_tests.csv   paired Sharpe-difference tests
to ``phase3.figure_dir``: forecast_ic.png, feature_importance.png
and to ``phase3.forecast_dir`` (git-ignored): forecasts_<model>.parquet and
omega_b_<model>.parquet for Phase 4.
"""

from __future__ import annotations

import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from src.backtest import (equal_weight, forecast_strategy, markowitz_strategy,
                          rebalance_dates, run_backtest)
from src.data import load_config, load_processed
from src.features import FEATURES, add_targets, build_feature_panel, rank_normalize
from src.metrics import (holm_adjust, performance_summary, sharpe_difference_bootstrap,
                         sharpe_difference_memmel)
from src.ml_eval import (hit_rate, permutation_importance_oos, quantile_spread, r2_oos,
                         rank_ic, summarize_series)
from src.models import (ElasticNetModel, HistoricalMeanModel, LightGBMModel, MLPModel,
                        walk_forward)

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
INK, INK_2, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#a3a29c", "#e4e3df", "#fcfcfb"
ML_MODELS = ["Elastic net", "LightGBM", "MLP"]


def style_axes(ax):
    ax.set_facecolor(SURFACE)
    ax.grid(color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=INK_2, labelsize=9)


def make_models(cfg: dict) -> dict:
    ml, seed = cfg["ml"], cfg["seed"]
    en, g, m = ml["elastic_net"], ml["lightgbm"], ml["mlp"]
    every = ml["retrain_every_months"]
    return {
        "Elastic net": ElasticNetModel(en["alphas"], en["l1_ratios"], en["max_iter"], seed,
                                       retrain_every=every),
        "LightGBM": LightGBMModel(g["num_leaves"], g["min_child_samples"], g["learning_rate"],
                                  g["max_trees"], g["early_stopping_rounds"],
                                  g["feature_fraction"], g["bagging_fraction"],
                                  g["bagging_freq"], g["lambda_l2"], seed, retrain_every=every),
        "MLP": MLPModel(m["hidden_layers"], m["alphas"], m["learning_rate_init"],
                        m["batch_size"], m["max_epochs"], m["patience"], seed,
                        retrain_every=every),
    }


def sharpe_test_row(a, b, ra, rb, rf, p2, seed):
    boot = sharpe_difference_bootstrap(ra, rb, rf, block=p2["sharpe_test_block"],
                                       n_boot=p2["sharpe_test_boot"], seed=seed)
    mem = sharpe_difference_memmel(ra, rb, rf)
    row = {"a": a, "b": b, "sharpe_diff": boot["diff"], "corr": mem["rho"],
           "boot_ci_lo": boot["ci_lo"], "boot_ci_hi": boot["ci_hi"],
           "boot_p": boot["p_value"], "memmel_p": mem["p_value"]}
    for blk in p2["sharpe_test_block_robust"]:
        row[f"boot_p_block{blk}"] = sharpe_difference_bootstrap(
            ra, rb, rf, block=blk, n_boot=p2["sharpe_test_boot"], seed=seed)["p_value"]
    return row


def main() -> None:
    cfg = load_config()
    ml, bt, p2, p3 = cfg["ml"], cfg["backtest"], cfg["phase2"], cfg["phase3"]
    tab_dir, fig_dir, fc_dir = (Path(p3[k]) for k in ("table_dir", "figure_dir",
                                                      "forecast_dir"))
    for d in (tab_dir, fig_dir, fc_dir):
        d.mkdir(parents=True, exist_ok=True)

    data = load_processed(cfg)
    stocks = list(cfg["universe"])
    raw = build_feature_panel(data, stocks, cfg["data"]["benchmark"], ml["windows"],
                              ml["rsi_days"])
    panel = add_targets(rank_normalize(raw), data["returns_monthly"][stocks],
                        data["rf_monthly"]["rf"])
    dates = rebalance_dates(data["returns_monthly"].index, bt["oos_start"])
    print(f"Panel: {panel.index.get_level_values('date').nunique()} months x {len(stocks)} "
          f"stocks, {len(FEATURES)} features; forecasts at {len(dates)} dates "
          f"({dates[0]:%b %Y} – {dates[-1]:%b %Y})")

    # ---------------------------------------------------------------- forecasts
    runs = {"Historical mean": walk_forward(HistoricalMeanModel(), panel, dates,
                                            ml["val_months"], ml["min_train_months"])}
    for name, model in make_models(cfg).items():
        t0 = time.time()
        runs[name] = walk_forward(model, panel, dates, ml["val_months"],
                                  ml["min_train_months"], n_copies=ml["ensemble"]["n_copies"],
                                  seed=cfg["seed"])
        print(f"{name:12s} {len(runs[name].log)} retrains + {ml['ensemble']['n_copies']} "
              f"copies each in {time.time() - t0:.0f}s")
        slug = name.lower().replace(" ", "_")
        runs[name].forecasts.to_parquet(fc_dir / f"forecasts_{slug}.parquet")
        runs[name].ensemble_var.to_parquet(fc_dir / f"omega_b_{slug}.parquet")

    log = pd.concat([runs[n].log.assign(model=n) for n in ML_MODELS], ignore_index=True)
    log["hidden"] = log.get("hidden", pd.Series(dtype=object)).astype(str)
    log.to_csv(tab_dir / "phase3_training_log.csv", index=False)
    split_cols = ["retrain_date", "tune_start", "tune_end", "val_start", "val_end",
                  "last_target_date"]
    fmt = {c: "{:%Y-%m}".format for c in split_cols}
    print("\nChronological splits (identical for all three models; feature months):")
    print(runs["Elastic net"].log[split_cols + ["n_rows"]].to_string(index=False,
                                                                       formatters=fmt))
    params = {"Elastic net": ["alpha", "l1_ratio"],
              "LightGBM": ["num_leaves", "min_child_samples", "n_trees"],
              "MLP": ["hidden", "alpha", "epochs"]}
    for n in ML_MODELS:
        print(f"\n{n}: chosen hyperparameters by retrain")
        print(runs[n].log[["retrain_date"] + params[n] + ["val_mse"]].to_string(
            index=False, formatters={"retrain_date": "{:%Y-%m}".format}))

    # ---------------------------------------------------------------- evaluation
    realized = panel["target"].unstack().loc[dates]
    hm = runs["Historical mean"].forecasts
    en_f = runs["Elastic net"].forecasts
    candidates = [("Zero forecast", hm * 0.0), ("Historical mean", hm)] + \
        [(n, runs[n].forecasts) for n in ML_MODELS]
    rows, ics = [], {}
    for name, f in candidates:
        ic = rank_ic(f, realized)
        ics[name] = ic
        s = summarize_series(ic) if ic.notna().any() else {}
        row = {"model": name, "r2_oos_vs_zero": r2_oos(f, realized),
               "r2_oos_vs_hist_mean": r2_oos(f, realized, benchmark=hm),
               "r2_oos_vs_elastic_net": r2_oos(f, realized, benchmark=en_f),
               "ic_mean": s.get("mean"), "ic_t": s.get("t_stat"),
               "ic_t_nw": s.get("t_stat_nw"),
               "ic_share_positive": s.get("share_positive"),
               "months_identical_forecasts": int(ic.isna().sum()), **hit_rate(f, realized)}
        if name != "Zero forecast":
            q = summarize_series(quantile_spread(f, realized, ml["quantiles"])["spread"])
            row.update(q_spread_monthly=q["mean"], q_spread_t=q["t_stat"],
                       q_spread_t_nw=q["t_stat_nw"])
        if name in ML_MODELS:
            row["ic_diff_vs_hist_mean_t"] = summarize_series(
                (ic - ics["Historical mean"]).dropna())["t_stat"]
            if name != "Elastic net":
                row["ic_diff_vs_elastic_net_t"] = summarize_series(
                    (ic - ics["Elastic net"]).dropna())["t_stat"]
        rows.append(row)
    ev = pd.DataFrame(rows).set_index("model")
    ev.to_csv(tab_dir / "phase3_forecast_eval.csv")
    print("\nOut-of-sample forecast evaluation (forecasts Dec 2015 – Jul 2026):")
    print(ev.T.round(4).to_string())

    imp = pd.concat({n: permutation_importance_oos(runs[n].models, panel, dates,
                                                   ml["permutation_repeats"], cfg["seed"])
                     ["importance"] for n in ML_MODELS}, axis=1)
    imp.to_csv(tab_dir / "phase3_importance.csv")
    print("\nPermutation importance (increase in out-of-sample MSE, × 1e-6):")
    print((imp * 1e6).round(2).sort_values("LightGBM", ascending=False).to_string())

    # Ω-B check: is ensemble disagreement larger where forecasts turn out worse?
    ob = []
    for n in ML_MODELS:
        var = runs[n].ensemble_var
        err2 = (realized - runs[n].forecasts) ** 2
        v = var.stack()
        e = err2.stack().reindex(v.index)
        ok = v.notna() & e.notna()
        monthly = pd.Series([spearmanr(var.loc[t], err2.loc[t]).statistic for t in dates
                             if var.loc[t].nunique() > 1])
        ob.append({"model": n, "median_omega_b": float(v.median()),
                   "rank_corr_pooled": float(spearmanr(v[ok], e[ok]).statistic),
                   "rank_corr_monthly_mean": float(monthly.mean()),
                   "rank_corr_monthly_t": summarize_series(monthly)["t_stat"]})
    ob = pd.DataFrame(ob).set_index("model")
    ob.to_csv(tab_dir / "phase3_omega_b_check.csv")
    print("\nΩ-B check (ensemble variance vs realized squared error):")
    print(ob.round(4).to_string())

    # ---------------------------------------------------------------- S5 backtest
    wtol, rf, c = cfg["optimization"]["weight_tol"], data["rf_monthly"]["rf"], bt["cost_bps"]
    res = {"S1 Equal weight": run_backtest("S1", equal_weight(stocks), data, dates, wtol)}
    for gamma in cfg["risk_tolerance"].values():
        res[f"S4 Markowitz γ={gamma}"] = run_backtest(
            "S4", markowitz_strategy(stocks, cfg, gamma), data, dates, wtol)
        for n in ML_MODELS:
            res[f"S5 {n} γ={gamma}"] = run_backtest(
                "S5", forecast_strategy(runs[n].forecasts, stocks, cfg, gamma), data, dates,
                wtol)
    m = pd.DataFrame({k: performance_summary(r.net(c), rf, alpha=cfg["tail_risk"]["alpha"],
                                             ci_level=p2["sharpe_ci_level"],
                                             turnover=r.turnover, weights=r.weights,
                                             daily=r.daily_net(c))
                      for k, r in res.items()}).T
    lag = bt["execution_lag_robust"]
    for gamma in cfg["risk_tolerance"].values():
        for n in ML_MODELS:
            k = f"S5 {n} γ={gamma}"
            lagged = run_backtest("S5", forecast_strategy(runs[n].forecasts, stocks, cfg, gamma),
                                  data, dates, wtol, lag_days=lag)
            m.loc[k, "sharpe_lag1"] = performance_summary(lagged.net(c), rf)["sharpe"]
    m.to_csv(tab_dir / "phase3_s5_metrics.csv")
    print(f"\nBacktest at {c} bps. avg_turnover = traded each month after price drift; "
          "weight_stability = change in target weights:")
    print(m[["ann_return", "ann_vol", "sharpe", "sharpe_lag1", "max_drawdown", "max_drawdown_d",
             "cvar_m", "avg_turnover", "weight_stability"]].round(3).to_string())

    g_main = cfg["risk_tolerance"][p3["main_gamma"]]
    tests = []
    for gamma in cfg["risk_tolerance"].values():
        for n in ML_MODELS:
            a = f"S5 {n} γ={gamma}"
            for b in (f"S4 Markowitz γ={gamma}", "S1 Equal weight"):
                tests.append(sharpe_test_row(a, b, res[a].net(c), res[b].net(c), rf, p2,
                                             cfg["seed"]))
    for n in ("LightGBM", "MLP"):       # does nonlinearity help the portfolio?
        a, b = f"S5 {n} γ={g_main}", f"S5 Elastic net γ={g_main}"
        tests.append(sharpe_test_row(a, b, res[a].net(c), res[b].net(c), rf, p2, cfg["seed"]))
    tests = pd.DataFrame(tests)
    tests["boot_p_holm"] = holm_adjust(tests["boot_p"])
    tests.to_csv(tab_dir / "phase3_s5_sharpe_tests.csv", index=False)
    print("\nPaired Sharpe-difference tests:")
    print(tests.round(3).to_string(index=False))

    # ---------------------------------------------------------------- figures
    fig, ax = plt.subplots(figsize=(7.5, 4.4), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    style_axes(ax)
    names = ML_MODELS + ["Historical mean"]
    for name, color in zip(names, SERIES):
        cum = ics[name].fillna(0).cumsum()
        ax.plot(cum.index, cum.values, color=color, lw=2,
                label=f"{name} (mean IC {ics[name].mean():+.3f})")
    lo, hi = ax.get_ylim()
    last = -np.inf
    for y, n in sorted((ics[n].fillna(0).cumsum().iloc[-1], n) for n in names):
        last = max(y, last + 0.05 * (hi - lo))
        ax.text(dates[-1] + pd.Timedelta(days=40), last, n.split()[0], va="center",
                fontsize=8.5, color=INK)
    ax.axhline(0, color=MUTED, lw=1)
    ax.set_ylim(lo, hi + 0.3 * (hi - lo))
    ax.text(0.99, 0.98, "Flat stretches: years in which a model's validation chose no signal,\n"
            "so every stock got the same forecast (IC undefined, counted as 0)",
            transform=ax.transAxes, ha="right", va="top", fontsize=7.5, color=INK_2)
    ax.set_ylabel("Cumulative monthly rank IC", color=INK_2)
    ax.set_title("Out-of-sample rank correlation of forecasts with next-month returns",
                 color=INK, fontsize=11, loc="left")
    ax.legend(frameon=False, fontsize=8, loc="upper left", labelcolor=INK)
    ax.margins(x=0.08)
    fig.tight_layout()
    fig.savefig(fig_dir / "forecast_ic.png", facecolor=SURFACE)
    plt.close(fig)

    order = imp.mean(axis=1).sort_values().index
    fig, axes = plt.subplots(1, 3, figsize=(9, 4.6), dpi=200, sharey=True, sharex=True)
    fig.patch.set_facecolor(SURFACE)
    for ax, n, color in zip(axes, ML_MODELS, SERIES):
        style_axes(ax)
        ax.barh(order, imp.loc[order, n] * 1e6, color=color, height=0.6)
        ax.axvline(0, color=MUTED, lw=1)
        ax.set_title(n, color=INK, fontsize=10, loc="left")
    axes[1].set_xlabel("Increase in out-of-sample MSE when shuffled (× 10⁻⁶); "
                       "negative = the feature hurt forecasts", color=INK_2, fontsize=8.5)
    fig.suptitle("Permutation feature importance, out of sample", color=INK, fontsize=11,
                 x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(fig_dir / "feature_importance.png", facecolor=SURFACE)
    plt.close(fig)
    print(f"\nTables in {tab_dir}/, figures in {fig_dir}/, forecasts in {fc_dir}/")


if __name__ == "__main__":
    main()

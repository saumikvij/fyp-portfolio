"""Final design (D6): elastic net + LightGBM views averaged, with the ML risk model.

Run from the repo root after Phase 4:  python -m notebooks.final_design

Chosen after the experiments on branch ``experiment/model-search`` (DECISIONS D6), so
it is reported next to the pre-registered Phase 4 headline, not in place of it.

1. Risk model accuracy: next-month log volatility from trailing 63-day volatility,
   HAR and the elastic net (``src.risk_model``); mean squared error, rank correlation,
   Diebold–Mariano tests (plan O3); 2016–2026 and the two subperiods.
2. Stated risk vs realized risk: for the BL prior portfolio and the final portfolio,
   the predicted next-month volatility under the sample Σ and under Σ = D·R·D, against
   the realized volatility of the daily path.
3. Portfolios (γ = 5, τ = 0.05, Ω-A, 10 bps): S1, SPY, BL prior (sample Σ and ML Σ),
   the Phase 4 headline S6 (elastic net), EN + LightGBM averaged (sample Σ), and the
   final design (EN + LightGBM averaged, ML Σ); paired Sharpe tests, Holm-adjusted.

Writes to ``final_design.table_dir``: final_risk_accuracy.csv, final_risk_dm.csv,
final_risk_level.csv, final_design_metrics.csv, final_design_tests.csv; and the
volatility forecasts to ``phase3.forecast_dir``/vol_forecasts_elastic_net.parquet.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.backtest import (black_litterman_strategy, buy_and_hold, equal_weight, history_at,
                          rebalance_dates, run_backtest)
from src.black_litterman import bl_inputs
from src.data import load_config, load_processed
from src.estimators import sample_covariance
from src.metrics import (holm_adjust, performance_summary, sharpe_difference_bootstrap,
                         sharpe_difference_memmel)
from src.ml_eval import diebold_mariano, rank_ic, summarize_series
from src.risk_model import (HARModel, dr_covariance, load_high_low, vol_forecasts,
                            vol_panel)


def main() -> None:
    cfg = load_config()
    bt, p2, fd = cfg["backtest"], cfg["phase2"], cfg["final_design"]
    tab_dir = Path(fd["table_dir"])
    fc_dir = Path(cfg["phase3"]["forecast_dir"])
    tab_dir.mkdir(parents=True, exist_ok=True)
    data = load_processed(cfg)
    stocks = list(cfg["universe"])
    dates = rebalance_dates(data["returns_monthly"].index, bt["oos_start"])
    cut = pd.Timestamp(fd["subperiod_end"])
    parts = {"2016–2026": dates, "2016–2021": dates[dates <= cut], "2022–2026": dates[dates > cut]}
    rf, c, wtol = data["rf_monthly"]["rf"], bt["cost_bps"], cfg["optimization"]["weight_tol"]
    window, dpm = bt["cov_window_days"], cfg["data"]["trading_days_per_year"] / 12

    # ------------------------------------------------------------- 1. risk model accuracy
    high, low = load_high_low(Path(cfg["data"]["raw_dir"]) / "prices_raw.parquet",
                              data["prices"].index, stocks)
    panel = vol_panel(data, stocks, cfg["data"]["benchmark"], high, low)
    realized = panel["target"].unstack().reindex(index=dates, columns=stocks)
    vol = vol_forecasts(data, stocks, dates, cfg, panel=panel)
    vol.to_parquet(fc_dir / "vol_forecasts_elastic_net.parquet")
    risk_fc = {"Trailing 63-day volatility": panel["lv_63"].unstack().reindex(index=dates, columns=stocks),
               "HAR": vol_forecasts(data, stocks, dates, cfg, model=HARModel(), panel=panel),
               "Elastic net (ML)": vol}
    loss = {k: ((f - realized) ** 2).mean(axis=1) for k, f in risk_fc.items()}
    acc = []
    for k, f in risk_fc.items():
        for p, d in parts.items():
            acc.append({"model": k, "period": p,
                        "mse_log_vol": float(((f.loc[d] - realized.loc[d]) ** 2).stack().mean()),
                        "rank_corr": summarize_series(rank_ic(f.loc[d], realized.loc[d]))["mean"]})
    acc = pd.DataFrame(acc)
    acc.to_csv(tab_dir / "final_risk_accuracy.csv", index=False)
    dm = []
    for base in ("Trailing 63-day volatility", "HAR"):
        for p, d in parts.items():
            r = diebold_mariano(loss["Elastic net (ML)"].loc[d], loss[base].loc[d])
            dm.append({"vs": base, "period": p, **r})
    dm = pd.DataFrame(dm)
    dm.to_csv(tab_dir / "final_risk_dm.csv", index=False)
    print("Risk model accuracy (MSE of log volatility; lower is better):")
    print(acc.pivot(index="model", columns="period", values="mse_log_vol").round(4).to_string())
    print("\nDiebold–Mariano, elastic net minus baseline (negative dm = elastic net better):")
    print(dm.round(4).to_string(index=False))

    # ------------------------------------------------------------- 3. portfolios
    g = cfg["risk_tolerance"][fd["gamma"]]
    tau = cfg["black_litterman"]["tau"]
    fc = {m: pd.read_parquet(fc_dir / f"forecasts_{m}.parquet") for m in fd["view_models"]}
    avg = sum(fc.values()) / len(fc)
    en = fc["elastic_net"]
    final = "Final: EN + LightGBM averaged, ML Σ"
    prior = "BL prior only, sample Σ"
    strategies = {
        "S1 Equal weight": equal_weight(stocks),
        f"S2 {cfg['data']['benchmark']} buy-and-hold": buy_and_hold(cfg["data"]["benchmark"]),
        prior: black_litterman_strategy(en, None, stocks, cfg, g, "A", tau, views=False),
        "BL prior only, ML Σ": black_litterman_strategy(en, None, stocks, cfg, g, "A", tau,
                                                        views=False, vol_forecasts=vol),
        "S6 headline: elastic net, sample Σ": black_litterman_strategy(en, None, stocks, cfg, g,
                                                                       "A", tau),
        "EN + LightGBM averaged, sample Σ": black_litterman_strategy(avg, None, stocks, cfg, g,
                                                                     "A", tau),
        final: black_litterman_strategy(avg, None, stocks, cfg, g, fd["omega"], tau,
                                        vol_forecasts=vol),
    }
    res = {}
    for name, strat in strategies.items():
        res[name] = run_backtest(name, strat, data, dates, wtol)
        print(f"  {name}")
    split = data["returns_monthly"].index[data["returns_monthly"].index > cut][0]
    rows = {}
    for name, r in res.items():
        net = r.net(c)
        m = performance_summary(net, rf, alpha=cfg["tail_risk"]["alpha"],
                                ci_level=p2["sharpe_ci_level"], turnover=r.turnover,
                                weights=r.weights, daily=r.daily_net(c))
        m["growth_10k"] = float(10_000 * (1 + net).prod())
        for label, seg in (("2016–2021", net.loc[:split]), ("2022–2026", net.loc[split:].iloc[1:])):
            p = performance_summary(seg, rf)
            m[f"return_{label}"], m[f"sharpe_{label}"] = p["ann_return"], p["sharpe"]
        rows[name] = m
    metrics = pd.DataFrame(rows).T
    metrics.to_csv(tab_dir / "final_design_metrics.csv")
    cols = ["ann_return", "growth_10k", "sharpe", "sharpe_2016–2021", "sharpe_2022–2026",
            "ann_vol", "max_drawdown_d", "cvar_m", "avg_turnover"]
    print("\nPortfolios at 10 bps:")
    print(metrics[cols].astype(float).round(3).to_string())

    tests = []
    for a, b in [(final, prior), (final, "S6 headline: elastic net, sample Σ"),
                 ("S6 headline: elastic net, sample Σ", prior), ("BL prior only, ML Σ", prior)]:
        ra, rb = res[a].net(c), res[b].net(c)
        boot = sharpe_difference_bootstrap(ra, rb, rf, block=p2["sharpe_test_block"],
                                           n_boot=p2["sharpe_test_boot"], seed=cfg["seed"])
        tests.append({"a": a, "b": b, "sharpe_diff": boot["diff"], "ci_lo": boot["ci_lo"],
                      "ci_hi": boot["ci_hi"], "boot_p": boot["p_value"],
                      "memmel_p": sharpe_difference_memmel(ra, rb, rf)["p_value"]})
    tests = pd.DataFrame(tests)
    tests["boot_p_holm"] = holm_adjust(tests["boot_p"])
    tests.to_csv(tab_dir / "final_design_tests.csv", index=False)
    print("\nPaired Sharpe tests (a minus b):")
    print(tests.round(3).to_string(index=False))

    # ------------------------------------------------------------- 2. stated vs realized risk
    lvl = []
    me = data["returns_monthly"].index
    for name in (prior, final):
        r = res[name]
        daily = r.daily_gross
        for t in dates:
            w = r.weights.loc[t].reindex(stocks).fillna(0).values
            s = me[me > t][0]
            real = float(daily.loc[(daily.index > t) & (daily.index <= s)].std() * np.sqrt(dpm))
            S1 = sample_covariance(data["returns_daily"][stocks], t, window, dpm)
            S2 = dr_covariance(data["returns_daily"], t, stocks, vol.loc[t], window, dpm)
            for k, S in (("sample Σ", S1), ("ML Σ = D·R·D", S2)):
                lvl.append({"portfolio": name, "risk_matrix": k, "date": t,
                            "log_error": np.log(np.sqrt(w @ S.values @ w)) - np.log(real)})
    lvl = pd.DataFrame(lvl)
    lvl_sum = (lvl.groupby(["portfolio", "risk_matrix"])["log_error"]
                  .agg(bias="mean", rmse=lambda e: float(np.sqrt((e ** 2).mean()))).reset_index())
    lvl_sum.to_csv(tab_dir / "final_risk_level.csv", index=False)
    print("\nPredicted vs realized next-month portfolio volatility (log error):")
    print(lvl_sum.round(3).to_string(index=False))


if __name__ == "__main__":
    main()

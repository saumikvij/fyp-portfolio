"""EXPERIMENT (branch experiment/model-search): elastic net + tree models as Black–Litterman
views, with the ML risk matrix (DECISIONS FN1). Student's proposed final design.

Run:  python -m notebooks.en_trees_risk_test               # EN + 4 tree models (FN1)
      python -m notebooks.en_trees_risk_test --lightgbm    # EN + LightGBM only (FN2)

51 stocks, monthly, γ = 5, τ = 0.05, Ω-A, 10 bps. Views: Phase 3 elastic net and LightGBM,
plus random forest, hist gradient boosting and extra trees (search setups: expanding
window, raw target, base signals — the same fixed setup for each, not picked by results).
Either stacked (one identity block per model, each with its own Ω-A) or averaged into
one view set. Σ: sample, or D·R·D with elastic-net volatility forecasts (MS15.3).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from notebooks.bl_vol_test import vol_forecasts
from src.backtest import rebalance_dates, run_backtest
from src.data import load_config, load_processed
from src.experiment_sectors import combined_bl_strategy
from src.metrics import (cvar_historical, holm_adjust, max_drawdown, performance_summary,
                         sharpe_difference_bootstrap)

OUT = Path("report/experiments")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lightgbm", action="store_true", help="EN + LightGBM only (FN2)")
    only_lgbm = ap.parse_args().lightgbm
    tag = "en_lgbm_risk" if only_lgbm else "en_trees_risk"
    label = "EN + LightGBM" if only_lgbm else "EN + trees"
    cfg = load_config()
    ms, bt = cfg["model_search"], cfg["backtest"]
    data = load_processed(cfg)
    stocks = list(cfg["universe"])
    dates = rebalance_dates(data["returns_monthly"].index, bt["oos_start"])
    split = data["returns_monthly"].index[data["returns_monthly"].index > pd.Timestamp(ms["dev_end"])][0]
    rf = data["rf_monthly"]["rf"]
    g, tau = cfg["risk_tolerance"]["medium"], cfg["black_litterman"]["tau"]
    p3, sd = Path(cfg["phase3"]["forecast_dir"]), Path(ms["out_dir"])
    views = {"Elastic net": pd.read_parquet(p3 / "forecasts_elastic_net.parquet"),
             "LightGBM": pd.read_parquet(p3 / "forecasts_lightgbm.parquet")}
    for fam in ([] if only_lgbm else ["random_forest", "hist_gradient_boosting", "extra_trees"]):
        views[fam] = pd.read_parquet(sd / f"{fam}__expanding__raw__base.parquet")
    views = {k: v.reindex(index=dates, columns=stocks) for k, v in views.items()}
    avg = {f"{label} (average)": sum(views.values()) / len(views)}
    vol = vol_forecasts(data, stocks, dates, cfg, Path(cfg["data"]["raw_dir"]) / "prices_raw.parquet")

    def s(v, ml_sigma=False):
        return combined_bl_strategy(v, None, None, stocks, cfg, g, tau, vol if ml_sigma else None)

    strategies = {
        "No AI views, sample Σ": s({}),
        "No AI views, AI risk Σ": s({}, True),
        "Elastic net only (Phase 4)": s({"Elastic net": views["Elastic net"]}),
        f"{label} stacked": s(views),
        f"{label} stacked, AI risk Σ": s(views, True),
        f"{label} averaged": s(avg),
        f"{label} averaged, AI risk Σ": s(avg, True),
    }
    rows, nets = [], {}
    for name, strat in strategies.items():
        res = run_backtest(name, strat, data, dates, cfg["optimization"]["weight_tol"])
        net, daily = res.net(bt["cost_bps"]), res.daily_net(bt["cost_bps"])
        nets[name] = net
        row = {"strategy": name, "growth_10k": float(10_000 * (1 + net).prod()),
               "avg_turnover": float(res.turnover.iloc[1:].mean())}
        for label, seg, dseg in (("dev", net.loc[:split], daily.loc[:split]),
                                 ("hold", net.loc[split:].iloc[1:], daily.loc[daily.index > split]),
                                 ("all", net, daily)):
            p = performance_summary(seg, rf)
            row.update({f"{label}_return": p["ann_return"], f"{label}_vol": p["ann_vol"],
                        f"{label}_sharpe": p["sharpe"], f"{label}_max_dd_daily": max_drawdown(dseg),
                        f"{label}_cvar95": cvar_historical(seg)})
        rows.append(row)
        print(f"{name:34s} done", flush=True)
    table = pd.DataFrame(rows)
    table.to_csv(OUT / f"{tag}.csv", index=False)

    base = "No AI views, sample Σ"
    tests = []
    for name in [n for n in nets if n != base]:
        for label, sl in (("dev", slice(None, split)), ("hold", slice(split, None)),
                          ("all", slice(None, None))):
            a, b = nets[name].loc[sl], nets[base].loc[sl]
            if label == "hold":
                a, b = a.iloc[1:], b.iloc[1:]
            boot = sharpe_difference_bootstrap(a, b, rf, block=cfg["phase2"]["sharpe_test_block"],
                                               n_boot=cfg["phase2"]["sharpe_test_boot"],
                                               seed=cfg["seed"])
            tests.append({"strategy": name, "period": label, "sharpe_diff": boot["diff"],
                          "boot_p": boot["p_value"]})
    tests = pd.DataFrame(tests)
    tests["boot_p_holm"] = holm_adjust(tests["boot_p"])
    tests.to_csv(OUT / f"{tag}_tests.csv", index=False)
    pd.set_option("display.width", 250)
    cols = ["strategy", "dev_return", "hold_return", "all_return", "growth_10k", "dev_sharpe",
            "hold_sharpe", "all_vol", "hold_max_dd_daily", "all_cvar95", "avg_turnover"]
    print("\n" + table[cols].round(3).to_string(index=False))
    print(f"\nPaired Sharpe tests vs '{base}':")
    print(tests.round(3).to_string(index=False))


if __name__ == "__main__":
    main()

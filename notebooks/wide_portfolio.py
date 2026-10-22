"""EXPERIMENT (branch experiment/model-search): portfolios on the wide universe (WU4, MS15.4).

Run:  python -m notebooks.wide_portfolio

Universes: all wide-universe stocks with a share count (~1,047), and mid + small caps
only. Ledoit–Wolf covariance, 2% cap per stock, γ = 5, 10 bps. Strategies: 1/N, SPY,
minimum variance, Black–Litterman prior only, and Black–Litterman with views from the
wide-search setup chosen by the pre-set rule (development scores, 3-block rule).
Development and holdout parts reported separately (holdout flagged post-holdout).
"""

from __future__ import annotations

import copy
import time
from pathlib import Path

import pandas as pd

from src.backtest import (black_litterman_strategy, buy_and_hold, equal_weight,
                          min_variance_strategy, rebalance_dates, run_backtest)
from src.data import load_config
from src.experiment_universe import load_wide
from src.metrics import max_drawdown, performance_summary

OUT = Path("report/experiments")


def main() -> None:
    cfg = load_config()
    ms, bt = cfg["model_search"], cfg["backtest"]
    port = ms["wide"]["portfolio"]
    wide = load_wide()
    uni = wide["universe"]
    have = [s for s in uni.index if s in wide["shares_outstanding"].index]
    universes = {"wide (all)": have,
                 "mid + small": [s for s in have if uni.loc[s, "index"] in ("S&P 400", "S&P 600")]}

    log = pd.read_csv(OUT / "wide_search_log.csv")
    chosen = log[log["consistent"]].sort_values("ic_all", ascending=False).iloc[0]["setup"]
    views = pd.read_parquet(Path(ms["wide"]["out_dir"]) / f"{chosen}.parquet")
    print(f"Views from the wide-search setup chosen by the rule: {chosen}")

    dates = rebalance_dates(wide["returns_monthly"].index, bt["oos_start"])
    dev_end = pd.Timestamp(ms["dev_end"])
    split = wide["returns_monthly"].index[wide["returns_monthly"].index > dev_end][0]
    rf = wide["rf_monthly"]["rf"]
    g, tau = cfg["risk_tolerance"]["medium"], cfg["black_litterman"]["tau"]
    rows = {}
    for uname, stocks in universes.items():
        c = copy.deepcopy(cfg)
        c["backtest"].update(cov_method=port["cov_method"], w_max=port["w_max"])
        c["universe"] = {s: uni.loc[s, "sector"] for s in stocks}
        strategies = {"1/N": equal_weight(stocks), "SPY": buy_and_hold(cfg["data"]["benchmark"]),
                      "Min variance (LW)": min_variance_strategy(stocks, c),
                      "BL prior only (LW)": black_litterman_strategy(views, None, stocks, c, g, "A",
                                                                     tau, views=False),
                      "BL + ML views (LW)": black_litterman_strategy(views, None, stocks, c, g, "A",
                                                                     tau)}
        print(f"\n{uname}: {len(stocks)} stocks")
        for name, strat in strategies.items():
            t0 = time.time()
            res = run_backtest(name, strat, wide, dates, cfg["optimization"]["weight_tol"])
            net = res.net(bt["cost_bps"])
            row = {"universe": uname, "strategy": name,
                   "avg_turnover": float(res.turnover.iloc[1:].mean()),
                   "avg_holdings": float((res.weights > 1e-5).sum(axis=1).mean())}
            for label, seg in (("dev", net.loc[:split]), ("hold", net.loc[split:].iloc[1:])):
                p = performance_summary(seg, rf)
                row.update({f"{label}_return": p["ann_return"], f"{label}_vol": p["ann_vol"],
                            f"{label}_sharpe": p["sharpe"], f"{label}_max_dd": max_drawdown(seg)})
            rows[(uname, name)] = row
            print(f"  {name:22s} done in {time.time() - t0:4.0f}s", flush=True)
    table = pd.DataFrame(rows.values())
    table.to_csv(OUT / "wide_portfolios.csv", index=False)
    cols = ["universe", "strategy", "dev_sharpe", "hold_sharpe", "dev_vol", "hold_vol",
            "dev_max_dd", "hold_max_dd", "avg_turnover", "avg_holdings"]
    print("\n" + table[cols].round(3).to_string(index=False))


if __name__ == "__main__":
    main()

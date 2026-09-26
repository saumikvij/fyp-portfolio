"""Phase 2: out-of-sample backtest of the baselines S1–S4 and the κ(Σ) check.

Run from the repo root:  python -m notebooks.phase2_backtest

Writes to ``phase2.table_dir``:
- phase2_metrics.csv       every metric, for every strategy and cost level
- phase2_tail_ranking.csv  strategies ranked by volatility and by CVaR (10 bps)
- phase2_condition.csv     κ(Σ), λ_min, λ_max of the covariance at each rebalance
- phase2_returns.csv       monthly net returns at the main cost level
and to ``phase2.figure_dir``:
- backtest_wealth.png      cumulative wealth of S1, S2, S3 and the headline S4
- condition_number.png     κ(Σ) over time (empirical side of T5)
"""

from __future__ import annotations

import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.backtest import (buy_and_hold, equal_weight, markowitz_strategy,
                          min_variance_strategy, rebalance_dates, run_backtest)
from src.data import load_config, load_processed
from src.estimators import condition_number, sample_covariance
from src.metrics import performance_summary

# Reference palette (dataviz skill), light mode, categorical slots in fixed order.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
INK, INK_2, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#a3a29c", "#e4e3df", "#fcfcfb"


def style_axes(ax):
    ax.set_facecolor(SURFACE)
    ax.grid(color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=INK_2, labelsize=9)


def main() -> None:
    cfg = load_config()
    p2, bt = cfg["phase2"], cfg["backtest"]
    fig_dir, tab_dir = Path(p2["figure_dir"]), Path(p2["table_dir"])
    fig_dir.mkdir(parents=True, exist_ok=True)
    tab_dir.mkdir(parents=True, exist_ok=True)

    data = load_processed(cfg)
    stocks = list(cfg["universe"])
    bench = cfg["data"]["benchmark"]
    dates = rebalance_dates(data["returns_monthly"].index, bt["oos_start"])
    rf = data["rf_monthly"]["rf"]
    wtol = cfg["optimization"]["weight_tol"]

    strategies = {
        "S1 Equal weight": equal_weight(stocks),
        f"S2 {bench} buy-and-hold": buy_and_hold(bench),
        "S3 Minimum variance": min_variance_strategy(stocks, cfg),
    }
    for level, gamma in cfg["risk_tolerance"].items():
        strategies[f"S4 Markowitz γ={gamma} ({level})"] = markowitz_strategy(stocks, cfg, gamma)
    main_s4 = f"S4 Markowitz γ={cfg['risk_tolerance'][p2['main_gamma']]} ({p2['main_gamma']})"

    results = {}
    for name, strat in strategies.items():
        t0 = time.time()
        results[name] = run_backtest(name, strat, data, dates, wtol)
        print(f"{name:32s} {len(dates)} rebalances in {time.time() - t0:.1f}s")

    # ---------------------------------------------------------------- metrics
    rows = []
    for cost in bt["cost_bps_grid"]:
        for name, res in results.items():
            m = performance_summary(res.net(cost), rf, alpha=cfg["tail_risk"]["alpha"],
                                    ci_level=p2["sharpe_ci_level"], turnover=res.turnover,
                                    weights=res.weights, daily=res.daily_net(cost))
            rows.append({"strategy": name, "cost_bps": cost, **m})
    metrics = pd.DataFrame(rows)
    metrics.to_csv(tab_dir / "phase2_metrics.csv", index=False)

    main = metrics[metrics.cost_bps == bt["cost_bps"]].set_index("strategy")
    rank = pd.DataFrame({"ann_vol": main["ann_vol"], "cvar_m": main["cvar_m"],
                         "rank_by_vol": main["ann_vol"].rank().astype(int),
                         "rank_by_cvar": main["cvar_m"].rank().astype(int)}).sort_values("ann_vol")
    rank.to_csv(tab_dir / "phase2_tail_ranking.csv")
    pd.DataFrame({n: r.net(bt["cost_bps"]) for n, r in results.items()}).to_csv(
        tab_dir / "phase2_returns.csv")

    period = f"{results[main_s4].gross.index[0]:%b %Y} – {results[main_s4].gross.index[-1]:%b %Y}"
    print(f"\nOut-of-sample {period}, {len(dates)} months, costs {bt['cost_bps']} bps")
    show = main[["ann_return", "ann_vol", "sharpe", "sharpe_lo", "sharpe_hi", "max_drawdown",
                 "var_m", "cvar_m", "avg_turnover", "weight_stability"]]
    print(show.round(3).to_string())
    print("\nRanking by volatility vs by monthly CVaR (1 = least risky):")
    print(rank.round(4).to_string())

    # ---------------------------------------------------------------- κ(Σ)
    cond = []
    for t in dates:
        S = sample_covariance(data["returns_daily"][stocks], t, bt["cov_window_days"],
                              cfg["data"]["trading_days_per_year"] / 12)
        eig = np.linalg.eigvalsh(S.values)
        cond.append({"t": t, "kappa": condition_number(S), "lambda_min": eig[0],
                     "lambda_max": eig[-1]})
    cond = pd.DataFrame(cond).set_index("t")
    cond.to_csv(tab_dir / "phase2_condition.csv")
    print(f"\nκ(Σ): median {cond.kappa.median():.0f}, range {cond.kappa.min():.0f}–"
          f"{cond.kappa.max():.0f} (max at {cond.kappa.idxmax():%b %Y})")

    # ---------------------------------------------------------------- figures
    plotted = ["S1 Equal weight", f"S2 {bench} buy-and-hold", "S3 Minimum variance", main_s4]
    fig, ax = plt.subplots(figsize=(7.5, 4.8), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    style_axes(ax)
    ends = []
    for name, color in zip(plotted, SERIES):
        r = results[name].net(bt["cost_bps"])
        start = r.index[0] - pd.offsets.MonthEnd(1)
        wealth = pd.concat([pd.Series([1.0], index=[start]), (1 + r).cumprod()])
        ax.plot(wealth.index, wealth.values, color=color, lw=2, label=name)
        ends.append((wealth.iloc[-1], name.split()[0], color))
    # Direct labels at the right edge, pushed apart so they never overlap.
    ends.sort()
    lo, hi = ax.get_ylim()
    gap, last_y = 0.045 * (hi - lo), -np.inf
    x_lab = wealth.index[-1] + pd.Timedelta(days=45)
    for y, label, _ in ends:
        last_y = max(y, last_y + gap)
        ax.text(x_lab, last_y, label, va="center", fontsize=8.5, color=INK)
    ax.set_ylabel("Growth of $1 (after 10 bps costs)", color=INK_2)
    ax.set_title(f"Out-of-sample backtest, {period}", color=INK, fontsize=11, loc="left")
    ax.legend(frameon=False, fontsize=8.5, loc="upper left", labelcolor=INK)
    ax.margins(x=0.06)
    fig.tight_layout()
    fig.savefig(fig_dir / "backtest_wealth.png", facecolor=SURFACE)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.5, 3.8), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    style_axes(ax)
    ax.plot(cond.index, cond.kappa, color=SERIES[0], lw=2)
    peak = cond.kappa.idxmax()
    ax.scatter([peak], [cond.kappa.max()], s=64, color=SERIES[0], edgecolor=SURFACE,
               linewidth=2, zorder=5)
    ax.annotate(f"max {cond.kappa.max():.0f} ({peak:%b %Y})", (peak, cond.kappa.max()),
                xytext=(8, -2), textcoords="offset points", fontsize=8.5, color=INK)
    ax.set_ylabel("κ(Σ) = λ_max / λ_min", color=INK_2)
    ax.set_ylim(0, None)
    ax.set_title(f"Condition number of the {len(stocks)}-stock covariance at each rebalance",
                 color=INK, fontsize=11, loc="left")
    fig.tight_layout()
    fig.savefig(fig_dir / "condition_number.png", facecolor=SURFACE)
    plt.close(fig)
    print(f"\nTables in {tab_dir}/, figures in {fig_dir}/")


if __name__ == "__main__":
    main()

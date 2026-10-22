"""EXPERIMENT (branch experiment/model-search): 2008–2009 crisis stress test (DECISIONS MS10).

Run after ``python -m src.experiment_crisis``:  python -m notebooks.crisis_test

The three finalist setups and the original Phase 4 elastic net are retrained with the
same walk-forward procedure on 2001–2007 data only, then forecast month by month from
Dec 2007 to Nov 2009 (holding Jan 2008 – Dec 2009). Portfolios are compared with 1/N,
SPY, Markowitz and the prior-only Black–Litterman portfolio.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from notebooks.model_search import build_model, build_wide
from src.backtest import (black_litterman_strategy, buy_and_hold, equal_weight,
                          markowitz_strategy, rebalance_dates, run_backtest)
from src.data import load_config
from src.experiment_crisis import CRISIS, load_crisis
from src.experiment_signals import build_niche_panel, rank_normalize_niche
from src.features import FEATURES, add_targets, build_feature_panel, rank_normalize
from src.metrics import cvar_historical, max_drawdown, performance_summary
from src.ml_eval import rank_ic, summarize_series
from src.models import walk_forward

OUT = Path("report/experiments")
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
INK, INK_2, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#a3a29c", "#e4e3df", "#fcfcfb"

# Finalists from the development search (DECISIONS MS11/MS12).
FINALISTS = [("Kernel ridge", 48, "raw"), ("KNN", 48, "demeaned"), ("LightGBM", 48, "demeaned")]


def main() -> None:
    cfg = load_config()
    ml, bt = cfg["ml"], cfg["backtest"]
    data = load_crisis(cfg)
    stocks = data["stocks"]
    bench = cfg["data"]["benchmark"]
    cfg = {**cfg, "universe": {s: cfg["universe"][s] for s in stocks}}
    dates = rebalance_dates(data["returns_monthly"].index, "2008-01-01")
    print(f"{len(stocks)} stocks; decisions {dates[0]:%b %Y} – {dates[-1]:%b %Y} ({len(dates)})")

    base = add_targets(rank_normalize(build_feature_panel(data, stocks, bench, ml["windows"],
                                                          ml["rsi_days"], "base")),
                       data["returns_monthly"][stocks], data["rf_monthly"]["rf"])
    niche = add_targets(rank_normalize_niche(build_niche_panel(
        data, stocks, bench, ml["windows"], ml["rsi_days"], CRISIS / "raw" / "prices_raw.parquet")),
        data["returns_monthly"][stocks], data["rf_monthly"]["rf"])
    risk_set = cfg["model_search"]["round4"]["risk_set"]

    fc = {"Original elastic net (Phase 4)": walk_forward(
        build_model("Elastic net", cfg, FEATURES), base, dates, ml["val_months"],
        ml["min_train_months"]).forecasts}
    for fam, win, tgt in FINALISTS:
        fc[f"{fam} (search finalist)"] = walk_forward(
            build_wide(fam, cfg, risk_set, 12), niche, dates, ml["val_months"],
            ml["min_train_months"], train_window_months=win, target_mode=tgt).forecasts

    realized = base["target"].unstack().loc[dates]
    print("\nForecast quality during the crisis (rank IC, NW t):")
    for name, f in fc.items():
        s = summarize_series(rank_ic(f, realized))
        print(f"  {name:34s} IC {s['mean']:+.3f}  (t {s['t_stat_nw']:+.2f})")

    g, tau = cfg["risk_tolerance"]["medium"], cfg["black_litterman"]["tau"]
    first = next(iter(fc.values()))
    strategies = {"1/N": equal_weight(stocks), "SPY": buy_and_hold(bench),
                  "Markowitz (S4)": markowitz_strategy(stocks, cfg, g),
                  "BL prior only": black_litterman_strategy(first, None, stocks, cfg, g, "A", tau,
                                                            views=False)}
    for name, f in fc.items():
        strategies[f"BL + {name}"] = black_litterman_strategy(f, None, stocks, cfg, g, "A", tau)

    rf, c, wt = data["rf_monthly"]["rf"], bt["cost_bps"], cfg["optimization"]["weight_tol"]
    rows, wealth = {}, {}
    for name, strat in strategies.items():
        res = run_backtest(name, strat, data, dates, wt)
        net, daily = res.net(c), res.daily_net(c)
        rows[name] = {"total_return": float((1 + net).prod() - 1),
                      "ann_vol": float(net.std() * np.sqrt(12)),
                      "worst_month": float(net.min()),
                      "max_drawdown_daily": max_drawdown(daily),
                      "cvar95_daily": cvar_historical(daily, 0.95),
                      "sharpe": performance_summary(net, rf)["sharpe"],
                      "avg_turnover": float(res.turnover.iloc[1:].mean())}
        wealth[name] = (1 + net).cumprod()
    table = pd.DataFrame(rows).T.sort_values("max_drawdown_daily")
    table.to_csv(OUT / "crisis_results.csv")
    print(f"\nPortfolios, Jan 2008 – Dec 2009 (γ = {g}, {c} bps):")
    print(table.round(3).to_string())

    plotted = ["SPY", "1/N", "BL prior only", "BL + KNN (search finalist)"]
    fig, ax = plt.subplots(figsize=(7.5, 4.4), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    ax.grid(color=GRID, linewidth=0.8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=INK_2, labelsize=9)
    for name, color in zip(plotted, SERIES):
        w = pd.concat([pd.Series([1.0], index=[dates[0]]), wealth[name]])
        ax.plot(w.index, w.values * 10, color=color, lw=2, label=name)
    ax.set_ylabel("Value of $10,000 ($ thousands)", color=INK_2)
    ax.set_title("2008–2009 crisis stress test (50 stocks; survivors only)", color=INK,
                 fontsize=10.5, loc="left")
    ax.legend(frameon=False, fontsize=8.5, loc="lower left", labelcolor=INK)
    fig.tight_layout()
    fig.savefig(OUT / "crisis_wealth.png", facecolor=SURFACE)
    plt.close(fig)


if __name__ == "__main__":
    main()

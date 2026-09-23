"""Phase 1 figures: efficient frontiers on the real data at one date.

Run from the repo root:  python -m notebooks.phase1_frontier

Writes to ``frontier_plot.figure_dir`` (config.yaml):
- frontier.png           σ–μ plane: closed-form frontier (short sales allowed), the
                         long-only frontier with the 10% cap, GMV, tangency + capital
                         market line, the three risk-tolerance portfolios, and stocks.
- frontier_parabola.png  (σ², μ) plane: closed-form parabola (T2) with solver points.
- phase1_portfolios.csv  ex-ante stats of the portfolios in the figures.
Axes are annualized: mean × 12, volatility × √12, variance × 12.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.data import load_config, load_processed
from src.estimators import historical_mean, sample_covariance
from src.optimization import (Constraints, efficient_frontier, frontier_constants,
                              frontier_return_range, frontier_variance, gmv_closed_form,
                              mean_variance, min_variance, portfolio_stats,
                              tangency_closed_form)

# Reference palette (dataviz skill), light mode, first three categorical slots.
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
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
    fp, bt = cfg["frontier_plot"], cfg["backtest"]
    out_dir = Path(fp["figure_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    solver = cfg["optimization"]["solver"]

    d = load_processed(cfg)
    stocks = list(cfg["universe"])
    t = pd.Timestamp(fp["as_of"])
    Sigma = sample_covariance(d["returns_daily"][stocks], t, bt["cov_window_days"],
                              cfg["data"]["trading_days_per_year"] / 12)
    mu = historical_mean(d["returns_monthly"][stocks], t,
                         cfg["estimators"]["mean_window_months"])
    # Risk-free for the month after t, from the yield observed at t.
    rf = (1 + d["rf_daily"]["annual"].loc[:t].iloc[-1]) ** (1 / 12) - 1
    cons = Constraints(long_only=True, w_max=bt["w_max"])

    consts = frontier_constants(mu, Sigma)
    w_gmv = gmv_closed_form(Sigma)
    w_tan = tangency_closed_form(mu, Sigma, rf)
    gmv, tan = portfolio_stats(w_gmv, mu, Sigma), portfolio_stats(w_tan, mu, Sigma)

    # Closed-form frontier (both branches) and the constrained frontier.
    m_lo, m_hi = frontier_return_range(mu, cons, solver)
    m_cf = np.linspace(2 * gmv["mean"] - 1.3 * m_hi, 1.3 * m_hi, 400)
    var_cf = frontier_variance(m_cf, consts)
    ef = efficient_frontier(mu, Sigma, np.linspace(m_lo, m_hi, fp["n_points"]), cons, solver)
    lo_gmv = portfolio_stats(min_variance(Sigma, cons, solver).weights, mu, Sigma)

    rows = {"GMV (unconstrained)": (w_gmv, gmv), "Tangency (unconstrained)": (w_tan, tan),
            "GMV (long-only, cap)": (None, lo_gmv)}
    for level, gamma in cfg["risk_tolerance"].items():
        w = mean_variance(mu, Sigma, gamma, cons, solver).weights
        rows[f"Risk {level} (γ={gamma})"] = (w, portfolio_stats(w, mu, Sigma))

    ann_m = lambda x: 12 * np.asarray(x) * 100          # noqa: E731  percent per year
    ann_v = lambda v: np.sqrt(12 * np.asarray(v)) * 100  # noqa: E731

    # ---------------------------------------------------------------- σ–μ figure
    fig, ax = plt.subplots(figsize=(7.5, 6.4), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    style_axes(ax)
    stock_vol, stock_m = ann_v(np.diag(Sigma.values)), ann_m(mu)
    # Zoom on the frontier region; stocks outside it are counted in the note.
    x_max = fp["x_max_vol_pa"]
    y_max = max(ann_m(ef["mean"]).max(), ann_m(tan["mean"])) + fp["y_headroom_pa"]
    inside = (stock_vol <= x_max) & (stock_m <= y_max)
    ax.scatter(stock_vol[inside], stock_m[inside], s=14, color=MUTED, zorder=2,
               label="Individual stocks")
    ax.plot(ann_v(var_cf), ann_m(m_cf), color=BLUE, lw=2, zorder=3,
            label="Frontier, short sales allowed (closed form, T2)")
    ax.plot(ann_v(ef["variance"]), ann_m(ef["mean"]), color=ORANGE, lw=2, zorder=4,
            label=f"Frontier, long-only, cap {bt['w_max']:.0%} (solver)")
    sharpe = (tan["mean"] - rf) / tan["vol"]
    x_cml = np.linspace(0, (y_max - ann_m(rf)) / (sharpe * np.sqrt(12)), 50)  # stop at y_max
    ax.plot(x_cml, ann_m(rf) + sharpe * np.sqrt(12) * x_cml, color=AQUA, lw=2, ls="--",
            zorder=3, label="Capital market line (T3)")

    def mark(stats, marker, color, label=None):
        ax.scatter([ann_v(stats["variance"])], [ann_m(stats["mean"])], s=64, marker=marker,
                   color=color, edgecolor=SURFACE, linewidth=2, zorder=6, label=label)

    mark(gmv, "o", BLUE, "GMV, short sales allowed")
    mark(lo_gmv, "o", ORANGE, "GMV, long-only")
    mark(tan, "D", AQUA, "Tangency (max Sharpe)")
    first = True
    for name, (_, s) in rows.items():
        if name.startswith("Risk"):
            mark(s, "s", ORANGE, "Risk-tolerance portfolios (label = γ)" if first else None)
            first = False
            gamma = cfg["risk_tolerance"][name.split()[1]]
            ax.annotate(f"{gamma}", (ann_v(s["variance"]), ann_m(s["mean"])),
                        xytext=(0, 8), textcoords="offset points", ha="center",
                        fontsize=8.5, color=INK, zorder=7)

    ax.set_xlim(0, x_max)
    ax.set_ylim(0, y_max)
    ax.set_xlabel("Volatility (% per year)", color=INK_2)
    ax.set_ylabel("Expected return (% per year)", color=INK_2)
    ax.set_title(f"Efficient frontiers, {len(stocks)} stocks, estimated at {t.date()}",
                 color=INK, fontsize=11, loc="left")
    ax.legend(frameon=False, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.12),
              ncol=2, labelcolor=INK)
    if (~inside).sum():
        ax.text(x_max, y_max, f"{(~inside).sum()} stocks lie outside the plotted range ",
                ha="right", va="top", fontsize=8, color=INK_2)
    fig.tight_layout()
    fig.savefig(out_dir / "frontier.png", facecolor=SURFACE)
    plt.close(fig)

    # ---------------------------------------------------------- (σ², μ) figure
    unc = Constraints(long_only=False, w_max=None)
    m_pts = np.linspace(m_cf.min(), m_cf.max(), 15)
    ef_unc = efficient_frontier(mu, Sigma, m_pts, unc, solver)
    fig, ax = plt.subplots(figsize=(7.5, 4.6), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    style_axes(ax)
    ax.plot(12 * var_cf, ann_m(m_cf), color=BLUE, lw=2,
            label="Closed form σ² = (Am² − 2Bm + C)/D")
    ax.scatter(12 * ef_unc["variance"], ann_m(ef_unc["mean"]), s=64, color=ORANGE,
               edgecolor=SURFACE, linewidth=2, zorder=4, label="Solver (cvxpy) points")
    ax.scatter([12 * gmv["variance"]], [ann_m(gmv["mean"])], s=64, color=BLUE,
               edgecolor=SURFACE, linewidth=2, zorder=5)
    ax.annotate("GMV: variance 1/A at mean B/A", (12 * gmv["variance"],
                ann_m(gmv["mean"])), xytext=(8, -4), textcoords="offset points",
                fontsize=8.5, color=INK)
    ax.set_xlabel("Variance σ² (per year)", color=INK_2)
    ax.set_ylabel("Expected return (% per year)", color=INK_2)
    ax.set_title("T2: the frontier is a parabola in (σ², μ) space", color=INK,
                 fontsize=11, loc="left")
    ax.legend(frameon=False, fontsize=8.5, loc="center right", labelcolor=INK)
    fig.tight_layout()
    fig.savefig(out_dir / "frontier_parabola.png", facecolor=SURFACE)
    plt.close(fig)

    # ---------------------------------------------------------------- table
    table = pd.DataFrame({
        name: {"exp_return_pa": 12 * s["mean"], "vol_pa": np.sqrt(12) * s["vol"],
               "sharpe_pa": np.sqrt(12) * (s["mean"] - rf) / s["vol"],
               "n_holdings": (w.abs() > cfg["optimization"]["weight_tol"]).sum()
               if w is not None else np.nan,
               "max_weight": w.max() if w is not None else np.nan,
               "min_weight": w.min() if w is not None else np.nan}
        for name, (w, s) in rows.items()}).T
    table.to_csv(out_dir / "phase1_portfolios.csv")
    print(f"As of {t.date()}, rf = {12 * rf:.2%} per year")
    print(table.round(3).to_string())
    print(f"Figures written to {out_dir}/")


if __name__ == "__main__":
    main()

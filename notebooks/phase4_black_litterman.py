"""Phase 4: Black–Litterman with ML views (S6), compared with S1, S2, S4 and S5.

Run from the repo root after Phase 3:  python -m notebooks.phase4_black_litterman

Uses the Phase 3 forecasts and Ω-B variances in ``phase3.forecast_dir``.
The headline S6 configuration (model, Ω, γ, τ) is fixed in config.yaml before any
result was computed; everything else is an ablation.

Writes to ``phase4.table_dir``:
- phase4_metrics.csv          every strategy and ablation (10 bps, plus Sharpe at 0/25 bps)
- phase4_diagnostics.csv      how far μ_BL moves from the prior toward the views
- phase4_sharpe_tests.csv     paired Sharpe-difference tests, Holm-adjusted
- phase4_turnover_tests.csv   paired tests of the monthly turnover difference
and to ``phase4.figure_dir``: bl_wealth.png, bl_turnover.png, bl_tau.png
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.backtest import (black_litterman_strategy, buy_and_hold, equal_weight,
                          forecast_strategy, history_at, markowitz_strategy,
                          rebalance_dates, run_backtest)
from src.black_litterman import bl_inputs
from src.data import load_config, load_processed
from src.metrics import (holm_adjust, performance_summary, sharpe_difference_bootstrap,
                         sharpe_difference_memmel)
from src.ml_eval import summarize_series

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
INK, INK_2, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#a3a29c", "#e4e3df", "#fcfcfb"
MODELS = ["Elastic net", "LightGBM", "MLP"]
OMEGAS = ["A", "B", "B-cal"]


def style_axes(ax):
    ax.set_facecolor(SURFACE)
    ax.grid(color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=INK_2, labelsize=9)


def slug(name: str) -> str:
    return name.lower().replace(" ", "_")


def main() -> None:
    cfg = load_config()
    bt, bl, p2, p4 = cfg["backtest"], cfg["black_litterman"], cfg["phase2"], cfg["phase4"]
    tab_dir, fig_dir = Path(p4["table_dir"]), Path(p4["figure_dir"])
    fc_dir = Path(cfg["phase3"]["forecast_dir"])
    tab_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    data = load_processed(cfg)
    stocks = list(cfg["universe"])
    dates = rebalance_dates(data["returns_monthly"].index, bt["oos_start"])
    rf, c, wtol = data["rf_monthly"]["rf"], bt["cost_bps"], cfg["optimization"]["weight_tol"]
    fc = {m: pd.read_parquet(fc_dir / f"forecasts_{slug(m)}.parquet") for m in MODELS}
    ob = {m: pd.read_parquet(fc_dir / f"omega_b_{slug(m)}.parquet") for m in MODELS}

    head = bl["headline"]
    g_head = cfg["risk_tolerance"][head["gamma"]]
    tau = bl["tau"]
    head_name = f"S6 {head['model']} Ω-{head['omega']} γ={g_head} τ={tau}"
    print(f"Headline (fixed in advance): {head_name}")

    strategies: dict[str, object] = {
        "S1 Equal weight": equal_weight(stocks),
        f"S2 {cfg['data']['benchmark']} buy-and-hold": buy_and_hold(cfg["data"]["benchmark"]),
        f"BL prior only γ={g_head}": black_litterman_strategy(
            fc[head["model"]], None, stocks, cfg, g_head, "A", tau, views=False),
    }
    for gamma in cfg["risk_tolerance"].values():
        strategies[f"S4 Markowitz γ={gamma}"] = markowitz_strategy(stocks, cfg, gamma)
    for m in MODELS:
        strategies[f"S5 {m} γ={g_head}"] = forecast_strategy(fc[m], stocks, cfg, g_head)
        for om in OMEGAS:
            strategies[f"S6 {m} Ω-{om} γ={g_head} τ={tau}"] = black_litterman_strategy(
                fc[m], ob[m], stocks, cfg, g_head, om, tau)
    # Ablations around the headline: γ, τ, Ω scale.
    hm, ho = head["model"], head["omega"]
    for gamma in cfg["risk_tolerance"].values():
        if gamma != g_head:
            strategies[f"S5 {hm} γ={gamma}"] = forecast_strategy(fc[hm], stocks, cfg, gamma)
            strategies[f"S6 {hm} Ω-{ho} γ={gamma} τ={tau}"] = black_litterman_strategy(
                fc[hm], ob[hm], stocks, cfg, gamma, ho, tau)
    for t_ in bl["tau_grid"]:
        if t_ != tau:
            strategies[f"S6 {hm} Ω-{ho} γ={g_head} τ={t_}"] = black_litterman_strategy(
                fc[hm], ob[hm], stocks, cfg, g_head, ho, t_)
    for k in bl["omega_scale_grid"]:
        if k != 1.0:
            strategies[f"S6 {hm} Ω-{ho}×{k} γ={g_head} τ={tau}"] = black_litterman_strategy(
                fc[hm], ob[hm], stocks, cfg, g_head, ho, tau, omega_scale=k)

    res = {}
    for i, (name, strat) in enumerate(strategies.items(), 1):
        res[name] = run_backtest(name, strat, data, dates, wtol)
        print(f"  [{i:2d}/{len(strategies)}] {name}")

    # ---------------------------------------------------------------- metrics
    rows = {}
    for name, r in res.items():
        m = performance_summary(r.net(c), rf, alpha=cfg["tail_risk"]["alpha"],
                                ci_level=p2["sharpe_ci_level"], turnover=r.turnover,
                                weights=r.weights, daily=r.daily_net(c))
        for cost in bt["cost_bps_grid"]:
            if cost != c:
                m[f"sharpe_{cost}bps"] = performance_summary(r.net(cost), rf)["sharpe"]
        rows[name] = m
    metrics = pd.DataFrame(rows).T
    metrics.to_csv(tab_dir / "phase4_metrics.csv")
    cols = ["ann_return", "ann_vol", "sharpe", "sharpe_0bps", "sharpe_25bps", "max_drawdown_d",
            "cvar_m", "avg_turnover", "weight_stability"]
    print(f"\nResults at {c} bps (turnover = traded after drift; weight_stability = change "
          "in target weights; drawdown on the daily path):")
    print(metrics[cols].round(3).to_string())

    # ---------------------------------------------------------------- diagnostics
    diag = []
    for m in MODELS:
        for om in OMEGAS:
            pull, pull_cs, ratio = [], [], []
            for t in dates:
                inp = bl_inputs(history_at(data, t), fc[m], ob[m], stocks, cfg, om, tau)
                d_view, d_post = inp["Q"] - inp["Pi"], inp["mu_bl"] - inp["Pi"]
                if np.linalg.norm(d_view) > 0:
                    pull.append(np.linalg.norm(d_post) / np.linalg.norm(d_view))
                # Cross-sectional part only: a shift common to all stocks does not change
                # mean-variance weights under 1ᵀw = 1 (D1.2), so this is what matters.
                cs_view, cs_post = d_view - d_view.mean(), d_post - d_post.mean()
                if np.linalg.norm(cs_view) > 1e-12:
                    pull_cs.append(np.linalg.norm(cs_post) / np.linalg.norm(cs_view))
                ratio.append(float((inp["Omega"] / (tau * np.diag(inp["Sigma"].values))).median()))
            diag.append({"model": m, "omega": om, "view_pull_mean": float(np.mean(pull)),
                         "view_pull_cross_section": float(np.mean(pull_cs)),
                         "omega_over_tau_sigma_median": float(np.median(ratio))})
    diag = pd.DataFrame(diag)
    diag.to_csv(tab_dir / "phase4_diagnostics.csv", index=False)
    print("\nHow far μ_BL moves from the prior toward the views (0 = prior, 1 = views), in "
          "total and across stocks only, and the typical ratio Ω / (τ σ²):")
    print(diag.round(3).to_string(index=False))

    # ---------------------------------------------------------------- tests
    pairs = []
    for m in MODELS:
        for om in OMEGAS:
            s6 = f"S6 {m} Ω-{om} γ={g_head} τ={tau}"
            pairs += [(s6, f"S5 {m} γ={g_head}"), (s6, f"S4 Markowitz γ={g_head}"),
                      (s6, "S1 Equal weight")]
    pairs.append((head_name, f"BL prior only γ={g_head}"))
    tests = []
    for a, b in pairs:
        ra, rb = res[a].net(c), res[b].net(c)
        boot = sharpe_difference_bootstrap(ra, rb, rf, block=p2["sharpe_test_block"],
                                           n_boot=p2["sharpe_test_boot"], seed=cfg["seed"])
        row = {"a": a, "b": b, "sharpe_diff": boot["diff"], "boot_ci_lo": boot["ci_lo"],
               "boot_ci_hi": boot["ci_hi"], "boot_p": boot["p_value"],
               "memmel_p": sharpe_difference_memmel(ra, rb, rf)["p_value"]}
        for blk in p2["sharpe_test_block_robust"]:
            row[f"boot_p_block{blk}"] = sharpe_difference_bootstrap(
                ra, rb, rf, block=blk, n_boot=p2["sharpe_test_boot"], seed=cfg["seed"])["p_value"]
        tests.append(row)
    tests = pd.DataFrame(tests)
    tests["boot_p_holm"] = holm_adjust(tests["boot_p"])
    tests.to_csv(tab_dir / "phase4_sharpe_tests.csv", index=False)
    print("\nPaired Sharpe-difference tests (10 bps), Holm-adjusted over the family:")
    print(tests[["a", "b", "sharpe_diff", "boot_ci_lo", "boot_ci_hi", "boot_p", "memmel_p",
                 "boot_p_holm"]].round(3).to_string(index=False))

    to_tests = []
    for m in MODELS:
        for om in OMEGAS:
            a, b = f"S6 {m} Ω-{om} γ={g_head} τ={tau}", f"S5 {m} γ={g_head}"
            d = (res[a].turnover - res[b].turnover).iloc[1:]      # skip the first build
            s = summarize_series(d)
            to_tests.append({"a": a, "b": b, "turnover_a": res[a].turnover.iloc[1:].mean(),
                             "turnover_b": res[b].turnover.iloc[1:].mean(),
                             "diff": s["mean"], "t_nw": s["t_stat_nw"]})
    to_tests = pd.DataFrame(to_tests)
    to_tests.to_csv(tab_dir / "phase4_turnover_tests.csv", index=False)
    print("\nMonthly turnover, S6 vs S5 (paired difference, Newey–West t):")
    print(to_tests.round(3).to_string(index=False))

    # ---------------------------------------------------------------- figures
    plotted = ["S1 Equal weight", f"S4 Markowitz γ={g_head}", f"S5 {hm} γ={g_head}", head_name]
    labels = ["S1 1/N", f"S4 Markowitz", f"S5 {hm} (raw forecasts)",
              f"S6 BL ({hm}, Ω-{ho})"]
    fig, ax = plt.subplots(figsize=(7.5, 4.6), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    style_axes(ax)
    ends = []
    for name, lab, color in zip(plotted, labels, SERIES):
        r = res[name].net(c)
        wealth = pd.concat([pd.Series([1.0], index=[r.index[0] - pd.offsets.MonthEnd(1)]),
                            (1 + r).cumprod()])
        ax.plot(wealth.index, wealth.values, color=color, lw=2, label=lab)
        ends.append((wealth.iloc[-1], lab.split()[0]))
    lo, hi = ax.get_ylim()
    last = -np.inf
    for y, lab in sorted(ends):
        last = max(y, last + 0.05 * (hi - lo))
        ax.text(wealth.index[-1] + pd.Timedelta(days=45), last, lab, va="center",
                fontsize=8.5, color=INK)
    ax.set_ylabel(f"Growth of $1 (after {c} bps costs)", color=INK_2)
    ax.set_title(f"Black–Litterman vs raw forecasts, γ = {g_head}, "
                 f"{res[head_name].gross.index[0]:%b %Y} – {res[head_name].gross.index[-1]:%b %Y}",
                 color=INK, fontsize=11, loc="left")
    ax.legend(frameon=False, fontsize=8.5, loc="upper left", labelcolor=INK)
    ax.margins(x=0.07)
    fig.tight_layout()
    fig.savefig(fig_dir / "bl_wealth.png", facecolor=SURFACE)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.5, 4.2), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    style_axes(ax)
    groups = ["S5 raw"] + [f"S6 Ω-{o}" for o in OMEGAS]
    x = np.arange(len(MODELS))
    width = 0.2
    for j, (grp, color) in enumerate(zip(groups, SERIES)):
        vals = [metrics.loc[f"S5 {m} γ={g_head}" if grp == "S5 raw"
                            else f"S6 {m} {grp[3:]} γ={g_head} τ={tau}", "avg_turnover"] * 100
                for m in MODELS]
        ax.bar(x + (j - 1.5) * width, vals, width * 0.92, color=color, label=grp)
    for ref, style in [(f"S4 Markowitz γ={g_head}", "--"), (f"BL prior only γ={g_head}", ":")]:
        ax.axhline(metrics.loc[ref, "avg_turnover"] * 100, color=INK_2, lw=1.2, ls=style)
        ax.text(len(MODELS) - 0.45, metrics.loc[ref, "avg_turnover"] * 100,
                " S4" if ref.startswith("S4") else " prior only", va="center", fontsize=8,
                color=INK_2)
    ax.set_xticks(x, MODELS)
    ax.set_ylabel("Average monthly turnover (%)", color=INK_2)
    ax.set_title(f"Turnover: raw forecasts (S5) vs Black–Litterman (S6), γ = {g_head}",
                 color=INK, fontsize=11, loc="left")
    ax.legend(frameon=False, fontsize=8.5, loc="upper left", labelcolor=INK, ncol=4)
    ax.set_ylim(0, None)
    lo, hi = ax.get_ylim()
    ax.set_ylim(0, hi * 1.12)
    fig.tight_layout()
    fig.savefig(fig_dir / "bl_turnover.png", facecolor=SURFACE)
    plt.close(fig)

    taus = sorted(bl["tau_grid"])
    tau_rows = metrics.loc[[f"S6 {hm} Ω-{ho} γ={g_head} τ={t_}" for t_ in taus]]
    fig, axes = plt.subplots(1, 2, figsize=(7.5, 3.4), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    for ax, col, lab, scale in [(axes[0], "sharpe", "Sharpe ratio (10 bps)", 1),
                                (axes[1], "avg_turnover", "Monthly turnover (%)", 100)]:
        style_axes(ax)
        ax.plot(taus, tau_rows[col].values * scale, color=SERIES[0], lw=2, marker="o", ms=5)
        ax.axvline(tau, color=MUTED, lw=1, ls=":")
        ax.set_xscale("log")
        ax.set_xticks(taus, [str(t_) for t_ in taus])
        ax.set_xlabel("τ", color=INK_2)
        ax.set_title(lab, color=INK, fontsize=10, loc="left")
    fig.suptitle(f"τ sensitivity of S6 ({hm}, Ω-{ho}, γ = {g_head}); dotted line = headline τ",
                 color=INK, fontsize=10.5, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(fig_dir / "bl_tau.png", facecolor=SURFACE)
    plt.close(fig)
    print(f"\nTables in {tab_dir}/, figures in {fig_dir}/")


if __name__ == "__main__":
    main()

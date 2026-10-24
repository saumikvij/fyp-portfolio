"""Phase 5: one complete worked example, from input to recommendation (advisor, A3/A4).

Run from the repo root after Phase 3:  python -m notebooks.phase5_walkthrough

Takes the example named in config ``recommend.walkthrough`` ($10,000, medium risk) and
records every step the tool takes, with the numbers at each step:
1. input → γ; 2. data as of t; 3. ML forecasts (views Q); 4. Black–Litterman prior Π,
view uncertainty Ω and posterior μ_BL (T6); 5. the constrained optimization, which
limits bind, and the KKT check (T4); 6. whole shares; 7. constraint checks, and the
check that the tool's weights equal the backtested S6 strategy's weights at t.

Writes ``report/phase5_walkthrough.md`` and the tables behind it to ``phase5.table_dir``
(phase5_walkthrough_views.csv, phase5_walkthrough_holdings.csv,
phase5_walkthrough_sectors.csv).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.backtest import black_litterman_strategy, history_at, strategy_constraints
from src.black_litterman import bl_inputs
from src.data import load_config, load_processed
from src.optimization import kkt_residuals, mean_variance
from src.recommend import (UserRequest, forecasts_up_to, recommend, resolve_as_of,
                           sector_bounds)


def md_table(df: pd.DataFrame, fmt: dict[str, str]) -> str:
    """A pipe table; ``fmt`` maps column -> format spec (others printed as is)."""
    cols = list(df.columns)
    lines = ["| " + " | ".join([df.index.name or ""] + cols) + " |",
             "|" + "---|" * (len(cols) + 1)]
    for idx, row in df.iterrows():
        cells = [str(idx)] + [format(row[c], fmt[c]) if c in fmt else str(row[c]) for c in cols]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main() -> None:
    cfg = load_config()
    p5, bl = cfg["phase5"], cfg["black_litterman"]
    tab_dir = Path(p5["table_dir"])
    tab_dir.mkdir(parents=True, exist_ok=True)
    ex = next(e for e in cfg["recommend"]["examples"] if e["name"] == cfg["recommend"]["walkthrough"])
    req = UserRequest(ex["amount"], ex["risk"])

    data = load_processed(cfg)
    stocks = list(cfg["universe"])
    sectors = data["sectors"]["sector"]
    t = resolve_as_of(data, cfg["recommend"]["as_of"])
    gamma = cfg["risk_tolerance"][req.risk]
    w_max, s_max = cfg["backtest"]["w_max"], cfg["optimization"]["sector_max"]

    # Steps 3–4: forecasts and Black–Litterman inputs, exactly as the tool computes them.
    fc, ev = forecasts_up_to(data, cfg, t)
    hist = history_at(data, t)
    inp = bl_inputs(hist, fc, ev, stocks, cfg, bl["headline"]["omega"], bl["tau"])
    Pi, Q, Om, mu = inp["Pi"], inp["Q"], inp["Omega"], inp["mu_bl"]
    d_view, d_post = Q - Pi, mu - Pi
    pull = float(np.linalg.norm(d_post) / np.linalg.norm(d_view))
    views = pd.DataFrame({"sector": sectors.reindex(stocks), "w_mkt": inp["w_mkt"], "Pi": Pi,
                          "Q": Q, "omega_sd": np.sqrt(Om), "mu_bl": mu}).rename_axis("ticker")
    views.to_csv(tab_dir / "phase5_walkthrough_views.csv")

    # Step 5: the optimization and its KKT certificate.
    cons = strategy_constraints(hist, cfg)
    res = mean_variance(mu, inp["Sigma"], gamma, cons, cfg["optimization"]["solver"])
    kkt = kkt_residuals(res, mu, inp["Sigma"], gamma, cons)
    tol = cfg["optimization"]["weight_tol"]

    # Steps 6–7: the tool itself.
    rec = recommend(req, cfg, data, as_of=t, forecasts=(fc, ev))
    again = recommend(req, cfg, data, as_of=t, forecasts=(fc, ev))
    reproducible = rec.holdings.equals(again.holdings) and rec.summary == again.summary
    strat = black_litterman_strategy(fc, ev, stocks, cfg, gamma, bl["headline"]["omega"], bl["tau"])
    w_bt = strat(hist).reindex(stocks).fillna(0.0)
    w_bt = w_bt.where(w_bt.abs() >= tol, 0.0)
    w_bt = w_bt / w_bt.sum()
    gap_bt = float((rec.target.reindex(stocks) - w_bt).abs().max())

    at_cap = rec.target[rec.target >= w_max - tol].index.tolist()
    sec_t = rec.sector_weights["target"]
    binding = [s for s, r in zip(res.sector_names, res.rho_up) if r > 1e-8]
    hold = rec.holdings.rename_axis("ticker")
    hold.to_csv(tab_dir / "phase5_walkthrough_holdings.csv")
    rec.sector_weights.rename_axis("sector").to_csv(tab_dir / "phase5_walkthrough_sectors.csv")
    s = rec.summary

    pct = lambda x: f"{100 * x:.2f}%"                                     # noqa: E731
    v = views.copy()
    for c in ("w_mkt", "Pi", "Q", "omega_sd", "mu_bl"):
        v[c] = v[c] * 100
    top = v.sort_values("mu_bl", ascending=False)
    show = pd.concat([top.head(5), top.tail(5)])
    fmt_v = {c: ".2f" for c in ("w_mkt", "Pi", "Q", "omega_sd", "mu_bl")}
    h = hold[["sector", "price", "shares", "value", "weight", "target"]].copy()
    h["weight"], h["target"] = h["weight"] * 100, h["target"] * 100
    sec = rec.sector_weights.copy() * 100
    sec["held_value"] = rec.holdings.groupby("sector")["value"].sum().reindex(sec.index).fillna(0)
    q_unique = int(Q.round(10).nunique())
    ck = lambda d: ", ".join(f"{k} {'✓' if v_ else '✗'}" for k, v_ in d.items())   # noqa: E731

    md = f"""# Worked example: from input to recommendation

Generated by `python -m notebooks.phase5_walkthrough` from the data and `config.yaml`;
re-running it gives the same numbers. Example: **{ex['name']}**, as of **{t.date()}**.

## 1. Input

The user gives two things: **${req.amount:,.0f}** and risk tolerance **{req.risk}**. The
risk tolerance sets the risk-aversion parameter **γ = {gamma}** (low / medium / high = 10 / 5 / 2).
Everything else is fixed by the project, not chosen by the user (advisor feedback A4):
the 51-stock universe, long-only, at most **{w_max:.0%} per stock**, and the sector rule,
**no sector above {s_max:.0%}**, the same constraints the backtest uses.

## 2. Data as of {t.date()}

Only data dated on or before {t.date()} is used (tested by scrambling everything after it).
Σ is the sample covariance of the last {cfg['backtest']['cov_window_days']} trading days
of daily returns, scaled to a month. The risk-free rate is {pct(s['risk_free_pa'])} a year.

## 3. ML forecasts (the views Q)

The Phase 4 headline model, the elastic net, is retrained with the Phase 3 walk-forward
on data up to {t.date()} and forecasts each stock's next-month excess return. The
forecasts range from {Q.min() * 100:.2f}% to {Q.max() * 100:.2f}% a month
({q_unique} distinct values across 51 stocks).

## 4. Black–Litterman: prior, view uncertainty, posterior

- **Prior Π = δΣw_mkt** (δ = {bl['delta']}): the returns that make today's market-cap weights optimal.
- **View uncertainty Ω-A**: each stock's mean squared past forecast error (last
  {bl['omega_a_window']} months), shrunk toward its return variance. Large Ω = little trust.
- **Posterior μ_BL** (T6), τ = {bl['tau']}: μ_BL moved **{pull:.0%}** of the way from the
  prior toward the views (‖μ_BL − Π‖ / ‖Q − Π‖), so the weak forecasts change little.

Monthly %, the 5 highest and 5 lowest μ_BL:

{md_table(show[["sector", "w_mkt", "Pi", "Q", "omega_sd", "mu_bl"]], fmt_v)}

(w_mkt = market-cap weight; omega_sd = √Ω, the view's standard error.)

## 5. Optimization

maximize μ_BLᵀw − (γ/2)wᵀΣw subject to 1ᵀw = 1, 0 ≤ wᵢ ≤ {w_max:.0%}, every sector ≤ {s_max:.0%}.

- Solver status **{res.status}**; **{int((rec.target > 0).sum())} stocks** in the target portfolio.
- At the {w_max:.0%} stock cap: {', '.join(at_cap) or 'none'}.
- Sectors at the {s_max:.0%} cap: {', '.join(binding) or 'none'} (largest sector:
  {sec_t.idxmax()}, {sec_t.max():.1%}).
- KKT check (T4), largest violation: stationarity {kkt['stationarity']:.1e}, primal
  {kkt['primal']:.1e}, dual {kkt['dual']:.1e}, complementarity {kkt['complementarity']:.1e},
  so the solution is the certified optimum.

## 6. Whole shares

Target weight × ${req.amount:,.0f} ÷ the closing price on {t.date()}, rounded down; the
leftover cash then buys single shares only where that brings a holding closer to its
target, never breaking a cap. Result: **${s['invested']:,.2f} invested in {s['n_holdings']}
stocks, ${s['cash']:,.2f} cash**; the largest gap between a held and a target weight is
{s['max_abs_weight_gap']:.2%}.

{md_table(h, {"price": ",.2f", "shares": ".0f", "value": ",.2f", "weight": ".1f", "target": ".1f"})}

(weight and target in %; weight = value ÷ ${req.amount:,.0f}, so cash is unallocated.)

## 7. Checks

- Constraints, target weights: {ck(rec.checks_target)}.
- Constraints, whole shares: {ck(rec.checks_holdings)}.
- **Same as the backtest:** the tool's target weights equal the weights the backtested
  S6 strategy holds from {t.date()} (largest difference {gap_bt:.1e}).
- Reproducible: a second run gives identical holdings and summary ({'✓' if reproducible else '✗'}).
- Expected volatility {s['held_vol_pa']:.1%} a year (target portfolio {s['target_vol_pa']:.1%});
  model expected excess return {s['target_exp_excess_return_pa']:.1%} a year, a model
  estimate, not a promise.

Sector weights (%), target and held, against the {s_max:.0%} rule:

{md_table(sec[["target", "held", "max"]], {"target": ".1f", "held": ".1f", "max": ".0f"})}
"""
    out = Path("report/phase5_walkthrough.md")
    out.write_text(md)
    print(md)
    assert all(rec.checks_target.values()) and all(rec.checks_holdings.values())
    assert gap_bt < 1e-9 and reproducible
    print(f"Written to {out}")


if __name__ == "__main__":
    main()

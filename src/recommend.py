"""Portfolio recommendation tool (plan Sections 1, 2.6; Phase 5).

A user gives an investment amount and a risk tolerance (low / medium / high); the
sector rule is chosen by us, not the user (A4: no sector above 25%, set in config).
The tool returns a simulated portfolio of whole shares, using the headline
Black–Litterman model fixed in Phase 4 (D4.1), with exactly the constraints of the
backtested strategy (``src.backtest.strategy_constraints``):

1. Forecasts: the headline ML model is run with the same chronological walk-forward
   procedure as Phase 3, up to and including the recommendation date t.
2. μ_BL at t: market prior Π = δΣw_mkt combined with the forecasts (T6), with Ω from
   past forecast errors (Ω-A).
3. Target weights: max μ_BLᵀw − (γ/2)wᵀΣw subject to 1ᵀw = 1, 0 ≤ wᵢ ≤ w_max and every
   sector ≤ sector_max, with γ set by the risk tolerance (10 / 5 / 2). These are the
   backtest's S6 weights at t (tested).
4. Whole shares: weights × amount ÷ the close price at t, rounded down; leftover cash
   then buys single shares where the target is most under-filled, never breaking a cap.

Every input is read from data dated <= t, so a recommendation "as of" a past date is
exactly what the tool would have said then (tested). No trades are placed.

Run:  python -m src.recommend --amount 10000 --risk medium
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from src.backtest import History, history_at, strategy_constraints
from src.black_litterman import bl_inputs
from src.data import load_config, load_processed
from src.features import add_targets, build_feature_panel, rank_normalize
from src.models import ElasticNetModel, LightGBMModel, MLPModel, walk_forward
from src.optimization import mean_variance, rule_sector_bounds


@dataclass
class UserRequest:
    """What the user asks for.

    Attributes
    ----------
    amount : investment amount in dollars (> 0).
    risk : "low", "medium" or "high" (maps to γ via config ``risk_tolerance``).
    """

    amount: float
    risk: str


def sector_bounds(sectors: pd.Series, cfg: dict) -> dict[str, tuple[float, float]]:
    """The sector rule from config as {sector: (0, sector_max)} (A4; the user sets none).

    Parameters
    ----------
    sectors : ticker -> sector.
    cfg : config (``optimization.sector_max``).

    Returns
    -------
    The same bounds the backtest uses (``src.optimization.rule_sector_bounds``).

    Theorem: none (constraint set of the main problem, plan 2.2).
    """
    return rule_sector_bounds(sectors, cfg["optimization"].get("sector_max"))


def validate_request(req: UserRequest, sectors: pd.Series, cfg: dict) -> None:
    """Reject requests that are malformed, and a configuration that cannot be met.

    Checks: amount > 0; a known risk level; and that the per-stock cap and the sector
    rule together leave room for 100% of the portfolio (else no portfolio exists).

    Parameters
    ----------
    req : the user request.
    sectors : ticker -> sector.
    cfg : config (risk levels, w_max, sector_max).

    Raises
    ------
    ValueError explaining what is wrong.

    Theorem: none (feasibility of the constraint set; T1 needs it non-empty).
    """
    if not req.amount > 0:
        raise ValueError("The investment amount must be positive.")
    if req.risk not in cfg["risk_tolerance"]:
        raise ValueError(f"Risk tolerance must be one of {list(cfg['risk_tolerance'])}.")
    w_max = cfg["backtest"]["w_max"]
    bounds = sector_bounds(sectors, cfg)
    capacity = sum(min(bounds.get(s, (0.0, 1.0))[1], int((sectors == s).sum()) * w_max)
                   for s in set(sectors))
    if capacity < 1 - 1e-12:
        raise ValueError(f"The caps leave room for only {capacity:.0%} of the portfolio "
                         f"({w_max:.0%} per stock, {cfg['optimization']['sector_max']:.0%} "
                         "per sector).")


def resolve_as_of(data: dict, as_of: str | pd.Timestamp | None) -> pd.Timestamp:
    """Last month-end in the data on or before ``as_of`` (default: the latest)."""
    me = data["returns_monthly"].index
    if as_of is None:
        return me[-1]
    eligible = me[me <= pd.Timestamp(as_of)]
    if eligible.empty:
        raise ValueError(f"No month-end on or before {as_of}.")
    return eligible[-1]


def _headline_model(cfg: dict):
    ml, seed = cfg["ml"], cfg["seed"]
    every = ml["retrain_every_months"]
    name = cfg["black_litterman"]["headline"]["model"]
    if name == "Elastic net":
        en = ml["elastic_net"]
        return ElasticNetModel(en["alphas"], en["l1_ratios"], en["max_iter"], seed,
                               retrain_every=every)
    if name == "LightGBM":
        g = ml["lightgbm"]
        return LightGBMModel(g["num_leaves"], g["min_child_samples"], g["learning_rate"],
                             g["max_trees"], g["early_stopping_rounds"], g["feature_fraction"],
                             g["bagging_fraction"], g["bagging_freq"], g["lambda_l2"], seed,
                             retrain_every=every)
    if name == "MLP":
        m = ml["mlp"]
        return MLPModel(m["hidden_layers"], m["alphas"], m["learning_rate_init"],
                        m["batch_size"], m["max_epochs"], m["patience"], seed,
                        retrain_every=every)
    raise ValueError(f"Unknown headline model {name!r}")


def forecasts_up_to(data: dict, cfg: dict, t: pd.Timestamp):
    """Headline-model forecasts at every month-end from the first out-of-sample decision
    date to t, by the Phase 3 walk-forward (same retrain dates, so earlier forecasts
    equal the Phase 3 ones).

    Parameters
    ----------
    data : processed data (only rows <= t are used: features and targets are cut at t).
    cfg : config.
    t : recommendation date (a month-end).

    Returns
    -------
    (forecasts, ensemble_var); ensemble_var is None unless the headline Ω is Ω-B.

    Theorem: none (plan 4.3).
    """
    ml = cfg["ml"]
    stocks = list(cfg["universe"])
    cut = {k: (v.loc[:t] if k not in ("sectors", "shares_outstanding") else v)
           for k, v in data.items()}
    raw = build_feature_panel(cut, stocks, cfg["data"]["benchmark"], ml["windows"],
                              ml["rsi_days"])
    panel = add_targets(rank_normalize(raw), cut["returns_monthly"][stocks],
                        cut["rf_monthly"]["rf"])
    me = cut["returns_monthly"].index
    first = me[me < pd.Timestamp(cfg["backtest"]["oos_start"])][-1]
    dates = me[(me >= first) & (me <= t)]
    copies = 0 if cfg["black_litterman"]["headline"]["omega"] == "A" \
        else ml["ensemble"]["n_copies"]
    res = walk_forward(_headline_model(cfg), panel, dates, ml["val_months"],
                       ml["min_train_months"], n_copies=copies, seed=cfg["seed"])
    return res.forecasts, res.ensemble_var


def whole_shares(target: pd.Series, prices: pd.Series, amount: float, w_max: float,
                 sectors: pd.Series, bounds: dict[str, tuple[float, float]]) -> pd.DataFrame:
    """Convert target weights to whole shares without breaking any cap.

    Step 1: shares = ⌊wᵢ·amount / priceᵢ⌋. Step 2: with the leftover cash, repeatedly
    buy the one share that most reduces the distance to the target, Σ|valueᵢ − wᵢ·amount|,
    provided it is affordable, the stock stays ≤ w_max of the amount and its sector
    stays ≤ its maximum. A share is only bought if it brings that stock *closer* to its
    target (gap > price / 2), so the holdings never drift away from the optimizer's
    portfolio just to spend cash. Weights are measured against the full amount (cash
    counts as unallocated).

    Parameters
    ----------
    target : target weights (sum 1) by ticker; must already satisfy the cap and the
        sector bounds (the optimizer guarantees this). Rounding down cannot then
        break a cap, and the top-up checks every cap before buying.
    prices : close prices at t by ticker.
    amount : investment amount.
    w_max : per-stock cap.
    sectors : ticker -> sector.
    bounds : sector bounds {sector: (lower, upper)}; only the upper bounds are used
        (the sector rule has no minimums).

    Returns
    -------
    DataFrame by ticker (only stocks with shares > 0, largest first) with sector,
    price, shares, value, weight (value / amount) and target weight.

    Theorem: none.
    """
    tickers = target.index
    px = prices.reindex(tickers)
    shares = np.floor(target * amount / px).astype(int)
    hi = {s: b[1] for s, b in bounds.items()}

    def value():
        return shares * px

    cash = amount - float(value().sum())
    sec_of = sectors.reindex(tickers)

    def allowed(v: pd.Series) -> pd.Series:
        """Stocks where one more share is affordable and breaks no stock or sector cap."""
        sector_val = v.groupby(sec_of).sum()
        ok = (px <= cash + 1e-9) & ((v + px) <= w_max * amount + 1e-9) & (target > 0)
        for i in tickers[ok.values]:
            s = sec_of[i]
            if s in hi and sector_val.get(s, 0.0) + px[i] > hi[s] * amount + 1e-9:
                ok[i] = False
        return ok

    # Step 2: spend what is left only where it moves a holding closer to its target.
    while True:
        v = value()
        gap = target * amount - v
        improvement = gap.abs() - (gap - px).abs()          # reduction in |gap| if bought
        ok = allowed(v) & (improvement > 0)
        if not ok.any():
            break
        pick = improvement[ok].idxmax()
        shares[pick] += 1
        cash -= float(px[pick])
    out = pd.DataFrame({"sector": sectors.reindex(tickers), "price": px, "shares": shares,
                        "value": value(), "weight": value() / amount, "target": target})
    return out[out["shares"] > 0].sort_values("value", ascending=False)


def check_constraints(weights: pd.Series, sectors: pd.Series, w_max: float,
                      bounds: dict[str, tuple[float, float]], tol: float,
                      full_investment: bool = True) -> dict[str, bool]:
    """Which constraints a set of weights satisfies (each True / False).

    Parameters
    ----------
    weights : weights by ticker (target weights, or realized holdings / amount).
    sectors : ticker -> sector.
    w_max : per-stock cap.
    bounds : sector bounds.
    tol : numerical tolerance.
    full_investment : check 1ᵀw = 1 (True for targets; holdings leave some cash).

    Returns
    -------
    dict: long_only, cap, sector_max, (budget).

    Theorem: none (feasible set of the main problem, plan 2.2).
    """
    by_sector = weights.groupby(sectors.reindex(weights.index)).sum()
    out = {
        "long_only": bool((weights >= -tol).all()),
        "cap": bool((weights <= w_max + tol).all()),
        "sector_max": all(by_sector.get(s, 0.0) <= hi + tol for s, (_, hi) in bounds.items()),
    }
    if full_investment:
        out["budget"] = bool(abs(weights.sum() - 1) <= tol)
    return out


@dataclass
class Recommendation:
    """Result of ``recommend``; ``summary`` holds the headline numbers."""

    as_of: pd.Timestamp
    request: UserRequest
    gamma: float
    holdings: pd.DataFrame          # whole-share portfolio, by ticker
    target: pd.Series               # optimizer weights (sum 1), by ticker
    sector_weights: pd.DataFrame    # target vs held, by sector
    summary: dict
    checks_target: dict
    checks_holdings: dict


def recommend(req: UserRequest, cfg: dict, data: dict | None = None,
              as_of: str | pd.Timestamp | None = None,
              forecasts: tuple | None = None) -> Recommendation:
    """Produce the recommended portfolio for one user request.

    Parameters
    ----------
    req : amount and risk tolerance.
    cfg : config.
    data : processed data (loaded if None).
    as_of : recommendation date; None = latest month-end (config ``recommend.as_of``).
    forecasts : precomputed ``forecasts_up_to`` output for the same date (optional;
        lets several requests share one model run).

    Returns
    -------
    Recommendation with target weights, whole-share holdings, sector breakdown,
    ex-ante statistics and constraint checks.

    Raises
    ------
    ValueError if the request is invalid or infeasible (see ``validate_request``).

    Theorem: T6 (μ_BL), T1 (unique target portfolio), T4 (the constrained QP).
    """
    data = load_processed(cfg) if data is None else data
    stocks = list(cfg["universe"])
    sectors = data["sectors"]["sector"]
    validate_request(req, sectors, cfg)
    t = resolve_as_of(data, as_of if as_of is not None else cfg["recommend"]["as_of"])
    bl, head = cfg["black_litterman"], cfg["black_litterman"]["headline"]
    gamma = cfg["risk_tolerance"][req.risk]
    w_max = cfg["backtest"]["w_max"]
    bounds = sector_bounds(sectors, cfg)

    fc, ev = forecasts if forecasts is not None else forecasts_up_to(data, cfg, t)
    hist: History = history_at(data, t)
    inp = bl_inputs(hist, fc, ev, stocks, cfg, head["omega"], bl["tau"])
    cons = strategy_constraints(hist, cfg)          # identical to the backtest's S6
    res = mean_variance(inp["mu_bl"], inp["Sigma"], gamma, cons, cfg["optimization"]["solver"])
    tol = cfg["optimization"]["weight_tol"]
    target = res.weights.where(res.weights.abs() >= tol, 0.0)
    target = target / target.sum()

    prices = data["close"].loc[t, stocks]
    hold = whole_shares(target, prices, req.amount, w_max, sectors, bounds)
    held_w = hold["weight"].reindex(stocks).fillna(0.0)

    S = inp["Sigma"].values
    mu = inp["mu_bl"].values
    rf_annual = float(data["rf_daily"]["annual"].loc[:t].iloc[-1])

    def stats(w: pd.Series) -> dict:
        var = float(w.values @ S @ w.values)
        return {"exp_excess_return_pa": 12 * float(mu @ w.values),
                "vol_pa": float(np.sqrt(12 * var))}

    st_t, st_h = stats(target), stats(held_w)
    summary = {
        "as_of": str(t.date()), "amount": req.amount, "risk": req.risk, "gamma": gamma,
        "n_holdings": int(len(hold)), "invested": float(hold["value"].sum()),
        "cash": float(req.amount - hold["value"].sum()),
        "target_vol_pa": st_t["vol_pa"], "held_vol_pa": st_h["vol_pa"],
        "target_exp_excess_return_pa": st_t["exp_excess_return_pa"],
        "risk_free_pa": rf_annual,
        "max_abs_weight_gap": float((held_w - target).abs().max()),
        "w_max": w_max,
    }
    sec = pd.DataFrame({"target": target.groupby(sectors).sum(),
                        "held": held_w.groupby(sectors).sum()})
    for s, (_, hi) in bounds.items():
        sec.loc[s, "max"] = hi
    return Recommendation(
        as_of=t, request=req, gamma=gamma, holdings=hold, target=target,
        sector_weights=sec.sort_values("target", ascending=False), summary=summary,
        checks_target=check_constraints(target, sectors, w_max, bounds, tol),
        checks_holdings=check_constraints(held_w, sectors, w_max, bounds, tol,
                                          full_investment=False))


def format_recommendation(rec: Recommendation) -> str:
    """Plain-text report of a recommendation for the command line."""
    s = rec.summary
    req = rec.request
    caps = rec.sector_weights["max"].dropna() if "max" in rec.sector_weights else pd.Series()
    rule = f"no sector above {caps.max():.0%}" if len(caps) else "no sector rule"
    lines = [
        f"Recommended portfolio as of {s['as_of']} (simulated; no trades are placed)",
        f"Amount ${req.amount:,.0f}  ·  risk tolerance {req.risk} (γ = {rec.gamma})  ·  "
        f"{rule}, at most {s['w_max']:.0%} per stock",
        "",
        f"{'Ticker':7s} {'Sector':24s} {'Price':>9s} {'Shares':>7s} {'Value':>11s} "
        f"{'Weight':>7s} {'Target':>7s}",
    ]
    for tk, r in rec.holdings.iterrows():
        lines.append(f"{tk:7s} {r['sector'][:24]:24s} {r['price']:9.2f} {int(r['shares']):7d} "
                     f"{r['value']:11,.2f} {r['weight']:7.1%} {r['target']:7.1%}")
    lines += [
        "",
        f"Invested ${s['invested']:,.2f} in {s['n_holdings']} stocks; cash ${s['cash']:,.2f}",
        f"Expected volatility {s['held_vol_pa']:.1%} a year (target portfolio "
        f"{s['target_vol_pa']:.1%}). Model expected excess return "
        f"{s['target_exp_excess_return_pa']:.1%} a year: a model estimate, not a promise.",
        "Constraints, target weights:  " + ", ".join(f"{k} {'✓' if v else '✗'}"
                                                    for k, v in rec.checks_target.items()),
        "Constraints, whole shares:    " + ", ".join(f"{k} {'✓' if v else '✗'}"
                                                    for k, v in rec.checks_holdings.items()),
    ]
    return "\n".join(lines)


def main() -> None:
    p = argparse.ArgumentParser(description="Recommend a simulated portfolio.")
    p.add_argument("--amount", type=float, required=True, help="investment amount in $")
    p.add_argument("--risk", required=True, choices=["low", "medium", "high"])
    p.add_argument("--as-of", default=None, help="recommendation date (default: latest)")
    p.add_argument("--json", default=None, help="also save the result to this JSON file")
    p.add_argument("--config", default="config.yaml")
    a = p.parse_args()
    cfg = load_config(a.config)
    req = UserRequest(a.amount, a.risk)
    try:
        rec = recommend(req, cfg, as_of=a.as_of)
    except ValueError as e:
        p.exit(2, f"Cannot build this portfolio: {e}\n")
    print(format_recommendation(rec))
    if a.json:
        with open(a.json, "w") as f:
            json.dump({"request": asdict(req), "summary": rec.summary,
                       "holdings": rec.holdings.reset_index(names="ticker").to_dict("records"),
                       "checks_target": rec.checks_target,
                       "checks_holdings": rec.checks_holdings}, f, indent=2, default=str)


if __name__ == "__main__":
    main()

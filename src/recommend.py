"""Portfolio recommendation tool (plan Sections 1, 2.6; Phase 5).

A user gives an investment amount, a risk tolerance (low / medium / high) and optional
sector preferences. The tool returns a simulated portfolio of whole shares, using the
headline Black–Litterman model fixed in Phase 4 (D4.1):

1. Forecasts: the headline ML model is run with the same chronological walk-forward
   procedure as Phase 3, up to and including the recommendation date t.
2. μ_BL at t: market prior Π = δΣw_mkt combined with the forecasts (T6), with Ω from
   past forecast errors (Ω-A).
3. Target weights: max μ_BLᵀw − (γ/2)wᵀΣw subject to 1ᵀw = 1, 0 ≤ wᵢ ≤ w_max and the
   user's sector bounds, with γ set by the risk tolerance (10 / 5 / 2).
4. Whole shares: weights × amount ÷ the close price at t, rounded down; leftover cash
   then buys single shares where the target is most under-filled, never breaking a cap.

Every input is read from data dated <= t, so a recommendation "as of" a past date is
exactly what the tool would have said then (tested). No trades are placed.

Run:  python -m src.recommend --amount 10000 --risk medium --exclude Energy \\
          --max-sector "Information Technology=0.2"
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd

from src.backtest import History, history_at
from src.black_litterman import bl_inputs
from src.data import load_config, load_processed
from src.features import add_targets, build_feature_panel, rank_normalize
from src.models import ElasticNetModel, LightGBMModel, MLPModel, walk_forward
from src.optimization import Constraints, mean_variance


@dataclass
class UserRequest:
    """What the user asks for.

    Attributes
    ----------
    amount : investment amount in dollars (> 0).
    risk : "low", "medium" or "high" (maps to γ via config ``risk_tolerance``).
    exclude : sectors to hold nothing in.
    max_sector : {sector: maximum total weight}.
    min_sector : {sector: minimum total weight}.
    """

    amount: float
    risk: str
    exclude: list[str] = field(default_factory=list)
    max_sector: dict[str, float] = field(default_factory=dict)
    min_sector: dict[str, float] = field(default_factory=dict)


def sector_bounds(req: UserRequest) -> dict[str, tuple[float, float]]:
    """Turn the user's preferences into the optimizer's {sector: (lower, upper)} bounds.

    Parameters
    ----------
    req : the user request.

    Returns
    -------
    Bounds for every sector the user mentioned: lower = the minimum (default 0), upper
    = 0 if excluded, else the maximum (default 1). Values are passed through unchanged
    so ``validate_request`` can reject impossible ones (e.g. a maximum of 150%).

    Theorem: none (constraint set of the main problem, plan 2.2).
    """
    bounds: dict[str, tuple[float, float]] = {}
    for s in set(req.exclude) | set(req.max_sector) | set(req.min_sector):
        lo = req.min_sector.get(s, 0.0)
        hi = 0.0 if s in req.exclude else req.max_sector.get(s, 1.0)
        bounds[s] = (lo, hi)
    return bounds


def validate_request(req: UserRequest, sectors: pd.Series, cfg: dict) -> None:
    """Reject requests that are malformed or cannot be satisfied, with a clear reason.

    Checks: amount > 0; known risk level; known sector names; bounds within [0, 1]
    with lower ≤ upper; minimums summing to at most 1; and that the allowed stocks
    can hold 100% under the per-stock cap and the sector maximums.

    Parameters
    ----------
    req : the user request.
    sectors : ticker -> sector.
    cfg : config (risk levels, w_max).

    Raises
    ------
    ValueError explaining what to change.

    Theorem: none (feasibility of the constraint set; T1 needs it non-empty).
    """
    if not req.amount > 0:
        raise ValueError("The investment amount must be positive.")
    if req.risk not in cfg["risk_tolerance"]:
        raise ValueError(f"Risk tolerance must be one of {list(cfg['risk_tolerance'])}.")
    known = set(sectors)
    named = set(req.exclude) | set(req.max_sector) | set(req.min_sector)
    if named - known:
        raise ValueError(f"Unknown sector(s) {sorted(named - known)}. Choose from {sorted(known)}.")
    bounds = sector_bounds(req)
    for s, (lo, hi) in bounds.items():
        if not (0 <= lo <= 1 and 0 <= hi <= 1):
            raise ValueError(f"Bounds for {s} must be between 0 and 1.")
        if lo > hi:
            raise ValueError(f"{s}: the minimum ({lo:.0%}) is above the maximum ({hi:.0%}).")
    if sum(lo for lo, _ in bounds.values()) > 1 + 1e-12:
        raise ValueError("The sector minimums add up to more than 100%.")
    w_max = cfg["backtest"]["w_max"]
    capacity = 0.0
    for s in known:
        n = int((sectors == s).sum())
        lo, hi = bounds.get(s, (0.0, 1.0))
        cap_s = min(hi, n * w_max)
        if lo > cap_s + 1e-12:
            raise ValueError(f"{s} cannot reach {lo:.0%}: it has {n} stocks capped at {w_max:.0%} each.")
        capacity += cap_s
    if capacity < 1 - 1e-12:
        raise ValueError(f"These preferences leave room for only {capacity:.0%} of the portfolio "
                         f"(each stock is capped at {w_max:.0%}). Exclude fewer sectors or raise caps.")


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

    Step 1: shares = ⌊wᵢ·amount / priceᵢ⌋. Step 2a: if rounding left a sector below its
    minimum, buy shares in that sector first (within every cap). Step 2b: with the
    leftover cash, repeatedly
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
    bounds : user sector bounds.

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
    lo = {s: b[0] for s, b in bounds.items() if b[0] > 0}
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

    # Step 2a: rounding down can leave a sector below its minimum; refill those first,
    # buying the share of the most under-filled stock in each short sector.
    while lo:
        v = value()
        sector_val = v.groupby(sec_of).sum()
        short = [s for s, m in lo.items() if sector_val.get(s, 0.0) < m * amount - 1e-9]
        if not short:
            break
        ok = allowed(v) & sec_of.isin(short)
        if not ok.any():
            break                               # reported by the constraint check
        pick = (target * amount - v)[ok].idxmax()
        shares[pick] += 1
        cash -= float(px[pick])

    # Step 2b: spend what is left only where it moves a holding closer to its target.
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
    dict: long_only, cap, sector_max, sector_min, excluded_zero, (budget).

    Theorem: none (feasible set of the main problem, plan 2.2).
    """
    by_sector = weights.groupby(sectors.reindex(weights.index)).sum()
    out = {
        "long_only": bool((weights >= -tol).all()),
        "cap": bool((weights <= w_max + tol).all()),
        "sector_max": all(by_sector.get(s, 0.0) <= hi + tol for s, (_, hi) in bounds.items()),
        "sector_min": all(by_sector.get(s, 0.0) >= lo - tol for s, (lo, _) in bounds.items()),
        "excluded_zero": all(by_sector.get(s, 0.0) <= tol for s, (_, hi) in bounds.items()
                             if hi == 0),
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
    req : amount, risk tolerance and sector preferences.
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
    bounds = sector_bounds(req)

    fc, ev = forecasts if forecasts is not None else forecasts_up_to(data, cfg, t)
    hist: History = history_at(data, t)
    inp = bl_inputs(hist, fc, ev, stocks, cfg, head["omega"], bl["tau"])
    cons = Constraints(long_only=True, w_max=w_max, sectors=sectors, sector_bounds=bounds)
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
    }
    sec = pd.DataFrame({"target": target.groupby(sectors).sum(),
                        "held": held_w.groupby(sectors).sum()})
    for s, (lo, hi) in bounds.items():
        sec.loc[s, "min"], sec.loc[s, "max"] = lo, hi
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
    prefs = []
    if req.exclude:
        prefs.append("exclude " + ", ".join(req.exclude))
    prefs += [f"{k} ≤ {v:.0%}" for k, v in req.max_sector.items()]
    prefs += [f"{k} ≥ {v:.0%}" for k, v in req.min_sector.items()]
    lines = [
        f"Recommended portfolio as of {s['as_of']} (simulated; no trades are placed)",
        f"Amount ${req.amount:,.0f}  ·  risk tolerance {req.risk} (γ = {rec.gamma})  ·  "
        f"preferences: {'; '.join(prefs) or 'none'}",
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


def _parse_bounds(items: list[str]) -> dict[str, float]:
    out = {}
    for item in items:
        name, _, val = item.rpartition("=")
        if not name:
            raise ValueError(f"Use the form 'Sector=0.2', got {item!r}")
        out[name.strip()] = float(val)
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="Recommend a simulated portfolio.")
    p.add_argument("--amount", type=float, required=True, help="investment amount in $")
    p.add_argument("--risk", required=True, choices=["low", "medium", "high"])
    p.add_argument("--exclude", action="append", default=[], help="sector to exclude")
    p.add_argument("--max-sector", action="append", default=[], help='e.g. "Energy=0.1"')
    p.add_argument("--min-sector", action="append", default=[], help='e.g. "Health Care=0.15"')
    p.add_argument("--as-of", default=None, help="recommendation date (default: latest)")
    p.add_argument("--json", default=None, help="also save the result to this JSON file")
    p.add_argument("--config", default="config.yaml")
    a = p.parse_args()
    cfg = load_config(a.config)
    req = UserRequest(a.amount, a.risk, a.exclude, _parse_bounds(a.max_sector),
                      _parse_bounds(a.min_sector))
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

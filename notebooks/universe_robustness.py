"""R2: survivorship and universe-selection check (DECISIONS R2).

Run from the repo root after ``python -m src.wide_universe``:
    python -m notebooks.universe_robustness

Reruns the main pipeline (13 features, elastic net with the Phase 3 walk-forward,
S1 / S5 / S6 / BL prior, 10% cap, 25% sector rule, 10 bps) on 22 universes of 51 stocks
with the original sector counts: the original 51 (a check that the main results are
reproduced), the "2015 liquidity" universe (highest 2015 dollar volume in each sector,
chosen with information available at the start of the backtest), and 20 random draws
from the 417 S&P 500 members with full history. Also compares 1/N on the original 51,
1/N on all 417 pool stocks, and SPY.

Writes ``report/tables/r2_universes.csv`` (one row per universe) and
``report/tables/r2_summary.csv``.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.backtest import (black_litterman_strategy, buy_and_hold, equal_weight,
                          forecast_strategy, rebalance_dates, run_backtest)
from src.data import load_config
from src.features import add_targets, build_feature_panel, rank_normalize
from src.metrics import performance_summary
from src.models import ElasticNetModel, walk_forward
from src.wide_universe import load_wide

N_RANDOM = 20


def draw_universes(wide: dict, original: list[str], cfg: dict) -> dict[str, list[str]]:
    """The original 51, the 2015-liquidity universe and N_RANDOM random universes, each
    with the original number of stocks per sector."""
    uni = wide["universe"]
    pool = uni[(uni["index"] == "S&P 500") & uni.index.isin(wide["shares_outstanding"].index)]
    counts = pool.loc[original, "sector"].value_counts()
    dv = (wide["close"] * wide["volume"]).loc["2015"].mean()
    out = {"Original 51": original}
    liq = []
    for sector, k in counts.items():
        members = pool.index[pool["sector"] == sector]
        liq += list(dv.reindex(members).sort_values(ascending=False).index[:k])
    out["2015 liquidity"] = liq
    for seed in range(N_RANDOM):
        rng = np.random.default_rng(seed)
        pick = []
        for sector, k in counts.items():
            members = sorted(pool.index[pool["sector"] == sector])
            pick += list(rng.choice(members, size=k, replace=False))
        out[f"Random {seed:02d}"] = pick
    return out


def run_universe(stocks: list[str], wide: dict, cfg: dict, dates) -> dict:
    """Retrain the elastic net on ``stocks`` and backtest S1, S5, S6 and BL prior."""
    ml, bt = cfg["ml"], cfg["backtest"]
    raw = build_feature_panel(wide, stocks, cfg["data"]["benchmark"], ml["windows"],
                              ml["rsi_days"])
    panel = add_targets(rank_normalize(raw), wide["returns_monthly"][stocks],
                        wide["rf_monthly"]["rf"])
    en = ml["elastic_net"]
    model = ElasticNetModel(en["alphas"], en["l1_ratios"], en["max_iter"], cfg["seed"],
                            retrain_every=ml["retrain_every_months"])
    fc = walk_forward(model, panel, dates, ml["val_months"], ml["min_train_months"]).forecasts
    g = cfg["risk_tolerance"]["medium"]
    tau = cfg["black_litterman"]["tau"]
    strategies = {"S1": equal_weight(stocks), "S5": forecast_strategy(fc, stocks, cfg, g),
                  "S6": black_litterman_strategy(fc, None, stocks, cfg, g, "A", tau),
                  "Prior": black_litterman_strategy(fc, None, stocks, cfg, g, "A", tau,
                                                    views=False)}
    rf = wide["rf_monthly"]["rf"]
    row = {}
    for name, strat in strategies.items():
        res = run_backtest(name, strat, wide, dates, cfg["optimization"]["weight_tol"])
        net = res.net(bt["cost_bps"])
        p = performance_summary(net, rf)
        row[f"{name}_return"] = p["ann_return"]
        row[f"{name}_sharpe"] = p["sharpe"]
        row[f"{name}_turnover"] = float(res.turnover.iloc[1:].mean())
    return row


def main() -> None:
    cfg = load_config()
    wide = load_wide()
    original = list(cfg["universe"])
    dates = rebalance_dates(wide["returns_monthly"].index, cfg["backtest"]["oos_start"])
    universes = draw_universes(wide, original, cfg)
    out = Path(cfg["phase2"]["table_dir"])

    rows = []
    for name, stocks in universes.items():
        t0 = time.time()
        row = {"universe": name, "overlap_with_original": len(set(stocks) & set(original)),
               **run_universe(stocks, wide, cfg, dates)}
        rows.append(row)
        pd.DataFrame(rows).to_csv(out / "r2_universes.csv", index=False)
        print(f"{name:15s} S5 to {row['S5_turnover']:.0%} -> S6 to {row['S6_turnover']:.0%}  "
              f"Sharpe S6 {row['S6_sharpe']:.2f} prior {row['Prior_sharpe']:.2f} "
              f"1/N {row['S1_sharpe']:.2f}  ({time.time() - t0:.0f}s)", flush=True)
    table = pd.DataFrame(rows).set_index("universe")
    table["S6_minus_S5_turnover"] = table["S6_turnover"] - table["S5_turnover"]
    table["S6_minus_prior_sharpe"] = table["S6_sharpe"] - table["Prior_sharpe"]
    table["S6_minus_S5_sharpe"] = table["S6_sharpe"] - table["S5_sharpe"]
    table.to_csv(out / "r2_universes.csv")

    rnd = table[table.index.str.startswith("Random")]
    summary = pd.DataFrame({"median": rnd.median(numeric_only=True),
                            "min": rnd.min(numeric_only=True),
                            "max": rnd.max(numeric_only=True)})
    summary.loc["share_S6_trades_less_than_S5", "median"] = float((rnd["S6_minus_S5_turnover"] < 0).mean())
    summary.loc["share_S6_beats_prior", "median"] = float((rnd["S6_minus_prior_sharpe"] > 0).mean())
    summary.to_csv(out / "r2_summary.csv")

    # Level comparison: how much the hand-picked 51 flatter returns.
    uni = wide["universe"]
    pool = list(uni.index[(uni["index"] == "S&P 500") & uni.index.isin(wide["shares_outstanding"].index)])
    rf = wide["rf_monthly"]["rf"]
    levels = {}
    for name, strat in (("1/N, original 51", equal_weight(original)),
                        ("1/N, all 417 pool stocks", equal_weight(pool)),
                        ("SPY", buy_and_hold(cfg["data"]["benchmark"]))):
        net = run_backtest(name, strat, wide, dates, cfg["optimization"]["weight_tol"]).net(
            cfg["backtest"]["cost_bps"])
        p = performance_summary(net, rf)
        levels[name] = {"ann_return": p["ann_return"], "sharpe": p["sharpe"]}
    levels = pd.DataFrame(levels).T
    levels.to_csv(out / "r2_levels.csv")

    pd.set_option("display.width", 200)
    cols = ["overlap_with_original", "S1_return", "S6_return", "Prior_return", "S5_turnover",
            "S6_turnover", "S1_sharpe", "S5_sharpe", "S6_sharpe", "Prior_sharpe",
            "S6_minus_prior_sharpe"]
    print("\n" + table[cols].round(3).to_string())
    print("\nRandom universes (20):\n" + summary.round(3).to_string())
    print("\nLevels:\n" + levels.round(3).to_string())


if __name__ == "__main__":
    main()

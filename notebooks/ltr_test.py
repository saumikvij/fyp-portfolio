"""EXPERIMENT (branch experiment/model-search): learning to rank, MS15 strategy 1.

Run:  python -m notebooks.ltr_test [--placebo]

Universes: the 51 large caps, the 1,052-stock wide universe, and its mid + small caps.
Same walk-forward and tests as the main search: choice on development data with the
3-block rule; holdout flagged post-holdout; Black–Litterman portfolio for the 51 stocks.
"""

from __future__ import annotations

import argparse
import time
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

from src.backtest import black_litterman_strategy, rebalance_dates, run_backtest
from src.data import load_config, load_processed
from src.experiment_models import RankerModel, shuffle_targets
from src.experiment_signals import NICHE_SET, build_niche_panel, rank_normalize_niche
from src.experiment_universe import WIDE, load_wide
from src.features import FEATURES, add_targets, build_feature_panel, rank_normalize
from src.metrics import max_drawdown, performance_summary
from src.ml_eval import rank_ic, summarize_series
from src.models import walk_forward

OUT = Path("report/experiments")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--placebo", action="store_true")
    placebo = ap.parse_args().placebo
    cfg = load_config()
    ms, ml, bt = cfg["model_search"], cfg["ml"], cfg["backtest"]
    dev_end = pd.Timestamp(ms["dev_end"])
    cuts = [pd.Timestamp(c) for c in ms["dev_blocks"]]
    fc_dir = Path(ms["out_dir"] + ("_ltr_placebo" if placebo else "_ltr"))
    fc_dir.mkdir(parents=True, exist_ok=True)
    feats = {"base": FEATURES, "niche": NICHE_SET, "risk": ms["round4"]["risk_set"]}

    # ---- panels per universe (rank-normalized signals + targets)
    data51 = load_processed(cfg)
    s51 = list(cfg["universe"])
    bench = cfg["data"]["benchmark"]
    p51 = {"base": add_targets(rank_normalize(build_feature_panel(
        data51, s51, bench, ml["windows"], ml["rsi_days"], "base")),
        data51["returns_monthly"][s51], data51["rf_monthly"]["rf"]),
        "niche": add_targets(rank_normalize_niche(build_niche_panel(
            data51, s51, bench, ml["windows"], ml["rsi_days"],
            Path(cfg["data"]["raw_dir"]) / "prices_raw.parquet")),
            data51["returns_monthly"][s51], data51["rf_monthly"]["rf"])}
    wide = load_wide()
    pw = {fs: pd.read_parquet(WIDE / f"panel_{fs}.parquet") for fs in ["base", "niche"]}
    ms_tickers = wide["universe"].index[wide["universe"]["index"].isin(["S&P 400", "S&P 600"])]
    pms = {fs: p[p.index.get_level_values("ticker").isin(ms_tickers)] for fs, p in pw.items()}
    universes = {"large51": p51, "wide": pw, "midsmall": pms}
    for u in universes.values():
        u["risk"] = u["niche"]
    if placebo:
        universes = {k: {fs: shuffle_targets(p, ms["placebo"]["seed"]) for fs, p in v.items()}
                     for k, v in universes.items()}

    dates = rebalance_dates(data51["returns_monthly"].index, bt["oos_start"])
    dev, hold = dates[dates <= dev_end], dates[dates > dev_end]
    grids = {"large51": {"num_leaves": [7, 15], "n_estimators": [100, 300]},
             "wide": {"num_leaves": [7, 15], "n_estimators": [100]},
             "midsmall": {"num_leaves": [7, 15], "n_estimators": [100]}}

    rows, fcs = [], {}
    for uni, win, fs in product(universes, [None, 60], ["base", "niche", "risk"]):
        panels = universes[uni]
        sid = f"ltr__{uni}__{'expanding' if win is None else f'roll{win}m'}__{fs}"
        path = fc_dir / f"{sid}.parquet"
        t0 = time.time()
        if path.exists():
            f = pd.read_parquet(path)
        else:
            f = walk_forward(RankerModel(grids[uni], cfg["seed"], features=feats[fs]),
                             panels[fs], dates, ml["val_months"], ml["min_train_months"],
                             train_window_months=win).forecasts
            f.to_parquet(path)
        fcs[sid] = f
        realized = panels["base"]["target"].unstack().reindex(index=dates, columns=f.columns)
        ic = rank_ic(f.loc[dev], realized.loc[dev])
        blocks = [ic[(ic.index > a) & (ic.index <= b)].mean()
                  for a, b in zip([pd.Timestamp.min] + cuts, cuts + [dev_end])]
        s = summarize_series(ic)
        rows.append({"setup": sid, "universe": uni, "dev_ic": s["mean"], "dev_t": s["t_stat_nw"],
                     **{f"block{j + 1}": v for j, v in enumerate(blocks)},
                     "consistent": all(np.nan_to_num(v, nan=-1) > 0 for v in blocks)})
        print(f"{sid:36s} dev IC {s['mean']:+.4f} (t {s['t_stat_nw']:+.2f}) "
              f"consistent {rows[-1]['consistent']}  {time.time() - t0:.0f}s", flush=True)
    log = pd.DataFrame(rows).sort_values("dev_ic", ascending=False)
    log.to_csv(OUT / ("ltr_placebo_log.csv" if placebo else "ltr_log.csv"), index=False)
    if placebo:
        print(log.groupby("universe")[["dev_ic"]].agg(["max", "median"]).round(4))
        return

    for uni in universes:
        cons = log[(log["universe"] == uni) & log["consistent"]]
        if cons.empty:
            print(f"\n{uni}: no consistent setup")
            continue
        sid = cons.iloc[0]["setup"]
        f = fcs[sid]
        realized = universes[uni]["base"]["target"].unstack().reindex(index=dates, columns=f.columns)
        ich = rank_ic(f.loc[hold], realized.loc[hold])
        sh = summarize_series(ich)
        print(f"\n{uni}: chosen {sid}; holdout IC {sh['mean']:+.4f} (NW t {sh['t_stat_nw']:+.2f}) "
              f"by year {ich.groupby(ich.index.year).mean().round(3).to_dict()} [post-holdout]")
        if uni == "large51":
            g, tau = cfg["risk_tolerance"]["medium"], cfg["black_litterman"]["tau"]
            res = run_backtest("S6 LTR", black_litterman_strategy(f, None, s51, cfg, g, "A", tau),
                               data51, dates, cfg["optimization"]["weight_tol"])
            net = res.net(bt["cost_bps"])
            split = data51["returns_monthly"].index[data51["returns_monthly"].index > dev_end][0]
            rf = data51["rf_monthly"]["rf"]
            for label, seg in (("dev", net.loc[:split]), ("hold", net.loc[split:].iloc[1:])):
                print(f"  S6 with LTR views, {label}: Sharpe {performance_summary(seg, rf)['sharpe']:.3f}, "
                      f"max drawdown {max_drawdown(seg):.1%}")


if __name__ == "__main__":
    main()

"""EXPERIMENT (branch experiment/model-search): IPCA, added after the holdout run (MS14).

Run:  python -m notebooks.ipca_test

Same walk-forward, windows, targets and signal sets as the main search. The best setup
is chosen on development scores with the 3-block rule (MS3 + MS5); its holdout result
and its Black–Litterman portfolio are then reported, flagged as post-holdout.
"""

from __future__ import annotations

from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

from notebooks.model_search import setup_id
from src.backtest import black_litterman_strategy, rebalance_dates, run_backtest
from src.data import load_config, load_processed
from src.experiment_models import IPCAModel
from src.experiment_signals import NICHE_SET, build_niche_panel, rank_normalize_niche
from src.features import FEATURE_SETS, add_targets, build_feature_panel, rank_normalize
from src.metrics import max_drawdown, performance_summary
from src.ml_eval import rank_ic, summarize_series
from src.models import walk_forward

OUT = Path("report/experiments")


def main() -> None:
    cfg = load_config()
    ms, ml, bt = cfg["model_search"], cfg["ml"], cfg["backtest"]
    data = load_processed(cfg)
    stocks = list(cfg["universe"])
    bench = cfg["data"]["benchmark"]
    dates = rebalance_dates(data["returns_monthly"].index, bt["oos_start"])
    dev_end = pd.Timestamp(ms["dev_end"])
    cuts = [pd.Timestamp(c) for c in ms["dev_blocks"]]
    dev, hold = dates[dates <= dev_end], dates[dates > dev_end]
    fc_dir = Path(ms["out_dir"])

    def panel(fs):
        if fs == "niche":
            raw = rank_normalize_niche(build_niche_panel(
                data, stocks, bench, ml["windows"], ml["rsi_days"],
                Path(cfg["data"]["raw_dir"]) / "prices_raw.parquet"))
        else:
            raw = rank_normalize(build_feature_panel(data, stocks, bench, ml["windows"],
                                                     ml["rsi_days"], fs))
        return add_targets(raw, data["returns_monthly"][stocks], data["rf_monthly"]["rf"])

    panels = {fs: panel(fs) for fs in ["base", "extended", "niche"]}
    panels["risk"] = panels["niche"]
    feats = {**FEATURE_SETS, "niche": NICHE_SET, "risk": ms["round4"]["risk_set"]}
    realized = panels["base"]["target"].unstack().loc[dates]

    rows, fcs = [], {}
    for win, tgt, fs in product(ms["train_windows"], ms["targets"], list(panels)):
        sid = setup_id("IPCA", win, tgt, fs)
        path = fc_dir / f"{sid}.parquet"
        if path.exists():
            f = pd.read_parquet(path)
        else:
            f = walk_forward(IPCAModel({"K": [1, 2, 3, 5]}, cfg["seed"], features=feats[fs]),
                             panels[fs], dates, ml["val_months"], ml["min_train_months"],
                             train_window_months=win, target_mode=tgt).forecasts
            f.to_parquet(path)
        fcs[sid] = f
        ic = rank_ic(f.loc[dev], realized.loc[dev])
        blocks = [ic[(ic.index > a) & (ic.index <= b)].mean()
                  for a, b in zip([pd.Timestamp.min] + cuts, cuts + [dev_end])]
        s = summarize_series(ic)
        rows.append({"setup": sid, "dev_ic": s["mean"], "dev_t": s["t_stat_nw"],
                     **{f"block{j + 1}": v for j, v in enumerate(blocks)},
                     "consistent": all(np.nan_to_num(v, nan=-1) > 0 for v in blocks)})
        print(f"{sid:44s} dev IC {s['mean']:+.4f}  consistent {rows[-1]['consistent']}", flush=True)
    log = pd.DataFrame(rows).sort_values("dev_ic", ascending=False)
    log.to_csv(OUT / "ipca_log.csv", index=False)
    print(f"\nConsistent in all 3 blocks: {int(log['consistent'].sum())} of {len(log)}")

    cons = log[log["consistent"]]
    if cons.empty:
        print("No consistent IPCA setup; nothing goes to the holdout.")
        return
    best = cons.iloc[0]["setup"]
    f = fcs[best]
    ich = rank_ic(f.loc[hold], realized.loc[hold])
    sh = summarize_series(ich)
    print(f"\nChosen (dev only): {best}\nHoldout IC {sh['mean']:+.4f} (NW t {sh['t_stat_nw']:+.2f}); "
          f"by year {ich.groupby(ich.index.year).mean().round(3).to_dict()}  [post-holdout, MS14]")

    g, tau = cfg["risk_tolerance"]["medium"], cfg["black_litterman"]["tau"]
    res = run_backtest("S6 IPCA", black_litterman_strategy(f, None, stocks, cfg, g, "A", tau),
                       data, dates, cfg["optimization"]["weight_tol"])
    net = res.net(bt["cost_bps"])
    split = data["returns_monthly"].index[data["returns_monthly"].index > dev_end][0]
    rf = data["rf_monthly"]["rf"]
    for label, seg in (("dev", net.loc[:split]), ("hold", net.loc[split:].iloc[1:])):
        print(f"S6 with IPCA views, {label}: Sharpe {performance_summary(seg, rf)['sharpe']:.3f}, "
              f"max drawdown {max_drawdown(seg):.1%}")


if __name__ == "__main__":
    main()

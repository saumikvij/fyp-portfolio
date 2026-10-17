"""EXPERIMENT (branch experiment/model-search): the model search on the wide universe
(~1,050 stocks, DECISIONS WU1). Development period only; holdout untouched.

Run after ``python -m src.experiment_universe``:
    python -m notebooks.wide_search             # real search
    python -m notebooks.wide_search --placebo   # same, targets shuffled within each month

Scores each setup on development forecasts (Dec 2015 – Dec 2021): rank IC over the whole
universe, by size group (S&P 500 / 400 / 600 members), and on the original 51 stocks,
plus the 3-block consistency rule. Re-running skips setups already saved.
"""

from __future__ import annotations

import argparse
import copy
import time
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

from notebooks.model_search import build_model, setup_id
from src.backtest import rebalance_dates
from src.data import load_config
from src.experiment_models import shuffle_targets
from src.experiment_signals import NICHE_SET, build_niche_panel, rank_normalize_niche
from src.experiment_universe import WIDE, load_wide
from src.features import FEATURE_SETS, add_targets, build_feature_panel, rank_normalize
from src.ml_eval import rank_ic, summarize_series
from src.models import walk_forward

OUT = Path("report/experiments")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--placebo", action="store_true")
    placebo = ap.parse_args().placebo
    cfg = load_config()
    ms, ml, bt = cfg["model_search"], cfg["ml"], cfg["backtest"]
    wcfg = ms["wide"]
    # Lighter settings for the larger data (WU1); everything else as in the main search.
    cfg = copy.deepcopy(cfg)
    for fam, over in wcfg["overrides"].items():
        cfg["model_search"][fam].update(over)
    fc_dir = Path(wcfg["out_dir"] + ("_placebo" if placebo else ""))
    log_name = "wide_placebo_log.csv" if placebo else "wide_search_log.csv"
    fc_dir.mkdir(parents=True, exist_ok=True)

    data = load_wide()
    uni = data["universe"]
    stocks = list(uni.index)
    bench = cfg["data"]["benchmark"]
    dev_end = pd.Timestamp(ms["dev_end"])
    cuts = [pd.Timestamp(c) for c in ms["dev_blocks"]]
    dates = rebalance_dates(data["returns_monthly"].index, bt["oos_start"])
    dev = dates[dates <= dev_end]

    def make_panel(fs):
        path = WIDE / f"panel_{fs}.parquet"
        if path.exists():
            return pd.read_parquet(path)
        t0 = time.time()
        if fs == "niche":
            raw = rank_normalize_niche(build_niche_panel(data, stocks, bench, ml["windows"],
                                                         ml["rsi_days"], WIDE / "raw" / "prices_raw.parquet"))
        else:
            raw = rank_normalize(build_feature_panel(data, stocks, bench, ml["windows"],
                                                     ml["rsi_days"], fs))
        p = add_targets(raw, data["returns_monthly"][stocks], data["rf_monthly"]["rf"])
        p.to_parquet(path)
        print(f"panel {fs}: {p.shape} in {time.time() - t0:.0f}s")
        return p

    panels = {fs: make_panel(fs) for fs in wcfg["feature_sets"]}
    if placebo:
        panels = {fs: shuffle_targets(p, ms["placebo"]["seed"]) for fs, p in panels.items()}
    feature_lists = {**FEATURE_SETS, "niche": NICHE_SET}
    realized = panels["base"]["target"].unstack().loc[dev]
    groups = {"all": stocks, **{g: list(uni.index[uni["index"] == g])
                                for g in ["S&P 500", "S&P 400", "S&P 600"]},
              "original 51": [s for s in cfg["universe"] if s in stocks]}

    rows = []
    fams = ms["placebo"]["families"] if placebo else ms["families"]
    grid = list(product(fams, wcfg["windows"], wcfg["targets"], wcfg["feature_sets"]))
    for i, (fam, win, tgt, fs) in enumerate(grid, 1):
        sid = setup_id(fam, win, tgt, fs)
        path = fc_dir / f"{sid}.parquet"
        t0 = time.time()
        if path.exists():
            f = pd.read_parquet(path)
        else:
            f = walk_forward(build_model(fam, cfg, feature_lists[fs]), panels[fs], dates,
                             ml["val_months"], ml["min_train_months"], train_window_months=win,
                             target_mode=tgt).forecasts
            f.to_parquet(path)
        fd = f.loc[dev]
        row = {"setup": sid, "family": fam, "target": tgt, "features": fs}
        for g, cols in groups.items():
            ic = rank_ic(fd[cols], realized[cols])
            s = summarize_series(ic) if ic.notna().any() else {"mean": np.nan, "t_stat_nw": np.nan}
            row[f"ic_{g}"], row[f"t_{g}"] = s["mean"], s["t_stat_nw"]
            if g == "all":
                blocks = [ic[(ic.index > a) & (ic.index <= b)].mean()
                          for a, b in zip([pd.Timestamp.min] + cuts, cuts + [dev_end])]
                row.update({f"block{j + 1}_ic": v for j, v in enumerate(blocks)})
                row["consistent"] = bool(all(np.nan_to_num(v, nan=-1) > 0 for v in blocks))
        rows.append(row)
        pd.DataFrame(rows).to_csv(OUT / log_name, index=False)   # save as we go
        print(f"[{i:2d}/{len(grid)}] {sid:52s} done  {time.time() - t0:5.0f}s", flush=True)

    log = pd.DataFrame(rows).sort_values("ic_all", ascending=False)
    log.to_csv(OUT / log_name, index=False)
    print(f"\nWide universe ({len(stocks)} stocks){' PLACEBO' if placebo else ''}, development only:")
    print(log[["setup", "ic_all", "t_all", "ic_S&P 500", "ic_S&P 400", "ic_S&P 600",
               "ic_original 51", "consistent"]].round(4).head(15).to_string(index=False))
    print(f"\nConsistent in all 3 blocks: {int(log['consistent'].sum())} of {len(log)}")


if __name__ == "__main__":
    main()

"""EXPERIMENT (branch experiment/model-search): IPCA and industry rotation as Black–Litterman
views on the wide universe (DECISIONS WU6). Run after ``notebooks.wide_bl_models``.

Run:  python -m notebooks.wide_bl_extra

IPCA: same grid and choice rule as MS14 (windows {expanding, 5 years} × targets × signal
sets {base, niche, risk}; 3-block rule, then best development IC), trained on the wide
panels. Industry rotation: the sector forecasts of MS15.2 (already built from the wide
universe) as sector views (general P, T6) on all wide stocks. Same portfolio settings,
outputs and paired tests as WU5; holdout flagged post-holdout.
"""

from __future__ import annotations

import copy
import time
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

from notebooks.model_search import setup_id
from notebooks.wide_bl_models import OUT, accuracy
from src.backtest import black_litterman_strategy, rebalance_dates, run_backtest
from src.data import load_config
from src.experiment_models import IPCAModel
from src.experiment_sectors import sector_bl_strategy, sector_returns
from src.experiment_signals import NICHE_SET
from src.experiment_universe import WIDE, load_wide
from src.features import FEATURE_SETS
from src.metrics import (cvar_historical, holm_adjust, max_drawdown, performance_summary,
                         sharpe_difference_bootstrap)
from src.ml_eval import rank_ic, summarize_series
from src.models import walk_forward


def ipca_views(cfg, dates, realized) -> pd.DataFrame:
    ms, ml = cfg["model_search"], cfg["ml"]
    dev_end = pd.Timestamp(ms["dev_end"])
    cuts = [pd.Timestamp(c) for c in ms["dev_blocks"]]
    dev = dates[dates <= dev_end]
    fc_dir = Path(ms["wide"]["out_dir"])
    panels = {fs: pd.read_parquet(WIDE / f"panel_{fs}.parquet") for fs in ["base", "niche"]}
    panels["risk"] = panels["niche"]
    feats = {**FEATURE_SETS, "niche": NICHE_SET, "risk": ms["round4"]["risk_set"]}
    rows, fcs = [], {}
    for win, tgt, fs in product([None, 60], ms["targets"], ["base", "niche", "risk"]):
        sid = setup_id("IPCA", win, tgt, fs)
        path = fc_dir / f"{sid}.parquet"
        t0 = time.time()
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
        rows.append({"setup": sid, "dev_ic": summarize_series(ic)["mean"],
                     "dev_t": summarize_series(ic)["t_stat_nw"],
                     **{f"block{j + 1}": v for j, v in enumerate(blocks)},
                     "consistent": all(np.nan_to_num(v, nan=-1) > 0 for v in blocks)})
        print(f"  {sid:40s} dev IC {rows[-1]['dev_ic']:+.4f} consistent {rows[-1]['consistent']}"
              f"  ({time.time() - t0:.0f}s)", flush=True)
    log = pd.DataFrame(rows).sort_values("dev_ic", ascending=False)
    log.to_csv(OUT / "wide_ipca_log.csv", index=False)
    pool = log[log["consistent"]] if log["consistent"].any() else log
    best = pool.iloc[0]["setup"]
    print(f"IPCA chosen on development data: {best}")
    return fcs[best]


def main() -> None:
    cfg = load_config()
    ms, bt = cfg["model_search"], cfg["backtest"]
    port = ms["wide"]["portfolio"]
    wide = load_wide()
    uni = wide["universe"]
    stocks = [s for s in uni.index if s in wide["shares_outstanding"].index]
    dates = rebalance_dates(wide["returns_monthly"].index, bt["oos_start"])
    dev_end = pd.Timestamp(ms["dev_end"])
    split = wide["returns_monthly"].index[wide["returns_monthly"].index > dev_end][0]
    rf = wide["rf_monthly"]["rf"]
    g, tau = cfg["risk_tolerance"]["medium"], cfg["black_litterman"]["tau"]
    c = copy.deepcopy(cfg)
    c["backtest"].update(cov_method=port["cov_method"], w_max=port["w_max"])
    c["universe"] = {s: uni.loc[s, "sector"] for s in stocks}
    realized = pd.read_parquet(WIDE / "panel_base.parquet")["target"].unstack()
    realized = realized.reindex(index=dates, columns=stocks)

    print("IPCA on the wide universe:")
    ipca = ipca_views(cfg, dates, realized).reindex(columns=stocks)
    rot = pd.read_parquet(Path(ms["out_dir"]) / "sector_rotation_forecasts.parquet")
    strategies = {"BL + IPCA": black_litterman_strategy(ipca, None, stocks, c, g, "A", tau),
                  "BL + Industry rotation": sector_bl_strategy(rot, sector_returns(wide), stocks,
                                                               c, g, tau)}
    cache = Path(ms["wide"]["out_dir"]) / "bl_runs"
    prior = pd.read_parquet(cache / "BL_prior_only.parquet")["net"]
    rows, tests = [], []
    for name, strat in strategies.items():
        path = cache / f"{name.replace(' ', '_')}.parquet"
        t0 = time.time()
        if path.exists():
            saved = pd.read_parquet(path)
        else:
            res = run_backtest(name, strat, wide, dates, cfg["optimization"]["weight_tol"])
            saved = pd.DataFrame({"net": res.net(bt["cost_bps"]), "turnover": res.turnover,
                                  "holdings": (res.weights > 1e-5).sum(axis=1).values})
            saved.to_parquet(path)
        net = saved["net"]
        row = {"strategy": name, "avg_turnover": float(saved["turnover"].iloc[1:].mean()),
               "avg_holdings": float(saved["holdings"].mean())}
        for label, seg in (("dev", net.loc[:split]), ("hold", net.loc[split:].iloc[1:]),
                           ("all", net)):
            p = performance_summary(seg, rf)
            row.update({f"{label}_return": p["ann_return"], f"{label}_vol": p["ann_vol"],
                        f"{label}_sharpe": p["sharpe"], f"{label}_max_dd": max_drawdown(seg),
                        f"{label}_cvar95": cvar_historical(seg)})
        if name == "BL + IPCA":
            for label, d in (("dev", dates[dates <= dev_end]), ("hold", dates[dates > dev_end])):
                for k, v in accuracy(ipca.loc[d], realized.loc[d]).items():
                    row[f"{label}_{k}"] = v
        rows.append(row)
        for label, sl in (("hold", slice(split, None)), ("all", slice(None, None))):
            a, b = net.loc[sl], prior.loc[sl]
            if label == "hold":
                a, b = a.iloc[1:], b.iloc[1:]
            boot = sharpe_difference_bootstrap(a, b, rf, block=cfg["phase2"]["sharpe_test_block"],
                                               n_boot=cfg["phase2"]["sharpe_test_boot"],
                                               seed=cfg["seed"])
            tests.append({"strategy": name, "period": label, "sharpe_diff": boot["diff"],
                          "ci_lo": boot["ci_lo"], "ci_hi": boot["ci_hi"], "boot_p": boot["p_value"]})
        print(f"{name:26s} done in {time.time() - t0:5.0f}s", flush=True)
    table = pd.DataFrame(rows)
    table.to_csv(OUT / "wide_bl_extra.csv", index=False)
    tests = pd.DataFrame(tests)
    tests["boot_p_holm"] = holm_adjust(tests["boot_p"])
    tests.to_csv(OUT / "wide_bl_extra_tests.csv", index=False)
    print("\n" + table.round(3).to_string(index=False))
    print("\nPaired Sharpe tests vs BL prior only (holdout = post-holdout):")
    print(tests.round(3).to_string(index=False))


if __name__ == "__main__":
    main()

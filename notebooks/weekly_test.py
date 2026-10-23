"""EXPERIMENT (branch experiment/model-search): weekly rebalancing and forecasting (E1, WK1).

Run:  python -m notebooks.weekly_test                    # 51 stocks
      python -m notebooks.weekly_test --universe wide    # 1,052 / 1,047 stocks (WK2)
      (--only real / --only placebo: just train and save those forecasts, e.g. in parallel)

Wide universe (WK2): forecasts for all 1,052 stocks; portfolios on the 1,047 with a share
count, with Ledoit–Wolf Σ and a 2% cap as in WU4; S4 and S5 are not run there.

51 stocks, week-ends from the first week of 2016 (last decision July 2026). For each of
the 15 model families (base signals, raw target, expanding window, settings rescaled to
weeks: E1): next-week forecasts, a placebo with targets shuffled within each week, rank
IC (development / holdout, Newey–West t), the 3-block rule, up/down accuracy. Then
weekly portfolios: 1/N, SPY, Markowitz with historical means (S4), raw elastic-net
forecasts (S5), Black–Litterman prior only, and Black–Litterman (Ω-A, γ = 5, 10 bps) with
the views of every family and of their average. Paired Sharpe tests against the prior
(26-week blocks, Holm). The holdout (2022–26) is post-holdout.
"""

from __future__ import annotations

import argparse
import copy
import time
from pathlib import Path

import numpy as np
import pandas as pd

from notebooks.model_search import build_model
from src.backtest import (black_litterman_strategy, buy_and_hold, equal_weight,
                          forecast_strategy, markowitz_strategy, rebalance_dates, run_backtest)
from src.data import load_config, load_processed
from src.experiment_models import shuffle_targets
from src.experiment_universe import load_wide
from src.experiment_weekly import weekly_cfg, weekly_data, weekly_feature_panel, weekly_metrics
from src.features import FEATURES
from src.metrics import holm_adjust, sharpe_difference_bootstrap
from src.ml_eval import rank_ic, summarize_series
from src.models import walk_forward

OUT = Path("report/experiments")
FC = Path("data/ml/weekly")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--universe", choices=["51", "wide"], default="51")
    ap.add_argument("--only", choices=["real", "placebo"], default=None)
    args = ap.parse_args()
    wide = args.universe == "wide"
    cfg0 = load_config()
    cfg = weekly_cfg(cfg0)
    ms, ml, bt = cfg["model_search"], cfg["ml"], cfg["backtest"]
    data = load_wide() if wide else load_processed(cfg0)
    wk = weekly_data(data)
    if wide:
        uni = data["universe"]
        stocks = list(uni.index)
        port_stocks = [s_ for s_ in stocks if s_ in data["shares_outstanding"].index]
        port = ms["wide"]["portfolio"]
        cfg = copy.deepcopy(cfg)
        cfg["backtest"].update(cov_method=port["cov_method"], w_max=port["w_max"])
        cfg["universe"] = {s_: uni.loc[s_, "sector"] for s_ in port_stocks}
    else:
        stocks = list(cfg["universe"])
        port_stocks = stocks
    tag = "weekly_wide" if wide else "weekly"
    global FC
    FC = Path(f"data/ml/{tag}")
    bench = cfg["data"]["benchmark"]
    dates = rebalance_dates(wk["returns_monthly"].index, bt["oos_start"])
    dev_end = pd.Timestamp(ms["dev_end"])
    cuts = [pd.Timestamp(c) for c in ms["dev_blocks"]]
    dev, hold = dates[dates <= dev_end], dates[dates > dev_end]
    split = wk["returns_monthly"].index[wk["returns_monthly"].index > dev_end][0]
    rf = wk["rf_monthly"]["rf"]
    FC.mkdir(parents=True, exist_ok=True)
    (FC / "placebo").mkdir(exist_ok=True)

    ppath = FC / "panel_base.parquet"
    if ppath.exists():
        panel = pd.read_parquet(ppath)
    else:
        panel = weekly_feature_panel(data, wk, stocks, bench, ml["windows"], ml["rsi_days"])
        panel.to_parquet(ppath)
    placebo_panel = shuffle_targets(panel, ms["placebo"]["seed"])
    realized = panel["target"].unstack().reindex(index=dates, columns=stocks)
    print(f"Weekly panel {panel.shape}; {len(dates)} weekly decisions ({len(dev)} dev, {len(hold)} holdout)")

    def ic_stats(f, part):
        ic = rank_ic(f.loc[part], realized.loc[part])
        s = summarize_series(ic, 52) if ic.notna().any() else {"mean": np.nan, "t_stat_nw": np.nan}
        return ic, s

    rows, fcs = [], {}
    for fam in ms["families"]:
        t0 = time.time()
        out = {}
        for kind, pnl, d in (("real", panel, FC), ("placebo", placebo_panel, FC / "placebo")):
            if args.only and kind != args.only:
                continue
            path = d / f"{fam.lower().replace(' ', '_')}.parquet"
            if path.exists():
                out[kind] = pd.read_parquet(path)
            else:
                out[kind] = walk_forward(build_model(fam, cfg, FEATURES), pnl, dates,
                                         ml["val_months"], ml["min_train_months"]).forecasts
                out[kind].to_parquet(path)
        if args.only:
            print(f"{fam:24s} {args.only} forecasts saved ({time.time() - t0:.0f}s)", flush=True)
            continue
        f = out["real"].reindex(columns=stocks)
        fcs[fam] = f
        ic, sdev = ic_stats(f, dev)
        _, shold = ic_stats(f, hold)
        blocks = [ic[(ic.index > a) & (ic.index <= b)].mean()
                  for a, b in zip([pd.Timestamp.min] + cuts, cuts + [dev_end])]
        _, pdev = ic_stats(out["placebo"].reindex(columns=stocks), dev)
        _, phold = ic_stats(out["placebo"].reindex(columns=stocks), hold)
        ok = realized.loc[hold].notna()
        fh = f.loc[hold]
        acc_rel = ((fh.gt(fh.median(axis=1), axis=0)) ==
                   (realized.loc[hold].gt(realized.loc[hold].median(axis=1), axis=0)))[ok].stack().mean()
        rows.append({"family": fam, "dev_ic": sdev["mean"], "dev_t": sdev["t_stat_nw"],
                     "hold_ic": shold["mean"], "hold_t": shold["t_stat_nw"],
                     **{f"block{j + 1}": v for j, v in enumerate(blocks)},
                     "consistent": all(np.nan_to_num(v, nan=-1) > 0 for v in blocks),
                     "placebo_dev_ic": pdev["mean"], "placebo_hold_ic": phold["mean"],
                     "hold_acc_vs_median": float(acc_rel)})
        pd.DataFrame(rows).to_csv(OUT / f"{tag}_forecasts.csv", index=False)
        print(f"{fam:24s} dev IC {sdev['mean']:+.4f} (t {sdev['t_stat_nw']:+.2f})  hold IC "
              f"{shold['mean']:+.4f} (t {shold['t_stat_nw']:+.2f})  placebo dev {pdev['mean']:+.4f}  "
              f"consistent {rows[-1]['consistent']}  ({time.time() - t0:.0f}s)", flush=True)
    if args.only:
        return
    fcs["Average of 15 models"] = sum(fcs.values()) / len(fcs)
    up_rate = (realized.loc[hold] > 0).stack().mean()
    print(f"Always 'up' accuracy, holdout weeks: {up_rate:.1%}")

    g, tau = cfg["risk_tolerance"]["medium"], cfg["black_litterman"]["tau"]
    ps = port_stocks
    strategies = {"1/N": equal_weight(ps), "SPY": buy_and_hold(bench)}
    if not wide:
        strategies.update({"Markowitz, hist. mean (S4)": markowitz_strategy(ps, cfg, g),
                           "Raw elastic-net forecasts (S5)": forecast_strategy(fcs["Elastic net"],
                                                                               ps, cfg, g)})
    strategies["BL prior only"] = black_litterman_strategy(fcs["Elastic net"], None, ps, cfg,
                                                           g, "A", tau, views=False)
    for name, f in fcs.items():
        strategies[f"BL + {name}"] = black_litterman_strategy(f.reindex(columns=ps), None, ps,
                                                              cfg, g, "A", tau)
    cache = FC / "runs"
    cache.mkdir(exist_ok=True)
    prow, nets = [], {}
    for name, strat in strategies.items():
        t0 = time.time()
        path = cache / f"{name.replace(' ', '_').replace('/', '-')}.parquet"
        if path.exists():
            saved = pd.read_parquet(path)
        else:
            res = run_backtest(name, strat, wk, dates, cfg["optimization"]["weight_tol"])
            saved = pd.DataFrame({"net": res.net(bt["cost_bps"]), "turnover": res.turnover})
            saved.to_parquet(path)
        net = saved["net"]
        nets[name] = net
        row = {"strategy": name, "growth_10k": float(10_000 * (1 + net).prod()),
               "avg_weekly_turnover": float(saved["turnover"].iloc[1:].mean())}
        for label, seg in (("dev", net.loc[:split]), ("hold", net.loc[split:].iloc[1:]), ("all", net)):
            row.update({f"{label}_{k}": v for k, v in weekly_metrics(seg, rf).items()})
        prow.append(row)
        pd.DataFrame(prow).to_csv(OUT / f"{tag}_portfolios.csv", index=False)
        print(f"{name:34s} done in {time.time() - t0:4.0f}s", flush=True)
    table = pd.DataFrame(prow)

    tests = []
    for name in [n for n in nets if n.startswith("BL + ")]:
        for label, sl in (("hold", slice(split, None)), ("all", slice(None, None))):
            a, b = nets[name].loc[sl], nets["BL prior only"].loc[sl]
            if label == "hold":
                a, b = a.iloc[1:], b.iloc[1:]
            boot = sharpe_difference_bootstrap(a, b, rf, block=26, n_boot=cfg["phase2"]["sharpe_test_boot"],
                                               seed=cfg["seed"], periods_per_year=52)
            tests.append({"strategy": name, "period": label, "sharpe_diff": boot["diff"],
                          "boot_p": boot["p_value"]})
    tests = pd.DataFrame(tests)
    tests["boot_p_holm"] = holm_adjust(tests["boot_p"])
    tests.to_csv(OUT / f"{tag}_tests.csv", index=False)

    pd.set_option("display.width", 250)
    print("\n" + pd.DataFrame(rows).round(4).to_string(index=False))
    cols = ["strategy", "dev_return", "hold_return", "all_return", "growth_10k", "dev_sharpe",
            "hold_sharpe", "hold_max_dd", "avg_weekly_turnover"]
    print("\n" + table[cols].round(3).to_string(index=False))
    print("\nPaired Sharpe tests vs BL prior only (weekly, 26-week blocks):")
    print(tests.round(3).to_string(index=False))


if __name__ == "__main__":
    main()

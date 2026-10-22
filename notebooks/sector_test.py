"""EXPERIMENT (branch experiment/model-search): industry rotation, MS15 strategy 2.

Run:  python -m notebooks.sector_test

Forecasts the 11 sector portfolios with elastic net (one model per sector), compares with
the historical-mean baseline and a placebo (sector returns shuffled across sectors each
month), applies the 3-block rule on development data, then reports the holdout
(post-holdout) and the Black–Litterman portfolio with sector views. (No 2008–2009 test:
the sector portfolios use the wide universe, which starts in 2010.)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from notebooks.model_search import build_model
from src.backtest import (black_litterman_strategy, equal_weight, rebalance_dates,
                          run_backtest)
from src.data import load_config, load_processed
from src.experiment_sectors import sector_bl_strategy, sector_panels, sector_returns
from src.experiment_universe import load_wide
from src.metrics import max_drawdown, performance_summary
from src.ml_eval import r2_oos, rank_ic, summarize_series
from src.models import HistoricalMeanModel, walk_forward

OUT = Path("report/experiments")


def forecast_all(panels: dict, cfg: dict, dates, model_fn) -> pd.DataFrame:
    ml = cfg["ml"]
    cols = {}
    for s, p in panels.items():
        f = walk_forward(model_fn([c for c in p.columns if c.startswith("lag_")]), p, dates,
                         ml["val_months"], ml["min_train_months"]).forecasts
        cols[s] = f[s]
    return pd.DataFrame(cols)


def main() -> None:
    cfg = load_config()
    ms, bt = cfg["model_search"], cfg["backtest"]
    dev_end = pd.Timestamp(ms["dev_end"])
    cuts = [pd.Timestamp(c) for c in ms["dev_blocks"]]
    wide = load_wide()
    sret = sector_returns(wide)
    data = load_processed(cfg)
    dates = rebalance_dates(data["returns_monthly"].index, bt["oos_start"])
    dev, hold = dates[dates <= dev_end], dates[dates > dev_end]

    rng = np.random.default_rng(ms["placebo"]["seed"])
    shuffled = sret.apply(lambda row: pd.Series(rng.permutation(row.values), index=row.index),
                          axis=1)
    runs = {}
    for label, sr in (("real", sret), ("placebo", shuffled)):
        panels = sector_panels(sr)
        realized = pd.DataFrame({s: p["target"].droplevel("ticker")
                                 for s, p in panels.items()}).reindex(dates)
        en = forecast_all(panels, cfg, dates, lambda f: build_model("Elastic net", cfg, f))
        hm = forecast_all(panels, cfg, dates, lambda f: HistoricalMeanModel())
        runs[label] = (en, hm, realized)
        for name, part in (("dev", dev), ("hold", hold)):
            ic = rank_ic(en.loc[part], realized.loc[part])
            s = summarize_series(ic)
            extra = ""
            if name == "dev":
                blocks = [ic[(ic.index > a) & (ic.index <= b)].mean()
                          for a, b in zip([pd.Timestamp.min] + cuts, cuts + [dev_end])]
                extra = f" blocks {np.round(blocks, 3).tolist()}"
            flag = " [post-holdout]" if name == "hold" else ""
            print(f"{label:8s} {name:4s}: sector rank IC {s['mean']:+.4f} (NW t {s['t_stat_nw']:+.2f}), "
                  f"R²oos vs hist. mean {r2_oos(en.loc[part], realized.loc[part], hm.loc[part]):+.4f}"
                  f"{extra}{flag}")
    en, hm, realized = runs["real"]
    en.to_parquet(Path(ms["out_dir"]) / "sector_rotation_forecasts.parquet")

    # Top-3 minus bottom-3 sectors, equal weight, monthly.
    spread = pd.Series({t: realized.loc[t, en.loc[t].nlargest(3).index].mean()
                        - realized.loc[t, en.loc[t].nsmallest(3).index].mean() for t in dates})
    for name, part in (("dev", dev), ("hold", hold)):
        s = summarize_series(spread.loc[part])
        print(f"Top-3 minus bottom-3 sectors, {name}: {s['mean']:+.4f}/month (NW t {s['t_stat_nw']:+.2f})")

    stocks = list(cfg["universe"])
    g, tau = cfg["risk_tolerance"]["medium"], cfg["black_litterman"]["tau"]
    stock_fc = pd.read_parquet(Path(cfg["phase3"]["forecast_dir"]) / "forecasts_elastic_net.parquet")
    strategies = {"1/N": equal_weight(stocks),
                  "BL prior only": black_litterman_strategy(stock_fc, None, stocks, cfg, g, "A",
                                                            tau, views=False),
                  "BL + sector views (rotation)": sector_bl_strategy(en, sret, stocks, cfg, g, tau)}
    split = data["returns_monthly"].index[data["returns_monthly"].index > dev_end][0]
    rf = data["rf_monthly"]["rf"]
    rows = {}
    for name, strat in strategies.items():
        res = run_backtest(name, strat, data, dates, cfg["optimization"]["weight_tol"])
        net = res.net(bt["cost_bps"])
        row = {"avg_turnover": float(res.turnover.iloc[1:].mean())}
        for label, seg in (("dev", net.loc[:split]), ("hold", net.loc[split:].iloc[1:])):
            row[f"{label}_sharpe"] = performance_summary(seg, rf)["sharpe"]
            row[f"{label}_max_dd"] = max_drawdown(seg)
        rows[name] = row
    table = pd.DataFrame(rows).T
    table.to_csv(OUT / "sector_rotation_portfolios.csv")
    print("\nPortfolios on the 51 stocks (γ = 5, 10 bps; holdout part is post-holdout):")
    print(table.round(3).to_string())


if __name__ == "__main__":
    main()

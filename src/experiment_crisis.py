"""EXPERIMENT (branch experiment/model-search): 2001–2009 data for the 2008 crisis stress
test (DECISIONS MS10). Stored in ``data/crisis/``; the main data are untouched.

Same tables and cleaning rules as Phase 0 for the original 51 stocks, keeping those
with a complete history from 2001 (some, e.g. GOOGL, listed later and drop out).

Run:  python -m src.experiment_crisis
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from src.data import (daily_returns, download_raw, load_config, month_end_prices,
                      monthly_returns, risk_free_rates)
from src.experiment_universe import complete_history

logger = logging.getLogger(__name__)
CRISIS = Path("data/crisis")
START, END = "2001-01-01", "2009-12-31"


def build_crisis(cfg: dict) -> None:
    d = cfg["data"]
    stocks = list(cfg["universe"])
    bench, rf_t = d["benchmark"], d["risk_free"]
    raw = download_raw(stocks + [bench, rf_t], START, END, CRISIS / "raw")
    raw = raw[(raw.index.get_level_values("Date") >= START)
              & (raw.index.get_level_values("Date") <= END)]
    adj = raw["Adj Close"].unstack("Ticker")
    cal = pd.DatetimeIndex(adj[bench].dropna().index, name="Date")
    keep = complete_history(adj[stocks], cal, d["max_ffill_days"], d["max_missing_frac"])
    cols = keep + [bench]
    prices = adj[cols].reindex(cal).where(lambda x: x > 0).ffill(limit=d["max_ffill_days"])
    close = raw["Close"].unstack("Ticker")[cols].reindex(cal).ffill(limit=d["max_ffill_days"])
    volume = raw["Volume"].unstack("Ticker")[cols].reindex(cal).fillna(0.0)
    rf_d, rf_m = risk_free_rates(adj[rf_t], cal, d["max_ffill_days"], d["trading_days_per_year"])
    pm = month_end_prices(prices)
    out = {"prices": prices, "close": close, "volume": volume,
           "returns_daily": daily_returns(prices), "prices_monthly": pm,
           "returns_monthly": monthly_returns(pm), "rf_daily": rf_d, "rf_monthly": rf_m.to_frame()}
    pdir = CRISIS / "processed"
    pdir.mkdir(parents=True, exist_ok=True)
    for k, v in out.items():
        v.to_parquet(pdir / f"{k}.parquet")
    pd.Series(keep, name="ticker").to_csv(pdir / "kept.csv", index=False)
    logger.info("Crisis data: %d of %d stocks kept; dropped: %s", len(keep), len(stocks),
                sorted(set(stocks) - set(keep)))


def load_crisis(cfg: dict) -> dict:
    pdir = CRISIS / "processed"
    names = ["prices", "close", "volume", "returns_daily", "prices_monthly", "returns_monthly",
             "rf_daily", "rf_monthly"]
    out = {k: pd.read_parquet(pdir / f"{k}.parquet") for k in names}
    keep = pd.read_csv(pdir / "kept.csv")["ticker"].tolist()
    main_sec = pd.read_csv(Path(cfg["data"]["processed_dir"]) / "sectors.csv", index_col="ticker")
    main_sh = pd.read_csv(Path(cfg["data"]["processed_dir"]) / "shares_outstanding.csv",
                          index_col="ticker")
    out["sectors"] = main_sec.loc[keep]
    out["shares_outstanding"] = main_sh.loc[keep]
    out["stocks"] = keep
    return out


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    build_crisis(load_config())

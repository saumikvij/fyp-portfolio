"""EXPERIMENT (branch experiment/model-search): wide universe (~1,500 stocks), DECISIONS WU1.

Builds a processed dataset in ``data/wide/processed`` with the same tables and cleaning
rules as Phase 0 (src/data.py), but keeping only the stocks with a complete history
instead of raising an error for the others. The main 51-stock data are not touched.

Run:  python -m src.experiment_universe
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from src.data import (daily_returns, download_raw, load_config, load_processed,
                      month_end_prices, monthly_returns)

logger = logging.getLogger(__name__)
WIDE = Path("data/wide")


def complete_history(prices: pd.DataFrame, calendar: pd.DatetimeIndex, max_ffill: int,
                     max_missing: float) -> list[str]:
    """Tickers that pass the Phase 0 rules: priced on the first day, few missing days,
    no gap longer than ``max_ffill``."""
    px = prices.reindex(calendar)
    px = px.where(px > 0)
    ok = []
    for t in px.columns:
        s = px[t]
        if pd.isna(s.iloc[0]) or s.isna().mean() > max_missing:
            continue
        if s.ffill(limit=max_ffill).isna().any():
            continue
        ok.append(t)
    return ok


def build_wide(cfg: dict, force: bool = False) -> dict:
    d = cfg["data"]
    cand = pd.read_csv(WIDE / "universe_candidates.csv")
    bench = d["benchmark"]
    raw = download_raw(cand["ticker"].tolist() + [bench], d["start"], d["end"], WIDE / "raw",
                       force)
    raw = raw[(raw.index.get_level_values("Date") >= d["start"])
              & (raw.index.get_level_values("Date") <= d["end"])]
    adj = raw["Adj Close"].unstack("Ticker")
    calendar = pd.DatetimeIndex(adj[bench].dropna().index, name="Date")
    keep = complete_history(adj.drop(columns=[bench]), calendar, d["max_ffill_days"],
                            d["max_missing_frac"])
    cols = keep + [bench]
    prices = adj[cols].reindex(calendar).where(lambda x: x > 0).ffill(limit=d["max_ffill_days"])
    close = raw["Close"].unstack("Ticker")[cols].reindex(calendar).ffill(
        limit=d["max_ffill_days"])
    volume = raw["Volume"].unstack("Ticker")[cols].reindex(calendar).fillna(0.0)
    pm = month_end_prices(prices)
    main = load_processed(cfg)                      # same risk-free series as the main data
    out = {"prices": prices, "close": close, "volume": volume,
           "returns_daily": daily_returns(prices), "prices_monthly": pm,
           "returns_monthly": monthly_returns(pm), "rf_daily": main["rf_daily"],
           "rf_monthly": main["rf_monthly"]}
    pdir = WIDE / "processed"
    pdir.mkdir(parents=True, exist_ok=True)
    for k, v in out.items():
        v.to_parquet(pdir / f"{k}.parquet")
    uni = cand.set_index("ticker").loc[keep]
    uni.to_csv(pdir / "universe.csv")
    out["sectors"] = uni[["sector"]]
    logger.info("Wide universe: %d of %d candidates kept (complete history)", len(keep), len(cand))
    logger.info("By index: %s", uni["index"].value_counts().to_dict())
    return out


def load_wide() -> dict:
    pdir = WIDE / "processed"
    names = ["prices", "close", "volume", "returns_daily", "prices_monthly", "returns_monthly",
             "rf_daily", "rf_monthly"]
    out = {k: pd.read_parquet(pdir / f"{k}.parquet") for k in names}
    uni = pd.read_csv(pdir / "universe.csv", index_col="ticker")
    out["sectors"] = uni[["sector"]]
    out["universe"] = uni
    return out


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    build_wide(load_config())


def wide_shares(tickers: list[str], force: bool = False) -> pd.DataFrame:
    """All-class share counts for the wide universe (impliedSharesOutstanding, falling
    back to sharesOutstanding); tickers with neither are left out (WU4)."""
    import yfinance as yf

    path = WIDE / "processed" / "shares_outstanding.csv"
    if path.exists() and not force:
        return pd.read_csv(path, index_col="ticker")
    rows = []
    for i, t in enumerate(tickers, 1):
        try:
            info = yf.Ticker(t).info
            n = info.get("impliedSharesOutstanding") or info.get("sharesOutstanding")
        except Exception:                                   # network / missing quote
            n = None
        if n:
            rows.append({"ticker": t, "shares_outstanding": int(n)})
        if i % 100 == 0:
            logger.info("shares: %d / %d", i, len(tickers))
    out = pd.DataFrame(rows).set_index("ticker")
    out.to_csv(path)
    logger.info("Share counts for %d of %d stocks", len(out), len(tickers))
    return out

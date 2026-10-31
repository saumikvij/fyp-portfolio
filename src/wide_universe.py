"""Wider stock dataset for the survivorship / universe-selection check (DECISIONS R2).

The S&P 1500 members listed in ``resources/sp1500_candidates.csv`` (taken from Wikipedia's
constituent lists) are downloaded and cleaned with the same Phase 0 rules as the main
51-stock data, keeping only stocks with a complete price history since the start date
instead of raising an error for the others. The result (about 1,050 stocks) is written
to ``data/wide/processed``; the main data are not touched.

Run:  python -m src.wide_universe        (download, clean, fetch share counts)

Limitation: the candidates are *today's* index members, so this dataset carries the same
survivorship bias as the main universe; it is used to test sensitivity to the choice of
stocks, not to remove the bias.
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path

import pandas as pd

from src.data import (daily_returns, download_raw, load_config, load_processed,
                      month_end_prices, monthly_returns)

logger = logging.getLogger(__name__)
WIDE = Path("data/wide")
CANDIDATES = Path("resources/sp1500_candidates.csv")


def complete_history(prices: pd.DataFrame, calendar: pd.DatetimeIndex, max_ffill: int,
                     max_missing: float) -> list[str]:
    """Tickers that pass the Phase 0 cleaning rules.

    Parameters
    ----------
    prices : adjusted prices (date x ticker).
    calendar : trading days (the benchmark's).
    max_ffill : longest gap, in days, that may be forward-filled.
    max_missing : largest allowed share of missing days.

    Returns
    -------
    Tickers priced on the first day, with at most ``max_missing`` missing days and no gap
    longer than ``max_ffill``.

    Theorem: none.
    """
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
    """Download and clean the candidates; write ``data/wide/processed``.

    Parameters
    ----------
    cfg : config (``data`` section: dates, benchmark, cleaning rules).
    force : re-download even if the raw file exists.

    Returns
    -------
    dict with the same tables as ``src.data.load_processed`` (risk-free series reused from
    the main data) plus ``sectors``.

    Theorem: none.
    """
    d = cfg["data"]
    WIDE.mkdir(parents=True, exist_ok=True)
    shutil.copy(CANDIDATES, WIDE / "universe_candidates.csv")
    cand = pd.read_csv(CANDIDATES)
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
    main = load_processed(cfg)
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
    return out


def wide_shares(tickers: list[str], force: bool = False) -> pd.DataFrame:
    """All-class share counts (impliedSharesOutstanding, else sharesOutstanding), as in
    Phase 0; tickers with neither are left out.

    Parameters
    ----------
    tickers : stocks to look up.
    force : look up again even if the file exists.

    Returns
    -------
    DataFrame indexed by ticker with ``shares_outstanding``.

    Theorem: none.
    """
    import yfinance as yf

    path = WIDE / "processed" / "shares_outstanding.csv"
    if path.exists() and not force:
        return pd.read_csv(path, index_col="ticker")
    rows = []
    for t in tickers:
        try:
            info = yf.Ticker(t).info
            n = info.get("impliedSharesOutstanding") or info.get("sharesOutstanding")
        except Exception:                                   # network / missing quote
            n = None
        if n:
            rows.append({"ticker": t, "shares_outstanding": int(n)})
    out = pd.DataFrame(rows).set_index("ticker")
    out.to_csv(path)
    logger.info("Share counts for %d of %d stocks", len(out), len(tickers))
    return out


def load_wide() -> dict:
    """Load ``data/wide/processed`` in the format of ``src.data.load_processed``.

    Returns
    -------
    dict with prices, close, volume, returns, risk-free tables, ``sectors`` (ticker ->
    sector), ``universe`` (sector and index membership) and ``shares_outstanding``.

    Theorem: none.
    """
    pdir = WIDE / "processed"
    names = ["prices", "close", "volume", "returns_daily", "prices_monthly", "returns_monthly",
             "rf_daily", "rf_monthly"]
    out = {k: pd.read_parquet(pdir / f"{k}.parquet") for k in names}
    uni = pd.read_csv(pdir / "universe.csv", index_col="ticker")
    out["sectors"] = uni[["sector"]]
    out["universe"] = uni
    out["shares_outstanding"] = pd.read_csv(pdir / "shares_outstanding.csv", index_col="ticker")
    return out


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    data = build_wide(load_config())
    wide_shares(list(data["sectors"].index))

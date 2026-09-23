"""Download and clean prices, benchmark, risk-free rate and sectors (Phase 0).

Run ``python -m src.data`` to build ``data/processed``. Raw downloads are cached in
``data/raw``; pass ``--force`` to download again.

Conventions used by every later module:
- Daily tables are indexed by NYSE trading day (the dates on which SPY trades).
- A monthly row dated t (the last trading day of a month) holds only information
  available at the close of t. ``returns_monthly`` at t is the return over the month
  ending at t; ``rf_monthly`` at t is the risk-free return earned over that same month,
  set from the T-bill yield observed at the previous month-end.
"""

from __future__ import annotations

import argparse
import logging
from datetime import date
from pathlib import Path

import pandas as pd
import yaml

logger = logging.getLogger(__name__)

PROCESSED_FILES = {
    "prices": "prices.parquet",
    "volume": "volume.parquet",
    "returns_daily": "returns_daily.parquet",
    "prices_monthly": "prices_monthly.parquet",
    "returns_monthly": "returns_monthly.parquet",
    "rf_daily": "rf_daily.parquet",
    "rf_monthly": "rf_monthly.parquet",
}


def load_config(path: str | Path = "config.yaml") -> dict:
    """Read the YAML config.

    Parameters
    ----------
    path : path to ``config.yaml``.

    Returns
    -------
    dict with the sections ``data``, ``backtest``, ``universe``, etc.

    Theorem: none (setup).
    """
    with open(path) as f:
        return yaml.safe_load(f)


# --------------------------------------------------------------------------- download


def download_raw(tickers: list[str], start: str, end: str, raw_dir: Path,
                 force: bool = False) -> pd.DataFrame:
    """Download daily OHLCV and adjusted close from Yahoo Finance, with a cache.

    Parameters
    ----------
    tickers : Yahoo symbols (stocks, benchmark and ``^IRX``).
    start, end : first and last date (inclusive), ``YYYY-MM-DD``.
    raw_dir : folder for ``prices_raw.parquet``.
    force : download again even when the cache covers the request.

    Returns
    -------
    DataFrame indexed by (Date, Ticker) with columns Adj Close, Close, High, Low,
    Open, Volume. The cache is reused only if it covers all tickers and dates.

    Theorem: none (data).
    """
    import yfinance as yf

    path = raw_dir / "prices_raw.parquet"
    if path.exists() and not force:
        raw = pd.read_parquet(path)
        dates = raw.index.get_level_values("Date")
        if (set(tickers) <= set(raw.index.get_level_values("Ticker"))
                and dates.min() <= pd.Timestamp(start) + pd.Timedelta(days=7)
                and dates.max() >= pd.Timestamp(end) - pd.Timedelta(days=7)):
            logger.info("Using cached raw prices %s", path)
            return raw
        logger.info("Cached raw prices do not cover the request; downloading again")

    # yfinance treats `end` as exclusive.
    end_excl = (pd.Timestamp(end) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    wide = yf.download(tickers, start=start, end=end_excl, auto_adjust=False,
                       progress=False, threads=True)
    if wide.empty:
        raise RuntimeError("Yahoo Finance returned no data")
    raw = wide.stack(level="Ticker", future_stack=True).dropna(how="all")
    raw.index = raw.index.set_names(["Date", "Ticker"])
    raw_dir.mkdir(parents=True, exist_ok=True)
    raw.to_parquet(path)
    return raw


def download_shares_outstanding(tickers: list[str], raw_dir: Path,
                                force: bool = False) -> pd.DataFrame:
    """Current shares outstanding per ticker, with a cache.

    Used later to approximate market-cap weights w_mkt for the Black–Litterman prior
    Π = δΣw_mkt (plan Section 2.4). Using *current* share counts is a known
    approximation (plan Section 3).

    Yahoo's ``sharesOutstanding`` counts one share class only (e.g. GOOGL class A,
    about half of Alphabet), so the main count is ``impliedSharesOutstanding``, which
    covers all classes and matches Yahoo's market cap (see DECISIONS.md D0.9).

    Parameters
    ----------
    tickers : stock symbols.
    raw_dir : folder for ``shares_outstanding.csv``.
    force : download again even when cached.

    Returns
    -------
    DataFrame indexed by ticker with columns ``shares_outstanding`` (all share
    classes), ``shares_primary`` (Yahoo's single-class count, for reference) and
    ``as_of`` (download date).

    Theorem: none (data).
    """
    import yfinance as yf

    path = raw_dir / "shares_outstanding.csv"
    if path.exists() and not force:
        cached = pd.read_csv(path, index_col="ticker")
        if set(tickers) <= set(cached.index) and "shares_primary" in cached:
            return cached.loc[tickers]

    rows = []
    for t in tickers:
        info = yf.Ticker(t).info
        total, primary = info.get("impliedSharesOutstanding"), info.get("sharesOutstanding")
        if not total or not primary:
            raise RuntimeError(f"Missing share counts from Yahoo for {t}")
        rows.append({"ticker": t, "shares_outstanding": int(total),
                     "shares_primary": int(primary), "as_of": date.today().isoformat()})
    out = pd.DataFrame(rows).set_index("ticker")
    raw_dir.mkdir(parents=True, exist_ok=True)
    out.to_csv(path)
    return out


# --------------------------------------------------------------------------- cleaning


def clean_prices(prices: pd.DataFrame, calendar: pd.DatetimeIndex, max_ffill_days: int,
                 max_missing_frac: float) -> pd.DataFrame:
    """Align prices to the trading calendar and fill short gaps with the last price.

    Forward filling only uses past prices, so it cannot introduce look-ahead.

    Parameters
    ----------
    prices : wide DataFrame (date x ticker) of adjusted closes.
    calendar : trading days to align to (SPY's trading days).
    max_ffill_days : longest gap, in trading days, that may be forward filled.
    max_missing_frac : largest allowed share of missing days per ticker.

    Returns
    -------
    DataFrame indexed by ``calendar`` with no missing values.

    Raises
    ------
    ValueError if a ticker has no price on the first calendar day, misses more than
    ``max_missing_frac`` of days, or has a gap longer than ``max_ffill_days``.
    Non-positive prices count as missing.

    Theorem: none (data).
    """
    px = prices.reindex(calendar)
    px = px.where(px > 0)

    missing_frac = px.isna().mean()
    too_sparse = missing_frac[missing_frac > max_missing_frac]
    if not too_sparse.empty:
        raise ValueError(f"Too many missing prices: {too_sparse.round(4).to_dict()}")

    late = [t for t in px.columns if pd.isna(px[t].iloc[0])]
    if late:
        raise ValueError(f"No price on first trading day {calendar[0].date()}: {late}")

    px = px.ffill(limit=max_ffill_days)
    gaps = px.columns[px.isna().any()].tolist()
    if gaps:
        raise ValueError(f"Gaps longer than {max_ffill_days} trading days: {gaps}")
    return px


def daily_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Simple daily returns r_d = P_d / P_{d-1} - 1.

    Parameters
    ----------
    prices : wide DataFrame (date x ticker) with no missing values.

    Returns
    -------
    DataFrame of the same shape minus the first row; row d uses only prices at d
    and the previous trading day.

    Theorem: none (data).
    """
    return prices.pct_change(fill_method=None).iloc[1:]


def flag_extreme_returns(returns: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """List daily returns large enough to deserve a manual check.

    Parameters
    ----------
    returns : wide DataFrame (date x ticker) of daily returns.
    threshold : absolute return above which a row is flagged (e.g. 0.40).

    Returns
    -------
    DataFrame with columns ``Date``, ``Ticker``, ``ret``.

    Theorem: none (data).
    """
    stacked = returns.stack()
    extreme = stacked[stacked.abs() > threshold]
    return extreme.rename("ret").reset_index().rename(
        columns={"level_0": "Date", "level_1": "Ticker"})


def month_end_dates(calendar: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Last trading day of each calendar month.

    Parameters
    ----------
    calendar : trading days.

    Returns
    -------
    DatetimeIndex with one date per month present in ``calendar``. These are the
    rebalancing dates t of the monthly walk-forward backtest.

    Theorem: none (data).
    """
    s = calendar.to_series()
    return pd.DatetimeIndex(s.groupby(calendar.to_period("M")).max().values)


def month_end_prices(prices: pd.DataFrame) -> pd.DataFrame:
    """Prices on the last trading day of each month.

    Parameters
    ----------
    prices : wide daily DataFrame (date x ticker).

    Returns
    -------
    DataFrame with the rows of ``prices`` at ``month_end_dates(prices.index)``.

    Theorem: none (data).
    """
    return prices.loc[month_end_dates(prices.index)]


def monthly_returns(monthly_prices: pd.DataFrame) -> pd.DataFrame:
    """Simple monthly returns between consecutive month-ends.

    Parameters
    ----------
    monthly_prices : output of ``month_end_prices``.

    Returns
    -------
    DataFrame whose row t is the return from the previous month-end to month-end t
    (the first month is dropped because it has no previous month-end).

    Theorem: none (data).
    """
    return monthly_prices.pct_change(fill_method=None).iloc[1:]


def risk_free_rates(irx: pd.Series, calendar: pd.DatetimeIndex, max_ffill_days: int,
                    days_per_year: int) -> tuple[pd.DataFrame, pd.Series]:
    """Convert the ^IRX yield (annualized percent) to daily and monthly rates.

    Rates are compounded: r_daily = (1+y)^(1/252) - 1, r_monthly = (1+y)^(1/12) - 1.

    Parameters
    ----------
    irx : daily ^IRX closes in percent (e.g. 4.0 for 4%).
    calendar : trading days to align to; short gaps are forward filled.
    max_ffill_days : longest gap that may be forward filled.
    days_per_year : trading days per year for the daily rate.

    Returns
    -------
    rf_daily : DataFrame with ``annual`` (decimal yield observed at the close of d)
        and ``daily`` (return earned on day d, from the previous day's yield).
    rf_monthly : Series ``rf``; row t is the return earned over the month ending t,
        from the yield observed at the previous month-end (known when the month
        starts, so no look-ahead).

    Theorem: none (data).
    """
    annual = irx.reindex(calendar).ffill(limit=max_ffill_days) / 100.0
    if annual.isna().any():
        raise ValueError("Risk-free series has gaps that cannot be forward filled")

    daily = (1 + annual.shift(1)) ** (1 / days_per_year) - 1
    rf_daily = pd.DataFrame({"annual": annual, "daily": daily}).iloc[1:]

    annual_me = annual.loc[month_end_dates(calendar)]
    rf_monthly = ((1 + annual_me.shift(1)) ** (1 / 12) - 1).iloc[1:].rename("rf")
    return rf_daily, rf_monthly


# --------------------------------------------------------------------------- pipeline


def build_dataset(cfg: dict, force_download: bool = False) -> dict[str, pd.DataFrame]:
    """Run the full Phase 0 pipeline: download, clean, and save processed data.

    Parameters
    ----------
    cfg : config dict from ``load_config``.
    force_download : ignore cached raw downloads.

    Returns
    -------
    dict of DataFrames, also written to ``processed_dir``: ``prices``, ``volume``,
    ``returns_daily``, ``prices_monthly``, ``returns_monthly``, ``rf_daily``,
    ``rf_monthly``, ``sectors``, ``shares_outstanding``. Extreme daily returns are
    logged and saved to ``extreme_returns.csv``.

    Theorem: none (data).
    """
    d = cfg["data"]
    raw_dir, out_dir = Path(d["raw_dir"]), Path(d["processed_dir"])
    universe = cfg["universe"]
    stocks = list(universe)
    bench, rf_ticker = d["benchmark"], d["risk_free"]

    raw = download_raw(stocks + [bench, rf_ticker], d["start"], d["end"], raw_dir,
                       force_download)
    raw = raw[(raw.index.get_level_values("Date") >= d["start"])
              & (raw.index.get_level_values("Date") <= d["end"])]
    adj = raw["Adj Close"].unstack("Ticker")
    vol = raw["Volume"].unstack("Ticker")

    calendar = pd.DatetimeIndex(adj[bench].dropna().index, name="Date")
    price_cols = stocks + [bench]
    prices = clean_prices(adj[price_cols], calendar, d["max_ffill_days"],
                          d["max_missing_frac"])
    # A day whose price was forward filled had no trading: volume 0.
    volume = vol[price_cols].reindex(calendar).fillna(0.0)

    rets = daily_returns(prices)
    extreme = flag_extreme_returns(rets, d["extreme_daily_return"])
    for row in extreme.itertuples(index=False):
        logger.warning("Extreme daily return %s %s: %.1f%%", row.Date.date(), row.Ticker,
                       100 * row.ret)

    rf_daily, rf_monthly = risk_free_rates(adj[rf_ticker], calendar, d["max_ffill_days"],
                                           d["trading_days_per_year"])
    pm = month_end_prices(prices)

    out = {
        "prices": prices,
        "volume": volume,
        "returns_daily": rets,
        "prices_monthly": pm,
        "returns_monthly": monthly_returns(pm),
        "rf_daily": rf_daily,
        "rf_monthly": rf_monthly.to_frame(),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    for key, df in out.items():
        df.to_parquet(out_dir / PROCESSED_FILES[key])

    sectors = pd.Series(universe, name="sector").rename_axis("ticker")
    sectors.to_csv(out_dir / "sectors.csv")
    shares = download_shares_outstanding(stocks, raw_dir, force_download)
    shares.to_csv(out_dir / "shares_outstanding.csv")
    extreme.to_csv(out_dir / "extreme_returns.csv", index=False)
    out.update(sectors=sectors.to_frame(), shares_outstanding=shares)
    return out


def load_processed(cfg: dict) -> dict[str, pd.DataFrame]:
    """Load everything written by ``build_dataset``.

    Parameters
    ----------
    cfg : config dict from ``load_config``.

    Returns
    -------
    dict with the same keys as ``build_dataset``'s output.

    Theorem: none (data).
    """
    out_dir = Path(cfg["data"]["processed_dir"])
    data = {k: pd.read_parquet(out_dir / f) for k, f in PROCESSED_FILES.items()}
    data["sectors"] = pd.read_csv(out_dir / "sectors.csv", index_col="ticker")
    data["shares_outstanding"] = pd.read_csv(out_dir / "shares_outstanding.csv",
                                             index_col="ticker")
    return data


def main() -> None:
    """Command-line entry point: ``python -m src.data [--config PATH] [--force]``."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--force", action="store_true", help="download again")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    data = build_dataset(load_config(args.config), args.force)
    p, rm = data["prices"], data["returns_monthly"]
    logger.info("Daily prices: %d days x %d series, %s to %s", *p.shape,
                p.index[0].date(), p.index[-1].date())
    logger.info("Monthly returns: %d months, %s to %s", len(rm), rm.index[0].date(),
                rm.index[-1].date())


if __name__ == "__main__":
    main()

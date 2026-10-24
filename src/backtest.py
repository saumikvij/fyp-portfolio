"""Monthly walk-forward backtest with transaction costs (plan Section 6).

At each rebalance date t (a month-end) a strategy sees only data dated <= t and
returns target weights; the portfolio is held over the next month s and its
realized return recorded. Look-ahead is prevented structurally: the engine slices
every table to ``.loc[:t]`` before calling the strategy.

Conventions
-----------
- Weights are decided at t and indexed by t; returns, turnover and costs are
  indexed by the holding month-end s (the month after t), matching ``returns_monthly``.
- Turnover at t is Σᵢ|wᵢ(target) − wᵢ(drifted)|, where the drifted weights are last
  month's weights after a month of price moves. The first trade from cash counts as
  turnover 1.
- Costs: net return = (1 − c·turnover)(1 + gross) − 1, with c = cost_bps / 10 000,
  paid at the start of the month.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd

from src.black_litterman import bl_inputs
from src.estimators import historical_mean, sample_covariance
from src.optimization import Constraints, mean_variance, min_variance, rule_sector_bounds


@dataclass
class History:
    """All data a strategy may use at rebalance date ``t`` (every table sliced to <= t)."""

    t: pd.Timestamp
    returns_daily: pd.DataFrame
    returns_monthly: pd.DataFrame
    rf_monthly: pd.Series
    sectors: pd.Series
    close: pd.DataFrame | None = None      # unadjusted close (for market caps), <= t
    shares: pd.Series | None = None        # shares outstanding snapshot (static)


Strategy = Callable[[History], pd.Series]


def history_at(data: dict, t: pd.Timestamp) -> History:
    """Slice the processed data to what is known at the close of ``t``.

    Parameters
    ----------
    data : output of ``src.data.load_processed``.
    t : rebalance date.

    Returns
    -------
    History whose tables contain no rows dated after t.

    Theorem: none (enforces plan Section 6.1, no look-ahead).
    """
    return History(
        t=t,
        returns_daily=data["returns_daily"].loc[:t],
        returns_monthly=data["returns_monthly"].loc[:t],
        rf_monthly=data["rf_monthly"]["rf"].loc[:t],
        sectors=data["sectors"]["sector"],
        close=data["close"].loc[:t] if "close" in data else None,
        shares=(data["shares_outstanding"]["shares_outstanding"]
                if "shares_outstanding" in data else None),
    )


def rebalance_dates(month_ends: pd.DatetimeIndex, oos_start: str) -> pd.DatetimeIndex:
    """Month-ends at which the portfolio is chosen.

    The first is the last month-end before ``oos_start`` (so the first month held is
    the first out-of-sample month); the last is the second-to-last month-end, since
    the final month is needed to measure the return.

    Parameters
    ----------
    month_ends : all month-end dates in the data (index of ``returns_monthly``).
    oos_start : first out-of-sample month, e.g. "2016-01-01".

    Returns
    -------
    DatetimeIndex of rebalance dates t.

    Theorem: none.
    """
    start = pd.Timestamp(oos_start)
    before = month_ends[month_ends < start]
    if before.empty:
        raise ValueError(f"No month-end before {oos_start}")
    return month_ends[(month_ends >= before[-1]) & (month_ends < month_ends[-1])]


def clean_weights(w: pd.Series, tol: float) -> pd.Series:
    """Set |wᵢ| < tol to zero and rescale to sum to 1.

    Interior-point solutions carry ~1e-9 dust on assets at a bound (D1.5); holding
    them would only create spurious turnover.

    Parameters
    ----------
    w : weights summing to 1.
    tol : threshold below which a weight is treated as zero.

    Returns
    -------
    Cleaned weights summing to 1.

    Theorem: none.
    """
    w = w.where(w.abs() >= tol, 0.0)
    return w / w.sum()


@dataclass
class BacktestResult:
    """Output of ``run_backtest``; costs are applied afterwards with ``net``."""

    name: str
    weights: pd.DataFrame     # target weights, index = rebalance date t
    gross: pd.Series          # monthly gross return, index = holding month-end s
    turnover: pd.Series       # turnover traded at the start of month s
    daily_gross: pd.Series    # daily gross return over the holding months
    first_day: pd.Series      # bool, True on the first trading day of each holding month

    def net(self, cost_bps: float) -> pd.Series:
        """Monthly net returns: (1 − c·turnover)(1 + gross) − 1, c = cost_bps / 10 000."""
        c = cost_bps / 1e4
        return (1 - c * self.turnover) * (1 + self.gross) - 1

    def daily_net(self, cost_bps: float) -> pd.Series:
        """Daily net returns; each month's cost is charged on its first trading day."""
        c = cost_bps / 1e4
        month = self.daily_gross.index.to_period("M")
        to = self.turnover.copy()
        to.index = to.index.to_period("M")
        charge = np.where(self.first_day.values, c * to.reindex(month).values, 0.0)
        return pd.Series((1 - charge) * (1 + self.daily_gross.values) - 1,
                         index=self.daily_gross.index)


def run_backtest(name: str, strategy: Strategy, data: dict, dates: pd.DatetimeIndex,
                 weight_tol: float, lag_days: int = 0) -> BacktestResult:
    """Run a strategy through the monthly walk-forward loop.

    Parameters
    ----------
    name : label for the result.
    strategy : function History -> target weights (Series over any columns of
        ``returns_monthly``; missing assets get weight 0). Weights must sum to 1.
    data : output of ``src.data.load_processed``.
    dates : rebalance dates t (see ``rebalance_dates``).
    weight_tol : dust threshold for ``clean_weights``.
    lag_days : trading days between the decision (close of t) and the trade. 0 = trade
        at the close of t, the standard assumption; 1 = trade at the next close, a
        robustness check. With a lag the holding window shifts by the same number of
        days (the last window is cut at the end of the data).

    Returns
    -------
    BacktestResult with weights, gross returns, turnover and daily gross returns.
    Monthly gross returns are compounded from the daily path, which for lag 0 equals
    ``returns_monthly`` (tested).

    Raises
    ------
    ValueError if a strategy returns NaN weights or weights not summing to 1.

    Theorem: none (plan Section 6, rules 1, 2 and 5).
    """
    rm, rd = data["returns_monthly"], data["returns_daily"]
    assets = rm.columns
    month_ends, day_idx = rm.index, rd.index
    w_rows, gross, turnover, daily, first = {}, {}, {}, [], []
    drifted = pd.Series(0.0, index=assets)   # start in cash

    for t in dates:
        s = month_ends[month_ends > t][0]
        w = strategy(history_at(data, t)).reindex(assets).fillna(0.0)
        if w.isna().any() or abs(w.sum() - 1) > 1e-6:
            raise ValueError(f"{name}: invalid weights at {t.date()} (sum {w.sum():.6f})")
        w = clean_weights(w, weight_tol)

        # Buy-and-hold over the holding window (trade day, trade day of the next month].
        start = day_idx.get_loc(t) + lag_days
        end = min(day_idx.get_loc(s) + lag_days, len(day_idx) - 1)
        days = rd.iloc[start + 1:end + 1][assets]
        value = (1 + days).cumprod() @ w
        g = float(value.iloc[-1] - 1)
        r = (1 + days).prod() - 1
        w_rows[t] = w
        gross[s] = g
        turnover[s] = float((w - drifted).abs().sum())
        drifted = w * (1 + r) / (1 + g)
        daily.append(value / value.shift(1, fill_value=1.0) - 1)
        first.append(pd.Series([True] + [False] * (len(days) - 1), index=days.index))

    return BacktestResult(
        name=name,
        weights=pd.DataFrame(w_rows).T.rename_axis("t"),
        gross=pd.Series(gross, name=name).rename_axis("s"),
        turnover=pd.Series(turnover, name=name).rename_axis("s"),
        daily_gross=pd.concat(daily).rename(name),
        first_day=pd.concat(first),
    )


# --------------------------------------------------------------------------- strategies


def equal_weight(stocks: list[str]) -> Strategy:
    """S1: 1/N in every stock, rebalanced monthly.

    Parameters
    ----------
    stocks : the universe.

    Returns
    -------
    Strategy function.

    Theorem: none (benchmark, DeMiguel et al. 2009).
    """
    w = pd.Series(1.0 / len(stocks), index=stocks)
    return lambda hist: w


def buy_and_hold(ticker: str) -> Strategy:
    """S2: 100% in one asset (SPY), so no trading after the first month.

    Parameters
    ----------
    ticker : asset held.

    Returns
    -------
    Strategy function.

    Theorem: none (benchmark).
    """
    w = pd.Series({ticker: 1.0})
    return lambda hist: w


def _sigma(hist: History, stocks: list[str], cfg: dict) -> pd.DataFrame:
    return sample_covariance(hist.returns_daily[stocks], hist.t,
                             cfg["backtest"]["cov_window_days"],
                             cfg["data"]["trading_days_per_year"] / 12)


def strategy_constraints(hist: History, cfg: dict) -> Constraints:
    """Constraints of every optimized strategy (S3–S6) and of the recommendation tool.

    Long-only, at most ``backtest.w_max`` per stock, and the sector rule (A4): no
    sector above ``optimization.sector_max`` of the portfolio. One function for the
    backtest and the tool, so the tool always runs the strategy that was evaluated.

    Parameters
    ----------
    hist : History at t (its ``sectors`` are used).
    cfg : config.

    Returns
    -------
    Constraints for ``mean_variance`` / ``min_variance``.

    Theorem: T1, T4 (the feasible set of the main problem).
    """
    return Constraints(long_only=True, w_max=cfg["backtest"]["w_max"], sectors=hist.sectors,
                       sector_bounds=rule_sector_bounds(hist.sectors,
                                                        cfg["optimization"].get("sector_max")))


def min_variance_strategy(stocks: list[str], cfg: dict) -> Strategy:
    """S3: minimum-variance portfolio (long-only, capped) from the trailing Σ.

    Parameters
    ----------
    stocks : the universe.
    cfg : config (covariance window, cap, sector bounds, solver).

    Returns
    -------
    Strategy function.

    Theorem: T1, T4 (the QP solved each month).
    """
    def strategy(hist: History) -> pd.Series:
        Sigma = _sigma(hist, stocks, cfg)
        return min_variance(Sigma, strategy_constraints(hist, cfg),
                            cfg["optimization"]["solver"]).weights
    return strategy


def markowitz_strategy(stocks: list[str], cfg: dict, gamma: float) -> Strategy:
    """S4: classical mean-variance with historical-mean μ and trailing Σ.

    Parameters
    ----------
    stocks : the universe.
    cfg : config (windows, cap, sector bounds, solver).
    gamma : risk aversion γ.

    Returns
    -------
    Strategy function.

    Theorem: T1, T4 (the QP solved each month); T5 (its instability is measured).
    """
    def strategy(hist: History) -> pd.Series:
        Sigma = _sigma(hist, stocks, cfg)
        mu = historical_mean(hist.returns_monthly[stocks], hist.t,
                             cfg["estimators"]["mean_window_months"])
        return mean_variance(mu, Sigma, gamma, strategy_constraints(hist, cfg),
                             cfg["optimization"]["solver"]).weights
    return strategy


def forecast_strategy(forecasts: pd.DataFrame, stocks: list[str], cfg: dict,
                      gamma: float) -> Strategy:
    """S5: mean-variance with ML forecasts of next-month excess returns as μ (no BL).

    ``forecasts.loc[t]`` must have been produced from data dated <= t (guaranteed by
    ``src.models.walk_forward`` and its look-ahead tests); the strategy reads only
    the row for the current rebalance date.

    Parameters
    ----------
    forecasts : forecast date x ticker, from ``walk_forward``.
    stocks : the universe.
    cfg : config (windows, cap, sector bounds, solver).
    gamma : risk aversion γ.

    Returns
    -------
    Strategy function.

    Theorem: T1, T4 (the QP solved each month).
    """
    def strategy(hist: History) -> pd.Series:
        mu = forecasts.loc[hist.t, stocks]
        return mean_variance(mu, _sigma(hist, stocks, cfg), gamma, strategy_constraints(hist, cfg),
                             cfg["optimization"]["solver"]).weights
    return strategy


def black_litterman_strategy(forecasts: pd.DataFrame, ensemble_var: pd.DataFrame | None,
                             stocks: list[str], cfg: dict, gamma: float, omega_kind: str,
                             tau: float, omega_scale: float = 1.0,
                             views: bool = True,
                             vol_forecasts: pd.DataFrame | None = None) -> Strategy:
    """S6: mean-variance with the Black–Litterman posterior mean μ_BL as μ.

    At each t: Σ from the trailing window, market prior Π = δΣw_mkt, views Q = ML
    forecasts made at t, Ω from past errors (Ω-A) or ensemble disagreement (Ω-B),
    μ_BL by T6, then the same constrained problem as S4 and S5.

    Parameters
    ----------
    forecasts : ML forecasts (date x ticker), produced from data <= each date.
    ensemble_var : bootstrap-copy variance (date x ticker), for Ω-B.
    stocks : the universe.
    cfg : config.
    gamma : risk aversion γ.
    omega_kind : "A", "B" or "B-cal" (see ``src.black_litterman.bl_inputs``).
    tau : τ.
    omega_scale : multiplies Ω (ablation).
    views : False uses μ = Π (the prior alone, the Ω → ∞ limit of Cor. 6.1).
    vol_forecasts : ML volatility forecasts; if given, Σ = D·R·D (``src.risk_model``, D6)
        is used for the prior, the posterior and the optimization.

    Returns
    -------
    Strategy function.

    Theorem: T6 (posterior), Cor. 6.1 (views=False), T1/T4 (the QP).
    """
    def strategy(hist: History) -> pd.Series:
        inp = bl_inputs(hist, forecasts, ensemble_var, stocks, cfg, omega_kind, tau,
                        omega_scale, vol_forecasts)
        mu = inp["mu_bl"] if views else inp["Pi"]
        return mean_variance(mu, inp["Sigma"], gamma, strategy_constraints(hist, cfg),
                             cfg["optimization"]["solver"]).weights
    return strategy

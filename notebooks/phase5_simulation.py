"""Phase 5: month-by-month simulation of the recommendation tool, Jan 2016 – Aug 2026.

Run from the repo root:  python -m notebooks.phase5_simulation

Start with ``amount`` dollars at the first decision date. At every month-end t the tool
recommends a whole-share portfolio from data up to t only (same tool, same model); the
portfolio is held for the next month at real (dividend-adjusted) prices; uninvested
cash earns the T-bill rate; trading costs ``backtest.cost_bps`` on the dollars traded.
The next month starts from the money actually there. Compared with buying SPY once.

Writes ``phase5.table_dir``/phase5_simulation.csv (monthly values) and
``phase5.figure_dir``/simulation_wealth.png.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.data import load_config, load_processed
from src.recommend import UserRequest, forecasts_up_to, recommend

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
INK, INK_2, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#a3a29c", "#e4e3df", "#fcfcfb"


def simulate(risk: str, amount: float, data: dict, cfg: dict, dates, forecasts) -> pd.Series:
    """Money in the account at each month-end when following the tool every month."""
    px, rf = data["prices"], data["rf_monthly"]["rf"]
    me = data["returns_monthly"].index
    cost = cfg["backtest"]["cost_bps"] / 1e4
    wealth = {dates[0]: amount}
    held = pd.Series(dtype=float)                      # $ per stock before rebalancing
    money = amount
    for t in dates:
        s = me[me > t][0]
        rec = recommend(UserRequest(money, risk), cfg, data, as_of=t, forecasts=forecasts)
        new = rec.holdings["value"]
        traded = new.subtract(held, fill_value=0.0).abs().sum()
        cash = money - new.sum() - cost * traded
        growth = px.loc[s, new.index] / px.loc[t, new.index]
        held = new * growth                            # value after a month of price moves
        money = float(held.sum()) + cash * (1 + rf.loc[s])
        wealth[s] = money
    return pd.Series(wealth)


def main() -> None:
    cfg = load_config()
    tab_dir, fig_dir = Path(cfg["phase5"]["table_dir"]), Path(cfg["phase5"]["figure_dir"])
    data = load_processed(cfg)
    me = data["returns_monthly"].index
    first = me[me < pd.Timestamp(cfg["backtest"]["oos_start"])][-1]
    dates = me[(me >= first) & (me < me[-1])]
    amount = cfg["recommend"]["examples"][0]["amount"]
    # One model run up to the last decision date; each month reads only its own row
    # and earlier errors (tested: forecasts match the Phase 3 walk-forward).
    fc = forecasts_up_to(data, cfg, dates[-1])

    out = {f"Bot, {r} risk": simulate(r, amount, data, cfg, dates, fc)
           for r in cfg["risk_tolerance"]}
    spy = data["prices"]["SPY"]
    out["Market (SPY, buy once)"] = amount * spy.loc[out["Bot, low risk"].index] / spy.loc[dates[0]]
    table = pd.DataFrame(out)
    table.to_csv(tab_dir / "phase5_simulation.csv")

    years = (table.index[-1] - table.index[0]).days / 365.25
    monthly = table.pct_change().dropna()
    print(f"${amount:,.0f} invested on {table.index[0]:%d %b %Y}, value on {table.index[-1]:%d %b %Y}:")
    for c in table:
        r = monthly[c]
        worst = (table[c] / table[c].cummax() - 1).min()
        print(f"  {c:24s} ${table[c].iloc[-1]:>9,.0f}   {(table[c].iloc[-1] / amount) ** (1 / years) - 1:5.1%} a year"
              f"   volatility {r.std() * np.sqrt(12):5.1%}   worst fall {worst:6.1%}"
              f"   beat SPY in {100 * (r > monthly['Market (SPY, buy once)']).mean():3.0f}% of months")

    fig, ax = plt.subplots(figsize=(7.5, 4.6), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    ax.grid(color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=INK_2, labelsize=9)
    ends = []
    for c, color in zip(table, SERIES):
        ax.plot(table.index, table[c] / 1000, color=color, lw=2, label=c)
        ends.append((table[c].iloc[-1] / 1000, c))
    lo, hi = ax.get_ylim()
    last = -np.inf
    for y, c in sorted(ends):
        last = max(y, last + 0.05 * (hi - lo))
        ax.text(table.index[-1] + pd.Timedelta(days=45), last, f"${y:,.0f}k", va="center",
                fontsize=8.5, color=INK)
    ax.set_ylabel("Account value ($ thousands)", color=INK_2)
    ax.set_title(f"Following the bot every month vs buying the market, ${amount:,.0f} from "
                 f"{table.index[0]:%b %Y}", color=INK, fontsize=10.5, loc="left")
    ax.legend(frameon=False, fontsize=8.5, loc="upper left", labelcolor=INK)
    ax.margins(x=0.08)
    fig.tight_layout()
    fig.savefig(fig_dir / "simulation_wealth.png", facecolor=SURFACE)
    plt.close(fig)


if __name__ == "__main__":
    main()

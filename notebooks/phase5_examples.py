"""Phase 5: worked examples of the recommendation tool (config ``recommend.examples``).

Run from the repo root:  python -m notebooks.phase5_examples

Writes to ``phase5.table_dir``:
- phase5_examples.csv           one row per scenario: inputs, γ, holdings, cash,
                                volatility, and every constraint check
- phase5_example_holdings.csv   whole-share holdings of every scenario
- phase5_example_sectors.csv    target and held sector weights of every scenario
and to ``phase5.figure_dir``: recommend_sectors.png (scenario A vs B by sector).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.data import load_config, load_processed
from src.recommend import (UserRequest, forecasts_up_to, format_recommendation, recommend,
                           resolve_as_of)

SERIES = ["#2a78d6", "#eb6834"]
INK, INK_2, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#a3a29c", "#e4e3df", "#fcfcfb"


def main() -> None:
    cfg = load_config()
    p5 = cfg["phase5"]
    tab_dir, fig_dir = Path(p5["table_dir"]), Path(p5["figure_dir"])
    tab_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    data = load_processed(cfg)
    t = resolve_as_of(data, cfg["recommend"]["as_of"])
    fc = forecasts_up_to(data, cfg, t)        # one model run shared by every scenario

    rows, holdings, sectors, recs = [], [], [], {}
    for ex in cfg["recommend"]["examples"]:
        req = UserRequest(ex["amount"], ex["risk"], ex.get("exclude", []),
                          ex.get("max_sector", {}), ex.get("min_sector", {}))
        rec = recommend(req, cfg, data, as_of=t, forecasts=fc)
        recs[ex["name"]] = rec
        rows.append({"scenario": ex["name"], **rec.summary,
                     **{f"target_{k}": v for k, v in rec.checks_target.items()},
                     **{f"held_{k}": v for k, v in rec.checks_holdings.items()}})
        holdings.append(rec.holdings.assign(scenario=ex["name"]))
        sectors.append(rec.sector_weights.assign(scenario=ex["name"]))
        print(f"\n=== {ex['name']} ===")
        print(format_recommendation(rec))

    summary = pd.DataFrame(rows)
    summary.to_csv(tab_dir / "phase5_examples.csv", index=False)
    pd.concat(holdings).rename_axis("ticker").to_csv(tab_dir / "phase5_example_holdings.csv")
    pd.concat(sectors).rename_axis("sector").to_csv(tab_dir / "phase5_example_sectors.csv")
    print("\nSummary:")
    cols = ["scenario", "gamma", "n_holdings", "invested", "cash", "target_vol_pa",
            "held_vol_pa", "max_abs_weight_gap"]
    print(summary[cols].round(4).to_string(index=False))
    checks = summary.filter(regex="^(target|held)_(long_only|cap|sector|excluded|budget)")
    print(f"\nAll constraint checks pass in every scenario: {bool(checks.all().all())}")

    # ---------------------------------------------------------------- figure
    names = [e["name"] for e in cfg["recommend"]["examples"][:2]]
    a, b = (recs[n].sector_weights["target"] for n in names)
    order = a.add(b, fill_value=0).sort_values().index
    y = np.arange(len(order))
    fig, ax = plt.subplots(figsize=(7.5, 4.8), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    ax.grid(color=GRID, linewidth=0.8, axis="x")
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=INK_2, labelsize=9)
    h = 0.38
    for k, (w, name, color) in enumerate(zip((a, b), names, SERIES)):
        ax.barh(y + (k - 0.5) * h, w.reindex(order).fillna(0).values * 100, h * 0.92,
                color=color, label=name.split(":")[0] + ": " + name.split(": ", 1)[1])
    it = list(order).index("Information Technology")
    ax.plot([20, 20], [it - 0.5, it + 0.5], color=INK, lw=1.5, ls="--")
    ax.text(20.5, it + 0.1, "20% cap (B)", fontsize=8, color=INK, va="center")
    en = list(order).index("Energy")
    ax.text(0.6, en + 0.5 * h, "0% (excluded in B)", fontsize=8, color=INK, va="center")
    ax.set_yticks(y, order)
    ax.set_xlabel("Target weight (%)", color=INK_2)
    ax.set_title(f"How sector preferences change the allocation (medium risk, as of {t.date()})",
                 color=INK, fontsize=10.5, loc="left")
    ax.legend(frameon=False, fontsize=8, loc="lower right", labelcolor=INK)
    fig.tight_layout()
    fig.savefig(fig_dir / "recommend_sectors.png", facecolor=SURFACE)
    plt.close(fig)
    print(f"\nTables in {tab_dir}/, figure in {fig_dir}/")


if __name__ == "__main__":
    main()

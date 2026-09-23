# Risk-Aware Portfolio Recommendation (FYP)

Mean-variance portfolio optimization with ML-informed Black–Litterman views, evaluated with a walk-forward backtest. See [FYP_PLAN.md](FYP_PLAN.md) for the full plan and [DECISIONS.md](DECISIONS.md) for the decision log.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
brew install libomp        # macOS only; needed by LightGBM from Phase 3
```

All parameters are in [config.yaml](config.yaml). Run every command from the repo root.

## Phases

| Phase | Status | Run |
|---|---|---|
| 0. Setup & data | done | `python -m src.data` |
| 1. Optimization core | done | `python -m notebooks.phase1_frontier` |
| 2. Backtest engine & baselines | not started | — |
| 3. ML forecasting | not started | — |
| 4. Black–Litterman integration | not started | — |
| 5. Demo & final report | not started | — |

### Phase 0 — data

```bash
python -m src.data           # download (cached in data/raw) and build data/processed
python -m src.data --force   # ignore the cache and download again
pytest tests/test_data.py
```

Outputs in `data/processed/` (git-ignored, re-created by the command above):

| File | Contents |
|---|---|
| `prices.parquet` | daily adjusted close, 51 stocks + SPY, NYSE trading days |
| `volume.parquet` | daily volume (0 on forward-filled days) |
| `returns_daily.parquet` | simple daily returns |
| `prices_monthly.parquet` | prices on the last trading day of each month |
| `returns_monthly.parquet` | row t = return over the month ending t |
| `rf_daily.parquet` | T-bill yield (`annual`) and daily risk-free return (`daily`) |
| `rf_monthly.parquet` | row t = risk-free return over the month ending t (yield set at the previous month-end) |
| `sectors.csv` | GICS sector per stock |
| `shares_outstanding.csv` | share counts snapshot, for approximate market caps (BL prior) |
| `extreme_returns.csv` | daily moves above 40%, for manual review |

Load them in code with `src.data.load_processed(load_config())`.

### Phase 1 — optimization core

```bash
pytest tests/test_estimators.py tests/test_optimization.py   # theorem checks T1–T4, Cor. 2.1
python -m notebooks.phase1_frontier                          # figures (needs Phase 0 data)
```

- [src/estimators.py](src/estimators.py): trailing-window sample covariance (scaled to monthly) and historical mean, both point-in-time.
- [src/optimization.py](src/optimization.py): closed forms (frontier constants A, B, C, D; GMV; frontier weights and variance; two-fund vectors; tangency; unconstrained mean-variance) and the constrained problem with cvxpy (long-only, weight cap, sector bounds), plus `kkt_residuals` for T4.

| Test | Result checked |
|---|---|
| `test_T1_*` | strong-concavity gap U(w*) − U(w) ≥ (γ/2)(w−w*)ᵀΣ(w−w*) ⇒ unique optimum |
| `test_T2_*` | closed-form frontier = solver; σ²(m) is the parabola (Am² − 2Bm + C)/D; GMV at the vertex |
| `test_cor21_*` | a frontier portfolio is the predicted affine combination of two others |
| `test_T3_*` | tangency formula = numerical Sharpe maximization; it lies on the frontier |
| `test_T4_*` | KKT conditions hold at the solver solution; multipliers match the marginal-utility interpretation |

Outputs in `report/figures/`:

| File | Contents |
|---|---|
| `frontier.png` | σ–μ plane at `frontier_plot.as_of`: both frontiers, GMV, tangency and capital market line, the three risk-tolerance portfolios, stocks |
| `frontier_parabola.png` | T2 parabola in (σ², μ) space with solver points on it |
| `phase1_portfolios.csv` | ex-ante return, volatility, Sharpe and holdings of the plotted portfolios |

## Tests

```bash
pytest            # all tests
pytest -m data    # only tests that read data/processed (skipped if not built)
```

`tests/pit.py` holds the shared look-ahead checks (plan Section 6): `assert_point_in_time` (outputs dated ≤ t must not change when data after t is scrambled) and `assert_uses_only_past` (an estimate made at t, such as Σ or μ, must not change).

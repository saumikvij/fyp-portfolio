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
| 2. Backtest engine & baselines | done | `python -m notebooks.phase2_backtest` |
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

### Phase 2 — backtest engine and baselines

```bash
pytest tests/test_metrics.py tests/test_backtest.py   # metrics, T7 checks, engine, look-ahead
python -m notebooks.phase2_backtest                    # S1–S4 results (about 5 seconds)
```

- [src/backtest.py](src/backtest.py): monthly walk-forward engine. At each month-end t the strategy receives only data sliced to `.loc[:t]`, returns target weights, and the portfolio is held over the next month. Tracks drift, turnover and costs (`net(cost_bps)`), and daily returns for the daily tail-risk check. Strategies S1 (1/N), S2 (SPY), S3 (minimum variance) and S4 (Markowitz with historical μ, one per γ level).
- [src/metrics.py](src/metrics.py): annualized return and volatility, Sharpe with the Lo (2002) confidence interval, maximum drawdown, historical VaR and CVaR, turnover, weight stability.
- `condition_number` in [src/estimators.py](src/estimators.py): κ(Σ) for the empirical side of T5.

| Test | Result checked |
|---|---|
| `test_T7_var_not_subadditive_counterexample` | two independent loans (4% default each): VaR₉₅(A) = VaR₉₅(B) = 0 but VaR₉₅(A+B) = 100 |
| `test_T7_cvar_subadditive_*`, `test_T7_cvar_other_coherence_axioms` | CVaR is subadditive, translation invariant, positively homogeneous and monotone on random heavy-tailed samples |
| `test_cvar_tail_average_equals_rockafellar_uryasev` | tail average = min_ζ {ζ + E[(L−ζ)⁺]/(1−α)}, including non-integer tail sizes |
| `test_sharpe_ci_coverage_iid_normal` | the Lo (2002) interval covers the true Sharpe ~95% of the time |
| `test_backtest_results_unchanged_by_future_data`, `test_strategy_never_sees_future_data` | no look-ahead in the engine or strategies |

Outputs:

| File | Contents |
|---|---|
| `report/tables/phase2_metrics.csv` | all metrics for every strategy at 0 / 10 / 25 bps |
| `report/tables/phase2_tail_ranking.csv` | strategies ranked by volatility vs by CVaR |
| `report/tables/phase2_condition.csv` | κ(Σ), λ_min, λ_max at each rebalance |
| `report/tables/phase2_returns.csv` | monthly net returns (10 bps) |
| `report/figures/backtest_wealth.png` | growth of $1 for S1, S2, S3 and S4 (γ = 5) |
| `report/figures/condition_number.png` | κ(Σ) over time |

## Tests

```bash
pytest            # all tests
pytest -m data    # only tests that read data/processed (skipped if not built)
```

`tests/pit.py` holds the shared look-ahead checks (plan Section 6): `assert_point_in_time` (outputs dated ≤ t must not change when data after t is scrambled) and `assert_uses_only_past` (an estimate made at t, such as Σ or μ, must not change).

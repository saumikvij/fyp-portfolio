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
| 1. Optimization core | not started | — |
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

## Tests

```bash
pytest            # all tests
pytest -m data    # only tests that read data/processed (skipped if not built)
```

`tests/pit.py` holds the shared look-ahead check (`assert_point_in_time`): outputs dated ≤ t must not change when data after t is scrambled (plan Section 6).

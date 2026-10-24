# Risk-Aware Portfolio Recommendation (FYP)

Mean-variance portfolio optimization with ML-informed Black–Litterman views, evaluated with a walk-forward backtest. The project plan is kept privately; see [DECISIONS.md](DECISIONS.md) for the decision log.

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
| 3. ML forecasting | done | `python -m notebooks.phase3_forecasting` |
| 4. Black–Litterman integration | done | `python -m notebooks.phase4_black_litterman` |
| 5. Demo & final report | tool done (CLI + tested examples); figures, report to come | `python -m src.recommend ...` |

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

- [src/backtest.py](src/backtest.py): monthly walk-forward engine. At each month-end t the strategy receives only data sliced to `.loc[:t]`, returns target weights, and the portfolio is held over the next month. Tracks drift, turnover and costs (`net(cost_bps)`), and daily returns for the daily tail-risk check. Strategies S1 (1/N), S2 (SPY), S3 (minimum variance) and S4 (Markowitz with historical μ, one per γ level). Every optimized strategy (S3–S6) uses `strategy_constraints`: long-only, ≤ 10% per stock, no sector above 25% (A4).
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

### Phase 3 — ML forecasting

Following the advisor's feedback (DECISIONS.md A2), the elastic net was built and evaluated first (D3.1–D3.12); LightGBM, the MLP and the Ω-B ensembles were then added with the same procedure (D3.13–D3.17).

```bash
pytest tests/test_features.py tests/test_models.py   # features, leakage, models, forecast metrics
python -m notebooks.phase3_forecasting               # 3 models + ensembles, evaluation, S5 (~1 min)
```

- [src/features.py](src/features.py): 13 price- and volume-based features at each month-end, each computed from data sliced to that date; cross-sectional rank normalization; target = next-month excess return with its realization date.
- [src/models.py](src/models.py): historical-mean baseline, elastic net, LightGBM and MLP (early stopping on chronological validation months); `walk_forward` runs the chronological procedure below, logs every retrain and fits the Ω-B bootstrap copies.
- [src/ml_eval.py](src/ml_eval.py): R²_oos (vs zero, historical mean or another model), monthly rank IC, hit rate, quintile spread, Newey–West t-statistics, out-of-sample permutation importance.
- `forecast_strategy` in [src/backtest.py](src/backtest.py): S5, mean-variance with the forecasts as μ.

**Chronological procedure** (at each month-end t; full detail in D3.5):
1. Only rows whose target is realized by t are available for training.
2. Once a year, the available feature months are split in time order: the last 24 months are validation, everything before is the tuning set. Each hyperparameter setting is fitted on the tuning set and scored by validation MSE (LightGBM and the MLP also early-stop on those months); the best is refit on all available months, plus 10 bootstrap copies for Ω-B.
3. The model forecasts next-month excess returns from the features dated t.

Every retrain is written to `report/tables/phase3_training_log.csv` (tuning, validation and training months, chosen parameters, coefficients).

| Test | Result checked |
|---|---|
| `test_leakage_training_targets_realized_by_retrain_date` | no target realized after t is used by the model forecasting at t |
| `test_leakage_validation_strictly_before_forecast_and_after_tuning` | validation months are after the tuning months and before the forecast date |
| `test_forecasts_unchanged_by_future_data`, `test_features_unchanged_by_future_data` | scrambling all data after t changes no feature or forecast dated ≤ t |
| `test_elastic_net_recovers_a_planted_signal` | with a known signal, the model finds it and reaches the theoretical best IC |
| `test_nonlinear_models_find_a_nonlinear_signal_a_linear_model_cannot` | LightGBM and the MLP find a U-shaped signal the elastic net cannot |
| `test_mlp_never_uses_random_validation_split` | the MLP's early stopping uses only the chronological validation months |
| `test_runs_are_reproducible` | identical forecasts and Ω-B on re-runs (fixed seeds) |

Outputs:

| File | Contents |
|---|---|
| `report/tables/phase3_training_log.csv` | every retrain of every model: splits, hyperparameters, validation MSE, coefficients |
| `report/tables/phase3_forecast_eval.csv` | R²_oos, IC, hit rate, quintile spread (with Newey–West t) for zero / historical mean / the three models |
| `report/tables/phase3_omega_b_check.csv` | does ensemble disagreement (Ω-B) track realized forecast errors? |
| `report/tables/phase3_importance.csv` | permutation importance per model |
| `report/tables/phase3_s5_metrics.csv` | S5 for each model and γ next to S1 and S4, incl. daily drawdown and 1-day-lag Sharpe |
| `report/tables/phase3_s5_sharpe_tests.csv` | paired Sharpe-difference tests (S5 vs S4, S1, and nonlinear vs linear), Holm-adjusted |
| `data/ml/forecasts_<model>.parquet`, `data/ml/omega_b_<model>.parquet` | forecasts and Ω-B for Phase 4 (git-ignored; rebuilt by the script) |
| `report/figures/forecast_ic.png` | cumulative rank IC over time |
| `report/figures/feature_importance.png` | permutation importance, one panel per model |

Phase 2 outputs now also include paired Sharpe-difference tests with Holm adjustment and alternative block lengths (`phase2_sharpe_tests.csv`, D2.11–D2.12) and a one-day execution-lag check (`phase2_execution_lag.csv`). See R1 in DECISIONS.md for the full methodology review.

### Phase 4 — Black–Litterman integration

```bash
pytest tests/test_black_litterman.py           # T6, Cor. 6.1, prior, Ω, look-ahead
python -m notebooks.phase4_black_litterman     # S6 + ablations (~1 min; needs Phase 3 outputs in data/ml)
```

- [src/black_litterman.py](src/black_litterman.py): market weights (close × shares), prior Π = δΣw_mkt, posterior mean in three equivalent forms (Bayesian, GLS, update), posterior covariance, forecast errors, Ω-A (past squared errors, shrunk toward return variance at the start) and Ω-B (ensemble disagreement, raw or level-calibrated), and `bl_inputs` for one date.
- `black_litterman_strategy` in [src/backtest.py](src/backtest.py): S6; `views=False` gives the prior alone (the Ω → ∞ limit).
- The headline S6 (elastic net, Ω-A, τ = 0.05, γ = 5) was fixed in `config.yaml` before any result was computed; all other variants are ablations (D4.1).

| Test | Result checked |
|---|---|
| `test_T6_bayesian_and_gls_derivations_agree` | Bayesian, GLS and update forms of μ_BL agree (P = I and 3 mixed views) |
| `test_T6_posterior_covariance`, `test_T6_scalar_case_is_precision_weighted_average` | M is symmetric PD and τΣ − M ⪰ 0; diagonal case is a precision-weighted average |
| `test_cor61_huge_omega_returns_prior`, `test_cor61_tiny_omega_makes_views_hold_exactly` | Ω → ∞ gives Π; Ω → 0 gives Pμ = Q |
| `test_reverse_optimization_recovers_market_weights` | at μ = Π, mean-variance with γ = δ returns w_mkt |
| `test_bl_inputs_unchanged_by_future_data`, `test_s6_backtest_unchanged_by_future_data_*` | Π, Q, Ω, μ_BL and S6 weights at t use nothing after t |

Outputs:

| File | Contents |
|---|---|
| `report/tables/phase4_metrics.csv` | every strategy and ablation (γ, τ, Ω scale, costs) |
| `report/tables/phase4_diagnostics.csv` | how far μ_BL moves from the prior toward the views; Ω relative to τσ² |
| `report/tables/phase4_sharpe_tests.csv` | S6 vs S5, S4, S1 and the prior, Holm-adjusted, with alternative block lengths |
| `report/tables/phase4_turnover_tests.csv` | paired test of monthly turnover, S6 vs S5 |
| `report/figures/bl_wealth.png` | growth of $1: S1, S4, S5 and headline S6 |
| `report/figures/bl_turnover.png` | turnover of S5 vs S6 (each Ω) per model |
| `report/figures/bl_tau.png` | Sharpe and turnover of S6 across τ |

### Phase 5 — recommendation tool

```bash
python -m src.recommend --amount 10000 --risk medium
python -m src.recommend --amount 10000 --risk low --json out.json
python -m src.recommend --amount 10000 --risk medium --as-of 2022-06-30     # what it would have said then
python -m notebooks.phase5_walkthrough   # one complete example, every step: report/phase5_walkthrough.md
python -m notebooks.phase5_examples      # the five worked examples in config.yaml
pytest tests/test_recommend.py
```

[src/recommend.py](src/recommend.py) turns a request into a whole-share portfolio:
1. **Inputs:** amount (> 0) and risk tolerance (low / medium / high → γ = 10 / 5 / 2). Nothing else: the sector rule is chosen by us, not the user (advisor feedback A4).
2. **Constraints:** exactly those of the backtested strategies (`src.backtest.strategy_constraints`): long-only, ≤ 10% per stock, **no sector above 25%** (`optimization.sector_max` in `config.yaml`). The same rule applies to every optimized strategy in Phases 2–4 (S3–S6).
3. **Model:** the Phase 4 headline: elastic-net forecasts (re-run with the Phase 3 walk-forward up to the recommendation date), Black–Litterman with Ω-A and τ = 0.05, the trailing 252-day Σ.
4. **Target weights:** the constrained mean-variance problem; these equal the backtest's S6 weights at that date (tested).
5. **Whole shares** at the close of the recommendation date: round down, then spend cash only where it moves a holding closer to its target, never breaking a cap.
6. **Output:** holdings (shares, value, weight vs target), cash, expected volatility, model expected excess return, and every constraint checked on both the target weights and the whole-share holdings.

[report/phase5_walkthrough.md](report/phase5_walkthrough.md) documents one complete example ($10,000, medium risk) from input to whole shares: γ, the forecasts, Π, Ω and μ_BL, which limits bind, the KKT check, the holdings and every constraint check.

| Test | Result checked |
|---|---|
| `test_worked_example_satisfies_every_constraint` | $10,000, medium: every constraint (stock cap, 25% sector rule, budget) holds for target weights and for whole shares |
| `test_tool_target_equals_backtest_weights` | at three dates the tool's target weights equal the backtested S6 weights (difference < 1e-9), with the same constraints |
| `test_sector_rule_binds_in_the_backtest` | without the rule, high-risk Markowitz puts > 25% in one sector in most months; with it, never |
| `test_sector_rule_comes_from_config`, `test_infeasible_sector_rule_is_rejected` | every sector gets (0, 25%); a rule leaving room for < 100% is rejected |
| `test_higher_risk_tolerance_gives_higher_volatility` | low < medium < high ex-ante volatility |
| `test_amount_scales_shares_not_target_weights` | the amount changes the shares, not the target; larger amounts track the target more closely |
| `test_invalid_requests_are_rejected` | non-positive amounts and unknown risk levels give a clear error |
| `test_recommendation_is_reproducible`, `test_forecasts_match_phase3` | identical output on re-runs; the tool's forecasts equal the Phase 3 forecasts |
| `test_past_recommendation_uses_no_later_data` | a recommendation as of 2022-06-30 is unchanged when all later data is scrambled |

Outputs: `report/phase5_walkthrough.md` with `report/tables/phase5_walkthrough_*.csv`; `report/tables/phase5_examples.csv` (one row per scenario, with every constraint check), `phase5_example_holdings.csv`, `phase5_example_sectors.csv`, and `report/figures/recommend_sectors.png`.

## Final design (D6): two return models + ML risk model

```bash
python -m notebooks.final_design
```

Chosen after the experiments on branch `experiment/model-search` (see DECISIONS.md D6), and reported next to the pre-registered Phase 4 headline:
1. **Views:** Phase 3 elastic-net and LightGBM forecasts, averaged into one view set (their errors are 0.999 correlated, so stacking them would double-count).
2. **ML risk model** ([src/risk_model.py](src/risk_model.py)): an elastic net forecasts each stock's next-month volatility from its 1-week, 1-month, 3-month and 1-year volatility, its high–low range and market volatility; Σ = D·R·D with D the forecast volatilities and R the trailing correlation. Passed to Black–Litterman with `vol_forecasts=` (prior, posterior and optimizer all use it).
3. **Black–Litterman** as in Phase 4 (Ω-A, τ = 0.05, γ = 5, 10% cap, 10 bps).

Outputs in `report/tables/`: `final_risk_accuracy.csv` and `final_risk_dm.csv` (risk model vs trailing volatility and HAR, Diebold–Mariano), `final_risk_level.csv` (stated vs realized portfolio risk), `final_design_metrics.csv`, `final_design_tests.csv` (paired Sharpe tests, Holm).

| Test (`tests/test_risk_model.py`) | Result checked |
|---|---|
| `test_target_is_next_month_realized_volatility` | the target at t is the volatility of the days after t up to the next month-end |
| `test_features_use_only_past_data`, `test_vol_forecasts_unchanged_by_future_data` | scrambling data after t changes nothing dated ≤ t |
| `test_dr_covariance_is_psd_with_forecast_diagonal`, `test_dr_covariance_uses_only_past` | Σ = D·R·D is symmetric PSD, its diagonal equals the forecast variances, and it uses only data ≤ t |
| `test_bl_uses_ml_sigma_only_when_given` | without `vol_forecasts` Black–Litterman is unchanged; with it, Σ = D·R·D |
| `test_har_recovers_a_linear_relation`, `test_diebold_mariano` | the HAR baseline and the Diebold–Mariano test behave as defined |

## Tests

```bash
pytest            # all tests
pytest -m data    # only tests that read data/processed (skipped if not built)
```

`tests/pit.py` holds the shared look-ahead checks (plan Section 6): `assert_point_in_time` (outputs dated ≤ t must not change when data after t is scrambled) and `assert_uses_only_past` (an estimate made at t, such as Σ or μ, must not change).

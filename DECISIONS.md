# Decision log

Every choice not fixed by the project plan is recorded here: what was chosen, what else was considered, and why. Newest entries go at the bottom of each phase.

## Phase 0 — Setup & data (23 Sep 2026)

### D0.1 Repo lives in `FYP/`, not `fyp-portfolio/`
- **Chosen:** use the existing `FYP/` git repo as the project root; plan Section 7 updated to match.
- **Alternative:** a `fyp-portfolio/` subfolder, as the plan originally showed.
- **Why:** the plan and git history already live in `FYP/`; a subfolder adds nesting for no benefit.

### D0.2 Stock universe: 51 stocks, fixed in `config.yaml`
- **Chosen:** 51 current S&P 500 large caps, 4–6 per GICS sector (6 in IT; 5 each in Consumer Discretionary, Consumer Staples, Health Care, Financials, Industrials; 4 each in the other five), all trading since before January 2010.
- **Rule:** the stock must have a full price history from the start of the sample, so every stock is in the backtest for the whole period and the covariance matrix is always n × n with n fixed.
- **Consequences:** stocks listed after 2010 are excluded even though they are large today, for example META (IPO 2012), TSLA (IPO June 2010), ABBV (2013), PYPL (2015), and LIN (current listing 2018). This makes survivorship bias stronger (plan Section 3), which the report must discuss.
- **Alternative:** a point-in-time S&P 500 membership list. Rejected because it needs membership data that isn't freely available, and a changing universe complicates Σ and the BL prior.

### D0.3 Sectors are GICS labels fixed in the config, not taken from Yahoo
- **Why:** Yahoo uses its own sector names (e.g. "Technology", "Consumer Cyclical") that don't match GICS, and they can change between downloads. Fixed labels keep sector constraints reproducible.

### D0.4 Adjusted close and simple returns
- **Chosen:** Yahoo's adjusted close (includes splits and dividends), simple returns r = P_t/P_{t−1} − 1.
- **Why:** portfolio returns are weighted sums of simple returns (μᵀw), which the whole mean-variance framework relies on. Log returns don't aggregate across assets.

### D0.5 Trading calendar = SPY's trading days
- **Why:** gives one NYSE calendar for all series. ^IRX trades on the bond calendar and has a few different holidays; it is aligned to this calendar.

### D0.6 Cleaning rules
- Short gaps (≤ 5 trading days) are forward filled with the last price, which uses only past data. A ticker with > 1% missing days, a longer gap, or no price on the first day is rejected with an error, not silently repaired.
- Volume is set to 0 on filled days (no trading happened).
- Non-positive prices are treated as missing.
- Daily moves above 40% are logged to `extreme_returns.csv` for manual review, not removed. None occurred in the current data (largest: ORCL +36% in Sept 2025, which is genuine).
- All thresholds live in `config.yaml`.

### D0.7 Risk-free rate conversion and timing
- **Chosen:** y = ^IRX / 100, treated as an annual rate and compounded: r_monthly = (1+y)^(1/12) − 1, r_daily = (1+y)^(1/252) − 1.
- **Timing:** the risk-free return for the month ending at t uses the yield observed at the previous month-end (known when the month starts). No look-ahead.
- **Approximation:** ^IRX is quoted on a discount basis, not as an effective annual yield. The difference is a few basis points per year, negligible next to equity returns. Note it in the report.
- **Alternative:** simple division y/12. The compounded version is almost identical and slightly more correct.

### D0.8 Sample end fixed at 2026-08-31
- **Chosen:** a fixed end date in the config (the last complete month) rather than "latest available".
- **Why:** reproducibility. The raw download is cached in `data/raw`, so re-running gives identical data. A fresh download (`--force`) can differ slightly because Yahoo rescales past adjusted prices after each new dividend; returns barely change. To extend the sample, change `data.end` and re-run with `--force`.

### D0.9 Shares outstanding: one snapshot (23 Sep 2026), all share classes
- Saved in `shares_outstanding.csv` with an `as_of` date. Used only in Phase 4 for the approximate market-cap weights of the BL prior (plan Section 3 limitation).
- **Chosen:** Yahoo's `impliedSharesOutstanding`, which counts all share classes and matches Yahoo's reported market cap for all 51 stocks.
- **Rejected:** Yahoo's `sharesOutstanding` (first version). It counts one share class only and understated five market caps: GOOGL by 2.08× (class A only: $2.0T instead of $4.3T), NKE 1.23×, SPG 1.17×, UPS 1.14×, PLD 1.02×. That would have distorted w_mkt and so the BL prior Π. Found in the data audit (D0.12); the single-class count is kept as `shares_primary` for reference.
- **Caveat:** for SPG and PLD the implied count includes operating-partnership units that convert into shares. This matches how Yahoo computes market cap.

### D0.10 Environment: Python 3.14 venv, pinned versions
- Versions pinned in `requirements.txt`. LightGBM needs the OpenMP runtime on macOS (`brew install libomp`); installed 23 Sep 2026 and LightGBM 4.7.0 verified to train.

### D0.11 Plan parameters entered in `config.yaml` now
- γ levels (10/5/2), τ = 0.05 and its grid, δ = 2.5, cost levels (0/10/25 bps), w_max = 10%, out-of-sample start 2016-01, 252-day covariance window, α = 0.95 are copied from the plan. The τ grid values (0.01–0.25) are my choice; the plan only says "a grid".

### D0.12 Data audit (23 Sep 2026): result and what was checked
A full audit of `data/raw` and `data/processed` beyond the unit tests. Result: **clean**, after the share-count fix in D0.9.
- **Integrity:** no NaN, no inf, no duplicate dates in any table; all 52 price series complete from 4 Jan 2010 (zero forward-filled days for stocks; 2 holiday days filled for ^IRX).
- **Calendar:** no weekends; 250–253 trading days per year. The only gap longer than a normal weekend is 29–30 Oct 2012 (market closed for Hurricane Sandy), which is genuine.
- **Stale prices:** longest run of unchanged prices is 2 days; ≤ 1.5% of days have zero return for any stock.
- **Splits:** no day where the unadjusted close jumps > 35% without the adjusted price following (the only > 35% move is ORCL +35.9% on 10 Sep 2025, a real earnings jump).
- **Dividends:** the adjustment factor (Adj Close / Close) is ≤ 1 and never decreases over time for any stock.
- **Spin-offs:** Yahoo has already adjusted past prices for every spin-off checked (ABT→AbbVie 2013, COP→Phillips 66 2012, HON 2018 and 2025, MRK→Organon 2021, PFE→Viatris 2020, SPG 2014, APD 2016, CMCSA→Versant 2026): no artificial price drop on the spin-off dates.
- **Large moves:** 73 daily moves above 15%. Most fall on market-wide event days (COVID crash and rebound March 2020, vaccine news 9 Nov 2020); the rest are single-stock moves on known news dates (e.g. NVDA earnings, UNH 2025). A bad price usually shows as a spike that reverses the next day: all 26 such cases are 9–23 Mar 2020, when SPY itself moved about ±9% a day, and there are none outside that window. None removed.
- **Volume:** split-adjusted (no step change at the AAPL, NVDA, AMZN, GOOGL, WMT splits), no zero-volume days.
- **Sanity:** SPY 14.2% per year over the sample; the equal-weight stock return has 0.95 correlation with SPY; ^IRX between −0.1% and 5.35%, with year-end levels matching the known rate history (near zero 2010–2015 and 2020–21, about 5% in 2023). The 7 slightly negative days are all 19–27 Mar 2020, when T-bill yields briefly dipped below zero.

## Phase 1 — Optimization core (23 Sep 2026)

### D1.1 Covariance: trailing 252 days, scaled to monthly by × 21
- Σ_monthly = (252/12) × Σ_daily, which assumes daily returns are uncorrelated over time. Using daily data keeps the estimate recent; monthly returns over 252 days would give only 12 observations for a 51 × 51 matrix, which would be singular.
- Checked on the real data: Σ is positive definite at every yearly check date (the standing assumption of T1–T5).

### D1.2 Historical mean: expanding window of raw (not excess) monthly returns
- **Why raw is fine:** with the budget constraint 1ᵀw = 1, adding a constant c to every μᵢ adds c to the objective and doesn't change the optimal w, even with bounds. So subtracting r_f doesn't matter for mean-variance. Only the tangency portfolio needs μ − r_f 1, and it takes r_f explicitly.
- Window length is in the config (`estimators.mean_window_months`, null = expanding).

### D1.3 Solver: CLARABEL (interior point) rather than OSQP
- T4 needs accurate Lagrange multipliers. CLARABEL returns duals accurate to about 1e-9; OSQP (first-order ADMM) is faster but less accurate. At n = 51 speed doesn't matter.

### D1.4 KKT sign convention and multiplier meaning
- Stationarity is written as ∇f + ν1 − λ + η + Sᵀ(ρ_up − ρ_low) = 0 for f = (γ/2)wᵀΣw − μᵀw, matching cvxpy's duals (verified numerically).
- Equivalently μᵢ − γ(Σw)ᵢ = ν − λᵢ + ηᵢ: every asset held strictly inside its bounds has the same marginal utility ν; a zero-weight asset has marginal utility ν − λᵢ ≤ ν; an asset at the cap has ν + ηᵢ ≥ ν. Tested in `test_T4_multiplier_interpretation`. Use the same convention in the written T4 proof.

### D1.5 Weight tolerance 1e-5 for "at the bound"
- Interior-point solvers stop a small distance inside active bounds (e.g. 4e-7 instead of 0, 0.1999993 instead of 0.2). `optimization.weight_tol` = 1e-5 decides when a weight counts as zero or capped (tests, holdings counts). Solver weights are not rounded, so the KKT checks stay exact.

### D1.6 How T1 is tested
- Instead of comparing two solvers, the test checks the inequality behind the uniqueness proof: at the optimum w*, U(w*) − U(w) ≥ (γ/2)(w − w*)ᵀΣ(w − w*) for 200 random feasible w. With Σ ≻ 0 the right side is > 0 for w ≠ w*, so no other feasible point ties.

### D1.7 Sector constraints as {sector: (lower, upper)}
- One format covers the plan's "optional sector constraints" and the user's sector preferences (Phase 5): excluding a sector is (0, 0), capping one is (0, u). Default: none (`optimization.sector_bounds: {}`). Unknown sector names raise an error rather than being ignored.

### D1.8 Figures come from a script, not a Jupyter notebook
- `notebooks/phase1_frontier.py` runs with one command and always rebuilds the figures from the config. Jupyter isn't installed, and a script is easier to re-run after data changes.
- Axes are annualized (mean × 12, volatility × √12). Stocks with volatility above 40% per year or returns above the plot range are left out of the zoomed view and counted in a note (3 stocks).
- r_f for the tangency portfolio is the T-bill yield observed at the estimation date (3.67% per year on 2026-08-31).

### D1.9 Finding to discuss in the report: in-sample optimism of historical means
- At 2026-08-31 the ex-ante Sharpe ratios are about 1.6–2.5 per year (`phase1_portfolios.csv`), far above what stocks deliver out of sample (SPY: about 0.74 over 2010–2026, from 14.2% return, 17.1% volatility, 1.5% average r_f). This is estimation error in μ (plus survivorship bias, D0.2) being "optimized" into the portfolio. It is the motivation for T5 (sensitivity) and for Black–Litterman, and the Phase 2 backtest will measure it out of sample.

## Advisor meeting (23 Sep 2026)

### A1 Phases 0–1 approved as they stand
- Presented: the data pipeline and audit, the optimization core, the T1–T4 checks and the frontier figures (progress page).
- Four points were raised for confirmation: universe of stocks listed before 2010 (D0.2), covariance from daily returns scaled × 21 (D1.1), current share counts for BL market caps (D0.9), and proving T5 for the unconstrained / equality-constrained case.
- **Outcome:** the advisor said all of it is fine as it is. No changes; the decisions above stand.

## Phase 2 — Backtest engine & baselines (26 Sep 2026)

### D2.1 Look-ahead is prevented by construction
- The engine slices every table to `.loc[:t]` into a `History` object before calling a strategy, so a strategy cannot read future data even by mistake. Phase 3 models and the Phase 4 BL strategy will use the same interface.
- Two tests back this up: a strategy asserts it never sees a date after t, and scrambling all data after t leaves every weight and return dated ≤ t unchanged.

### D2.2 Timing: 128 out-of-sample months, Jan 2016 – Aug 2026
- The first portfolio is chosen at 31 Dec 2015 using only data to that date and held over January 2016; the last is chosen at 31 Jul 2026 and held over August 2026.

### D2.3 Turnover measured against drifted weights; the first trade counts
- Turnover at t = Σ|w_target − w_drifted|, where drifted weights are last month's after a month of price moves. So even 1/N trades a little each month (4.6% on average).
- Building the portfolio from cash is turnover 1 and is charged costs, the same for every strategy. The *average* turnover reported leaves this first month out, so SPY buy-and-hold shows 0.

### D2.4 Costs: multiplicative, charged at the start of the month
- Net return = (1 − c·turnover)(1 + gross) − 1 with c = bps / 10 000 (grid 0 / 10 / 25). For the daily series the cost is charged on the month's first trading day, so daily net returns compound exactly to the monthly net return (tested).
- Weights don't depend on costs, so each strategy runs once and costs are applied afterwards.

### D2.5 Solver dust removed from backtest weights
- Weights below `weight_tol` (1e-5) are set to 0 and the rest rescaled to sum to 1, so solver noise of ~1e-9 doesn't show up as turnover.

### D2.6 VaR and CVaR estimators
- VaR_α is the lower empirical quantile of the loss (smallest ℓ with F̂(ℓ) ≥ α). CVaR is computed with the Rockafellar–Uryasev formula at ζ = VaR_α; the tests confirm it equals both the direct tail average and the RU minimum over ζ.
- **Caveat for the report:** 128 monthly returns give a 5% tail of only about 6.4 observations, so monthly VaR / CVaR are noisy. As the plan suggests, daily VaR / CVaR (about 2,680 days) are reported as a check, and they give the same ranking.

### D2.7 Sharpe confidence interval: Lo (2002), i.i.d. version
- SE = √((1 + SR²/2)/T) per month, annualized by √12, z = 1.96. The formula is cited (proof is optional item O1). A Monte Carlo test confirms about 95% coverage for i.i.d. normal returns. Lo's autocorrelation-adjusted version isn't used.

### D2.8 T7 counterexample used in the tests
- Two independent loans, each losing 100 with probability 4% (else 0). Alone, P(loss > 0) = 4% < 5%, so VaR₉₅ = 0; together, P(any default) = 1 − 0.96² = 7.84% > 5%, so VaR₉₅(A+B) = 100 > 0 + 0. The same numbers can be used in the written proof of T7.

### D2.9 Strategy set
- S3 and S4 use the same constraints as the plan's main problem: long-only, 10% cap, no sector bounds (config default). S4 is run for all three γ levels; γ = 5 (medium) is the headline S4 in figures.
- Weight stability = mean Σ|w_t − w_{t−1}| between consecutive *target* portfolios, which separates the optimizer changing its mind (T5) from price drift.

### D2.10 Findings from the baseline backtest (10 bps)
- **Markowitz doesn't beat 1/N out of sample**, in line with DeMiguel et al. (2009): Sharpe 0.96 for S1 against 0.85–0.95 for S4. *(Corrected 3 Oct, see A2: the original wording inferred non-significance from overlapping individual intervals, which is not valid. The paired tests in D2.11 now support it.)*
- **Drawdowns (corrected 3 Oct, see D2.12):** on month-end values S4 has the worst drawdowns (26–30% against 19% for 1/N and 24% for SPY), but on the daily path the order reverses: 1/N 33%, SPY 34%, S4 29–31%, S3 28%. Month-end prices hid how far 1/N and SPY fell inside March 2020, so *no strategy can be said to have the worst drawdown*. S4 does have the least stable weights. Its *target* weights change by 13–20% a month (Σ|w_t − w_{t−1}|), and its *turnover* after price drift is 16–21% a month (Σ|w_t − w_drifted|), against 0% and 4.6% for 1/N. See A2 for the difference between the two measures. This is the empirical side of T5.
- **Minimum variance (S3) does what it promises:** lowest volatility (11.9%), CVaR (6.4% monthly) and drawdown (16%), but also the lowest return (12.0%) and the highest turnover (25%).
- **Variance and CVaR disagree once:** SPY has lower volatility than S4 (γ = 5) (15.1% vs 15.3%) but a worse CVaR (monthly 9.1% vs 8.7%; daily 2.71% vs 2.57%). The gap is small but consistent at both frequencies. This is the tail-risk discussion the plan asks for.
- **κ(Σ)** ranges from 129 to 1,540 (median 371), with the maximum in April 2020, when the COVID crash made one market-wide eigenvalue dominate. Errors in μ can therefore be amplified by factors in the hundreds (T5).
- Costs matter most for the high-turnover strategies: going from 0 to 25 bps lowers S3's Sharpe from 0.86 to 0.79 but barely moves 1/N or SPY.

### D2.11 Paired tests of Sharpe-ratio differences (added 3 Oct after advisor feedback)
- **Chosen:** for every pair of strategies, a paired circular block bootstrap of the annualized Sharpe difference (block 6 months ≈ T^(1/3), 10 000 samples, p-value from the re-centred bootstrap distribution, in the spirit of Ledoit & Wolf 2008). It draws the same months for both strategies, so it keeps their correlation and short-run autocorrelation, and it doesn't assume normality.
- **Check:** the Jobson–Korkie test with Memmel's (2003) correction, an analytic test that accounts for the correlation ρ between the two strategies but assumes i.i.d. normal returns. Both are in `src/metrics.py`; tests confirm Memmel's test has ≈ 5% size for equal Sharpes with ρ = 0.9, and that the bootstrap detects a real gap.
- **Result (10 bps, `phase2_sharpe_tests.csv`):** none of the 15 pairs differs significantly at 5%. Smallest bootstrap p-value 0.29 (S4 γ = 5 vs γ = 2); 1/N vs SPY +0.09 [−0.11, 0.30], p = 0.37; 1/N vs Markowitz γ = 5 +0.08 [−0.33, 0.48], p = 0.70. The two methods agree on every pair (all p > 0.29).
- Because the strategies are highly correlated (ρ = 0.71–0.97), the intervals for the *differences* are much narrower than the individual Sharpe intervals suggested, so the paired test is the right tool. No multiple-testing correction is needed since nothing rejects.

## Advisor feedback (received 3 Oct 2026, on the Phase 2 report)

### A2 Three points, and what was done
1. **Statistical significance.** Overlapping individual Sharpe intervals do not show that a difference is insignificant. → Added direct paired tests of Sharpe differences that account for dependence between strategies (D2.11). The conclusion now rests on those tests, and D2.10 is corrected.
2. **"13–20% monthly change" was ambiguous.** That figure is the change in *target* weights between consecutive months, Σ|w_t − w_{t−1}| (weight stability). The amount actually *traded* is turnover, Σ|w_t − w_drifted|, measured after a month of price drift: 16–21% a month for Markowitz. Both sum buys and sells, so 20% turnover means about 10% of the portfolio sold and 10% bought. From now on reports name the measure every time and lead with turnover (what costs are charged on). D2.10 updated.
3. **Next-week priority: one forecasting model first.** → Phase 3 starts with a single model (elastic net) and a documented chronological training/validation procedure, compared with simple baselines, before LightGBM and the MLP are added (D3.x).

## Phase 3 — ML forecasting, step 1 (3 Oct 2026)

### D3.1 One model first: elastic net
- Per the advisor (A2.3), one model is completed and compared with baselines before LightGBM and the MLP. The elastic net goes first: it is linear, its coefficients can be read directly, and ridge and lasso are special cases.

### D3.2 Features: 13, not "about 15"
- The plan's list, with one merge: "1-month momentum" and "1-month reversal" are the same signal (last month's return; the sign is learned), so it appears once as `rev_1m`. Definitions are in `src/features.py`.
- Windows: 21 / 63 / 252 trading days. RSI uses simple 14-day averages of gains and losses (Wilder's exponential smoothing would also be point-in-time, but the simple version is easier to state). "Volume trend" uses share volume (log of 1-month over 12-month average), because historical shares outstanding aren't available for a true turnover ratio.
- Features start in Jan 2011, the first month-end with 12 months of monthly prices and 252 trading days.

### D3.3 Cross-sectional rank normalization to [−0.5, 0.5]
- Each month every feature is replaced by its rank among the 51 stocks, scaled to [−0.5, 0.5], ties averaged (plan 4.1). This removes outliers and level shifts over time.

### D3.4 Target: next-month excess return, not ranked
- target(i, t) = r_{i,s} − r_f,s for the next month-end s, kept in return units because the Black–Litterman views Q (Phase 4) must be return forecasts. The realization date s is stored with every row, so training sets are cut by when a target became known, not by feature date.

### D3.5 Chronological training and validation procedure
At each forecast date t (Dec 2015 – Jul 2026, 128 month-ends):
1. **Available data:** rows with target realized by t (target_date ≤ t), i.e. feature months up to t − 1 month. Expanding window starting Jan 2011.
2. **Retrain once a year** (plan 4.3), starting Dec 2015. Split the available feature months in time order: the **last 24 months are validation**; all earlier months are the **tuning set**. Fit every (α, l1_ratio) on the tuning set, score by validation MSE, keep the best.
3. **Refit** the chosen setting on tuning + validation months, so the most recent data is used.
4. **Forecast** every month until the next retrain, using the fixed model and that month's features.
- Example, first retrain (Dec 2015): tuning Jan 2011 – Nov 2013, validation Dec 2013 – Nov 2015, refit on Jan 2011 – Nov 2015 (3,009 rows, last target realized Dec 2015), forecasts made Dec 2015 – Nov 2016.
- Every retrain is logged in `phase3_training_log.csv` (11 retrains).
- No random k-fold anywhere; the seed is fixed (42) for reproducibility.

### D3.6 Hyperparameter grid
- α ∈ {1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2}, l1_ratio ∈ {0.1, 0.5, 0.9, 1.0}: 28 settings. The α range is set so that, for rank features in [−0.5, 0.5] and monthly returns with sd ≈ 7%, it spans "all features kept" to "all coefficients zero".

### D3.7 Baselines
- **Historical mean (plan baseline):** each stock's average realized excess return so far, refit monthly (expanding).
- **Zero forecast:** the Gu, Kelly & Xiu (2020) benchmark for R²_oos.

### D3.8 Leakage guarantees
- Features: each month-end's features are computed from data sliced to that date; a test scrambles all data after t and checks no feature dated ≤ t changes.
- Training: the procedure only uses rows with target_date ≤ t. Tests check, for every retrain, that the last target used is realized by the retrain date, that validation months lie after the tuning months and before the forecast date, and (end to end) that scrambling all raw data after t leaves every forecast dated ≤ t identical.
- Target alignment was checked on the real data: the target at t has correlation 0.9998 with next month's return (the gap is r_f) and −0.003 with the current month's.

### D3.9 Evaluation definitions
- R²_oos pooled over all stock-months, against zero and against the historical mean.
- Rank IC: monthly Spearman correlation; its mean and t-statistic are over months where the forecasts differ across stocks (all 128 for the historical mean, 80 for the elastic net, see D3.11). The difference to the baseline uses a paired t-test on the monthly IC.
- Hit rate reported with the base rate (share of positive realized returns, 56.5%).
- Quintile spread: top minus bottom fifth by forecast, equal-weighted, monthly.
- Permutation importance: each retrained model is scored only on the months it actually forecast; MSE increase when one feature is shuffled, 20 repeats, weighted by block size.

### D3.10 S5 definition
- Mean-variance with μ = elastic-net forecasts, the same Σ and constraints as S4 (long-only, 10% cap), all three γ levels.

### D3.11 Findings (step 1): no reliable predictive benefit detected under the tested conditions
- **Forecast accuracy:** R²_oos against zero is 2.01% (elastic net) vs 1.72% (historical mean). Most of both comes from predicting a positive average return (mean² / variance of the target ≈ 2.7%), not from telling stocks apart. Against the historical mean the elastic net improves R²_oos by only 0.30%.
- **Ranking stocks:** the elastic net's rank IC is −0.013 (t = −0.48), the historical mean's +0.018 (t = 0.80); the difference has t = −1.44. Quintile spread: elastic net −0.14% a month (t = −0.26), historical mean +0.40% (t = 0.81). Hit rate 56.5% for both, equal to the base rate. Neither model ranks stocks better than chance.
- **The model often switches itself off:** in 4 of 11 retrains (Dec 2016, 2017, 2020, 2021) validation chose α large enough to zero every coefficient, so every stock got the same forecast (48 of 128 months). The procedure is doing its job: in those years no feature helped on validation data.
- **Why:** no feature has a significant univariate rank IC in this universe, either out of sample or over 2011–2026 (largest t-statistic 1.67, for 1-month volatility). This fits the literature that return anomalies are much weaker among the largest stocks (e.g. Gu, Kelly & Xiu 2020; Hou, Xue & Zhang 2020).
- **Permutation importance:** beta, idiosyncratic volatility and RSI helped out of sample; 1-month volatility, max drawdown, max daily return and 12–1 momentum made forecasts worse.
- **S5 (10 bps):** Sharpe 0.69–0.76 against 0.85–0.95 for S4 and 0.96 for 1/N, with much lower drawdowns (16–19%) but **turnover of 60–80% a month** after drift (target weights change 58–79% a month). Against S4 the differences are not significant (bootstrap p 0.44–0.50); against 1/N they are borderline (p = 0.07, 0.08, 0.21 for γ = 10, 5, 2).
- **Interpretation for the project:** noisy forecasts fed straight into mean-variance produce unstable, expensive portfolios, which is exactly the T5 problem. This is the case Black–Litterman is designed for: with Ω set from each stock's past forecast errors, weak forecasts should be shrunk toward the market prior. S6 vs S5 in Phase 4 tests that directly.

### D3.12 Open question for the advisor before step 2
- Given D3.11, LightGBM and the MLP might find nothing a linear model missed. **Resolved 3 Oct:** the student chose option (a), to proceed as planned and report all three models honestly (D3.13–D3.17).

### D2.12 Robustness of the Phase 2 results (added 3 Oct)
- **Bootstrap block length:** Sharpe-difference p-values with 3- and 12-month blocks are close to the 6-month ones for every pair (`boot_p_block3`, `boot_p_block12` in `phase2_sharpe_tests.csv`); no conclusion changes.
- **Multiple testing:** the 15 pairwise tests form one family; Holm-adjusted p-values (`boot_p_holm`) are all 1.0. Nothing was significant before adjustment either.
- **Daily drawdowns:** `max_drawdown_d` is now reported next to the month-end figure; it changes the drawdown ranking (D2.10 corrected).
- **Execution timing:** the main backtest trades at the close of the decision day t, as in the academic literature, since all signals use data up to that close. Trading one day later instead (`phase2_execution_lag.csv`) changes no Sharpe ratio by more than 0.02.

## Phase 3 — ML forecasting, step 2: all three models (3 Oct 2026)

### D3.13 LightGBM and MLP set-up
- **LightGBM:** grid num_leaves ∈ {7, 15} × min_child_samples ∈ {100, 300}; learning rate 0.02, row and feature subsampling 0.8, L2 penalty 1. For each setting the number of trees (up to 2,000) is chosen by early stopping on the chronological validation months; the final refit uses that count. Deterministic mode, one thread, fixed seed.
- **MLP:** scikit-learn MLPRegressor with 2 or 3 hidden layers ((32, 16) or (32, 16, 8)) × L2 penalty {1e-4, 1e-2}, Adam, batch 256, targets scaled to unit variance. **Early stopping uses the chronological validation months**: the network is trained one epoch at a time and the epoch with the lowest validation MSE is kept (patience 20, up to 200). scikit-learn's built-in early stopping is *not* used because it holds out a random 10% of rows, which would mix future months into training (plan 4.3). A test checks it stays off.
- Same splits, retrain dates and leakage tests as the elastic net (D3.5, D3.8); the leakage and look-ahead tests now run for all three models.
- **Capability check:** with a planted U-shaped signal that has no linear trend, LightGBM and the MLP reach rank IC > 0.3 while the elastic net stays near 0 (test). So the weak real-data results are not from models unable to learn.

### D3.14 Ω-B ensembles
- For each model and retrain, 10 extra copies are fitted on bootstrap samples of whole months (keeps each month's cross-section), each with its own seed. Ω-B for stock i at t = variance of the 10 copies' forecasts. The main forecast stays the single tuned model; the copies only measure disagreement. Saved to `data/ml/omega_b_<model>.parquet` for Phase 4.

### D3.15 Findings: nonlinearity does not help, in forecasts or portfolios
- **Validation kept the models very simple:** LightGBM chose 1–136 trees (of 2,000 allowed); the MLP chose 1–27 epochs, mostly ≤ 8; the elastic net zeroed every coefficient in 4 of 11 years. All three point the same way: no reliable predictive benefit was detected under the tested conditions (these 13 features, these 51 stocks, 2016–2026).
- **Forecasts (Dec 2015 – Jul 2026):**

| | R²_oos vs 0 | vs hist. mean | Rank IC (NW t) | Quintile spread/month (NW t) | Hit rate |
|---|---|---|---|---|---|
| Historical mean | 1.72% | 0 | +0.018 (0.71) | +0.40% (0.69) | 56.4% |
| Elastic net | 2.01% | +0.30% | −0.013 (−0.55) | −0.14% (−0.29) | 56.5% |
| LightGBM | 2.01% | +0.30% | +0.005 (0.35) | +0.20% (0.73) | 56.4% |
| MLP | 1.53% | −0.19% | −0.007 (−0.50) | +0.07% (0.28) | 55.8% |

  Base rate of positive returns: 56.5%. No model ranks stocks better than chance; LightGBM vs elastic net IC difference t = 1.12, MLP vs elastic net t = −0.09.
- **Portfolios (S5, 10 bps):** Sharpe 0.69–0.76 (elastic net), 0.62–0.70 (LightGBM), 0.48–0.55 (MLP), against 0.85–0.95 for S4 and 0.96 for 1/N. Turnover a month after drift: 60–80%, 66–106% and 113–133%. Nonlinearity does not help the portfolio: S5 LightGBM − S5 elastic net = −0.02 (p = 0.89), MLP − elastic net = −0.17 (p = 0.23) at γ = 5.
- **Significance after Holm correction (20 tests):** only "S5 with the MLP is worse than 1/N" survives (adjusted p = 0.010–0.017 at all γ). Everything else, including every S5-vs-S4 comparison, is not significant.
- **Robustness:** conclusions unchanged with 3- or 12-month bootstrap blocks and with Newey–West t-statistics; trading one day later moves S5 Sharpe ratios by ≤ 0.03; daily drawdowns of S5 are 28–33%, similar to the baselines.
- **Permutation importance** differs across models; the one consistent message is that 1-month volatility hurt out-of-sample forecasts in all three.

### D3.16 Ω-B check: does disagreement track forecast errors?
- Within each month, stocks with more ensemble disagreement do have larger squared forecast errors: average monthly rank correlation 0.13–0.14 (t = 7.7–9.0). Pooled over time it is ≈ 0, because the overall level of disagreement moves with how much each model was allowed to learn, not with how hard the month was.
- Much of the within-month link is volatility: disagreement has rank correlation 0.42–0.51 with 3-month volatility, which itself predicts squared errors (0.22). After controlling for volatility a small link remains (partial correlation 0.03–0.05, t = 2.1–3.3).
- For Phase 4: Ω-B should give larger uncertainty to more volatile stocks and is mildly informative beyond that; Ω-A (past errors) is the natural comparison (plan 4.4).

### D3.17 What Phase 3 means for the project
- The ML forecasts are weak, so the main research question moves to Phase 4: whether Black–Litterman, with Ω set from forecast errors (Ω-A) or disagreement (Ω-B), turns weak, noisy forecasts into stable portfolios (S6 vs S5) without losing to the classical benchmarks (S6 vs S4, S1). With weak views BL should stay close to the market prior, which is the intended behaviour.

## Methodology review (3 Oct 2026)

### R1 Review of Phases 0–3 before reporting
Checked every method and claim for points an examiner could challenge. Changes made:
1. Paired Sharpe-difference tests replace the comparison of separate intervals (D2.11).
2. Holm correction for each family of Sharpe tests (D2.12, D3.15).
3. Bootstrap p-values at 3 / 6 / 12-month blocks (D2.12).
4. Newey–West t-statistics for IC and quintile spreads (D3.15).
5. Daily drawdowns reported; this **corrected a Phase 2 claim** about drawdowns (D2.10, D2.12).
6. One-day execution-lag robustness for every backtest (D2.12, D3.15).
7. MLP early stopping on chronological, not random, validation (D3.13).
8. Turnover vs target-weight change named explicitly everywhere (A2).
9. Planted-signal tests show each model can find a real signal, linear or nonlinear, so weak results come from the data, not the code (D3.13).
10. Ω-B checked against realized errors and volatility before it is used (D3.16).

Unchanged and already documented: survivorship bias (D0.2, approved A1), current share counts (D0.9), daily-to-monthly covariance scaling (D1.1), Lo's i.i.d. Sharpe interval for single strategies (D2.7), simple-average RSI (D3.2).

## Phase 4 — Black–Litterman integration (10 Oct 2026)

### D4.1 Headline configuration fixed in advance
- Headline S6 = elastic net views, Ω-A, τ = 0.05, γ = 5 (medium), written into `config.yaml` before any Phase 4 result was computed. Elastic net because it was the first, documented model (A2.3); Ω-A because it is the plan's main specification (Section 2.4); τ and γ are the plan's defaults. Every other combination is reported as an ablation, so the main conclusion cannot come from picking the best-looking variant.

### D4.2 Market weights for the prior
- w_mkt at t = close_t × shares outstanding, normalized over the 51 stocks. The close is split-adjusted but not dividend-adjusted (a true price level), added to the Phase 0 data as `close.parquet`; all earlier outputs are unchanged (largest difference 3e-15). Shares are the current all-class snapshot (D0.9, approved A1).

### D4.3 Prior: Π = δΣw_mkt with δ = 2.5
- Monthly Σ (same estimate the optimizer uses), so Π is a monthly excess return. A test confirms the defining property: with only the budget constraint and γ = δ, mean-variance at μ = Π returns exactly w_mkt.

### D4.4 Ω-A: past squared errors with a warm-up
- σ²ᵢ = (Σ of the last 24 squared errors + 12·varᵢ) / (nᵢ + 12), where varᵢ is the stock's monthly return variance (diagonal of Σ). Only errors already realized at t are used (forecasts made before t, whose month has ended).
- **Why the warm-up:** forecasts start in Dec 2015, so early in the backtest there are no past errors. varᵢ is the error variance of a forecast with no skill, a neutral starting value that fades as errors accumulate (half weight after 12 months).

### D4.5 Ω-B: raw (plan definition) and level-calibrated
- **Raw Ω-B** = variance of the 10 bootstrap copies' forecasts. Its median is about 0.05 × τσ², against about 19 × τσ² for Ω-A: the copies agree far more closely than the forecasts match reality. Disagreement measures how much the *estimate* depends on the sample, not how wrong the *forecast* is.
- **Calibrated Ω-B ("B-cal")** keeps the relative disagreement across stocks but rescales the level each month so its average equals Ω-A's. Reported as a variant so the comparison Ω-A vs Ω-B isn't decided by scale alone. A floor of 1e-12 keeps Ω positive when copies agree exactly.

### D4.6 Posterior computation and the covariance used in the optimizer
- μ_BL uses the update form Π + τΣ(τΣ + Ω)⁻¹(Q − Π), which equals the T6 precision form (tested, with the GLS form too) and stays stable as Ω → 0.
- The optimizer uses Σ, not the posterior predictive Σ + M of He & Litterman (2002). Since M ⪯ τΣ, Σ + M ⪯ 1.05Σ at τ = 0.05: at most the effect of a 5% higher γ. Using Σ keeps S4, S5 and S6 on the same risk model, so they differ only in μ.

### D4.7 Strategy set and tests
- S6 for every model × {Ω-A, Ω-B, Ω-B-cal} at γ = 5; "BL prior only" (μ = Π, the Ω → ∞ limit of Cor. 6.1) as a reference. Same constraints as S4 and S5.
- Sharpe tests: S6 vs S5, S4 and S1 for all nine S6 variants plus headline vs prior: one family of 28 tests, Holm-adjusted, with 3- and 12-month blocks as a check.
- Stability: paired test of the monthly turnover difference S6 − S5, Newey–West t.

### D4.8 Findings
- **BL makes ML portfolios far more stable.** Headline turnover 30% a month against 72% for S5 (difference −41.5 points, NW t = −5.3). With Ω-A the reduction holds for every model: LightGBM 83% → 32% (t = −9.0), MLP 125% → 52% (t = −20.2). Calibrated Ω-B gives similar reductions (t = −5.6 to −20.7).
- **Raw Ω-B does not stabilize anything:** turnover is unchanged (difference −3 to +3 points) because the views are treated as almost certain (μ_BL moves 92–99% of the way to the views).
- **Returns:** headline Sharpe 0.876 against 0.693 (S5), 0.876 (S4), 0.957 (1/N), 0.963 (prior only). After Holm correction no Sharpe difference in the family of 28 is significant (smallest adjusted p = 0.13). Before correction, BL improves the MLP portfolio (S6 Ω-A vs S5: +0.34, p = 0.011; Ω-B-cal: +0.36, p = 0.004), which does not survive the correction.
- **The views do not add value over the prior:** headline vs prior only −0.09 (p = 0.23). This is consistent with Phase 3: where no reliable predictive benefit was detected, the best BL can do is stay near the market prior, and it does better the closer it stays (D4.9).
- **Tail risk and drawdowns** of S6 (monthly CVaR 7.6%, daily drawdown 31.5%) are in the same range as the other strategies; no strategy is clearly worse in the tail.

### D4.9 Ablations (around the headline)
- **τ:** Sharpe falls and turnover rises steadily as τ grows (τ = 0.01: 0.94 and 19%; 0.25: 0.79 and 52%). **Ω scale:** ×4 gives 0.94 and 19%, ×0.25 gives 0.81 and 49%. Both say the same: the less weight on the views, the better, as expected when the views show no reliable predictive benefit.
- **γ:** S6 beats S5 at every γ (γ = 10: 0.85 vs 0.69; γ = 2: 1.00 vs 0.76). γ = 2 has the highest Sharpe of any strategy (1.00), but it is one of many ablations, not a pre-specified hypothesis, so it is not presented as a finding.
- **Costs:** the BL advantage over S5 grows with costs (headline vs S5 at 25 bps: 0.835 vs 0.592), because S5 trades so much more.
- **Models:** with Ω-A the three models end close together (0.86–0.89), because BL shrinks all of them toward the same prior.

### D4.10 Why the views still move μ_BL noticeably
- Diagonal Ω-A is about 19 × τσ², yet μ_BL moves about 50% of the way to the views across stocks (elastic net). The reason is that τΣ is far from diagonal: along the few market and sector directions with large eigenvalues the prior is uncertain, so views pass through there, while stock-specific differences are shrunk hard. A diagonal ratio understates the views' influence; the diagnostics table reports the actual movement.

## Advisor feedback (received 17 Oct 2026, on the Phase 4 report)

### A3 Comments and actions
- **Positive:** BL integration clearly explained; reduced turnover correctly separated from statistically unconfirmed performance gains; the Ω-A vs Ω-B comparison and the diagnosis and rescaling of bootstrap disagreement were noted as thoughtful; negative findings and sensitivity checks strengthen the work.
- **Wording:** replace "forecasts carry no real signal" with "no reliable predictive benefit was detected under the tested conditions." → Done in D3.11, D3.15, D4.8, D4.9 and in all reports from now on. The precise claim is about these features, this universe and this period, not about prediction in general.
- **Next-week priority:** one reproducible, tested example of the tool, showing how amount, risk tolerance and sector preferences determine an allocation that satisfies the constraints. → Phase 5, D5.x.
- **Reports:** explain not only which model is used but how it is set up and used (inputs, parameters, procedure). → From this week the weekly report includes a set-up section with every parameter.

## Phase 5 — Recommendation tool (17 Oct 2026)

### D5.1 One tested, reproducible example first (advisor A3)
- The tool is built as a command-line program (`python -m src.recommend`) plus six fixed worked examples in `config.yaml`, all covered by tests. A Streamlit interface is optional (the plan's first cut if time is short) and can sit on top of the same `recommend()` function later.

### D5.2 The model behind a recommendation
- The Phase 4 headline, unchanged: elastic-net forecasts → Black–Litterman (Ω-A, τ = 0.05, δ = 2.5, market-cap prior) → constrained mean-variance. No new model choice is made for the tool, so its behaviour is the one that was backtested.
- Forecasts are produced by re-running the Phase 3 walk-forward up to the recommendation date (same yearly retrain dates). A test confirms they equal the Phase 3 forecasts on every shared date. Runtime about 2 seconds.
- **Date:** the latest month-end in the data (31 Aug 2026) by default, or any earlier month-end with `--as-of`. Everything is read from data up to that date (tested by scrambling all later data). Prices are therefore those of the data end date, not today's: the tool is a simulation, as the plan states.

### D5.3 Risk tolerance
- low / medium / high → γ = 10 / 5 / 2 (plan 2.6). The output reports the resulting expected volatility so the choice has a concrete meaning: 8.4% / 10.2% / 15.2% a year in the worked examples B–D.

### D5.4 Sector preferences are constraints, not filters
- Exclude → sector bound (0, 0); maximum → (0, u); minimum → (l, 1). They enter the same optimization problem as the per-stock cap, so the solver guarantees them (T4 holds with the extra multipliers) and the rest of the portfolio is re-optimized around them, instead of deleting stocks afterwards and rescaling.
- Requests are validated before solving, with a message saying what to change: amount ≤ 0, unknown risk level or sector, bounds outside [0, 1] (not silently clipped), minimum > maximum (e.g. excluding a sector and also asking for a minimum in it), minimums summing to more than 100%, or too few allowed stocks to reach 100% under the 10% cap.

### D5.5 From weights to whole shares
- Price = close at the recommendation date (split-adjusted, not dividend-adjusted: the real share price).
- Step 1: round each holding down. Step 2a: if rounding left a sector below its minimum, buy shares in that sector first. Step 2b: buy single shares only where it brings the holding closer to its target (gap > price/2), choosing the largest improvement, within every stock and sector cap.
- **Rejected:** "spend the cash on whatever is most under-filled". The first version did this and bought one $944 COST share to fill a $236 gap, putting 9.4% in a stock targeted at 2.4%. Within the caps, but away from the optimizer's portfolio.
- **Rejected:** fractional shares. Not every broker offers them, and the plan asks for whole-share allocations with leftover cash.
- Weights of the holdings are measured against the full amount, so cash counts as unallocated; that is the stricter way to check minimums.

### D5.6 Worked examples (as of 31 Aug 2026; `phase5_examples.csv`)
| Scenario | γ | Holdings | Cash | Vol (target → held) | Max weight gap |
|---|---|---|---|---|---|
| A: $10k, medium, no preferences | 5 | 36 | $526 | 10.2% → 9.7% | 1.5 pts |
| B: $10k, medium, no Energy, IT ≤ 20% | 5 | 31 | $1,400 | 10.2% → 8.9% | 2.4 pts |
| C: $10k, low, as B | 10 | 31 | $129 | 8.4% → 8.2% | 3.8 pts |
| D: $10k, high, as B | 2 | 14 | $1,652 | 15.2% → 12.6% | 4.9 pts |
| E: $250k, medium, as B | 5 | 44 | $958 | 10.2% → 10.2% | 0.1 pts |
| F: $10k, medium, Health Care ≥ 15% | 5 | 31 | $870 | 10.2% → 9.1% | 2.4 pts |
- Every constraint holds in every scenario, for target weights and for whole shares.
- Preferences change the allocation as intended: in B, Energy goes from 4.9% to 0 and IT from 23.3% to exactly 20%, with the freed weight spread mainly into Consumer Discretionary, Consumer Staples and Industrials. The model's expected excess return barely changes (5.86% → 5.85% a year), so in this model these preferences cost almost nothing.

### D5.7 Limitation found: small amounts can't follow the target closely
- At $10,000, whole shares of high-priced stocks leave 1–17% in cash and holdings up to 4.9 points away from target; held volatility is below the target's because cash has none. At $250,000 the gap is 0.1 points and cash 0.4%. The output always shows target vs held weights and both volatilities, so the user sees the difference. Measured for scenario B's preferences: largest weight gap 2.40 / 1.38 / 0.75 / 0.44 / 0.14 points at $10k / $25k / $50k / $100k / $250k (cash 14.0% / 1.9% / 0.2% / 0.0% / 0.4%). So about $100,000 keeps every holding within half a point of its target. Fractional shares or a stated minimum amount would fix it; noted for the report's limitations.

### D5.8 Wording of model outputs
- The expected excess return is labelled "a model estimate, not a promise" in the output, given the Phase 3–4 finding that no reliable predictive benefit was detected under the tested conditions.

## Decisions for the week of 19 Oct 2026 (recorded 17 Oct, before any work on them)

### A4 Advisor: sectors are chosen for the user (it's a project, not an app)
- The tool will take only **amount** and **risk tolerance**. Sector constraints are set by us in `config.yaml`; the user has no sector input.
- **Rule chosen: no sector above 25% of the portfolio, no sector excluded**, on top of the 10% per-stock cap. Reasons: a standard diversification limit; it rarely binds (IT reaches 23.3% without it), so it changes the tested strategy as little as possible; it needs no view on which sectors are "good".
- **Consistency:** the same rule will be applied to every optimized strategy in the backtest (S3–S6) and Phases 2–4 re-run, so the tool still runs exactly the strategy that was evaluated. Implementation next week; the user-preference code (`--exclude`, `--max-sector`, `--min-sector`) and its tests will be removed or turned into fixed config.

### E1 Planned experiment: weekly rebalancing (specified before any weekly result)
- **Question:** do the conclusions hold at a weekly frequency, and do the forecast tests gain power from ~4× more cross-sections?
- **Why weekly, not daily:** the precision of Sharpe-ratio estimates depends on the length of the sample in years, not on the number of observations (Lo 2002: the annualized standard error ≈ √((1 + SR²/2)/years), frequency cancels), so higher frequency will not sharpen the portfolio tests. Daily targets are the noisiest and daily rebalancing the most expensive; the advisor's own guidance (meeting of 15 Oct) was to aggregate to lower frequency.
- **One change only:** rebalance and forecast weekly (last trading day of each week); target = next-week excess return. Everything else as in the monthly headline: same 13 feature definitions and windows in trading days, same universe, elastic net with the same α / l1-ratio grid, Black–Litterman with Ω-A, τ = 0.05, δ = 2.5, γ = 5, 10% stock cap, 10 bps costs. Horizon-dependent settings scaled to keep the same calendar length: retrain every 52 weeks; validation = last 104 weeks; Ω-A window 104 weeks with a 52-week warm-up; covariance from 252 trading days scaled ×5 to weekly; out-of-sample from the first week of 2016.
- **Reported:** the same forecast metrics (R²_oos, rank IC with Newey–West t, quintile spread) and portfolio metrics (annualized ×52), S1, S4, S5, S6 side by side with the monthly results.
- **Status:** a robustness experiment. **The monthly results stay the main results whatever the weekly ones show**, and the weekly ones are reported in full either way.

## Model search (designed 17 Oct 2026, before any search result)

### MS1 Purpose and rule against overfitting
- **Goal (student's request):** try different model families, settings and extra price-based signals, and find the setup per model that forecasts best, then feed the best **three** models into Black–Litterman. The original Phase 3–4 results stay as they are (they were pre-specified); this is an additional study.
- **Same forecasting method as before:** walk-forward. At each month-end the model is trained only on data available then and forecasts the next month (D3.5), retrained yearly with chronological validation.
- **Development vs holdout:** setups are compared **only on forecasts made Dec 2015 – Dec 2021** (73 months, the development period). Forecasts made Jan 2022 – Jul 2026 (55 months, the holdout) are **not used for any choice**; they are evaluated once, for the three finalists only. Every setup tried is logged, not just the winners.
- **Caveat:** the original headline's 2022–2026 results were seen before this search; the search procedure itself does not use those years.

### MS2 What is varied (one grid, fixed in config before running)
- **Model families (4):** elastic net, LightGBM, MLP (as in Phase 3) and random forest (new: averaging many deep trees, a common strong baseline for tabular data).
- **Training window (3):** expanding (as now), rolling 6 years, rolling 4 years (advisor's suggestion: older data may mislead). 4 years is the shortest that still leaves 24 tuning + 24 validation months.
- **Target (2):** raw next-month excess return (as now), or excess return minus that month's cross-sectional average (learn which stocks beat the others, which removes the market's ups and downs from the target). For Black–Litterman the relative forecast is turned back into a return level by adding the average past excess return of the universe (known at t).
- **Signals (2):** the 13 Phase 3 features, or those plus 5 extra price-based ones: last-week return, 3-month return skewness, 3-month illiquidity (Amihud: |return| / dollar volume), 12–1 momentum relative to the stock's sector, and short- vs long-term volatility ratio.
- 4 × 3 × 2 × 2 = **48 setups**. Inside each, hyperparameters are still tuned by the chronological validation (D3.5).

### MS3 Selection rule (fixed in advance)
- Score = mean monthly rank IC of the forecasts over the development period. Ranking is what Black–Litterman and the optimizer can use, and IC is less noisy than portfolio returns.
- Finalists: the best setup of each model family, then the **three families with the highest development score**.
- Then, for the three finalists only: holdout forecast metrics, and S6 (Black–Litterman, Ω-A, τ = 0.05, γ = 5) with Sharpe, turnover and drawdown reported separately for development and holdout, next to the original headline S6, the prior alone, S4 and 1/N.

### MS4 Search widened (17 Oct, after the first run was stopped; holdout still unused)
- At the student's request: **3 more model families**: Huber regression (linear, but down-weights extreme months, since returns have fat tails), extra trees (a random forest with random split points, less prone to fit noise) and k-nearest neighbours (forecast = average outcome of the most similar past stock-months). **More training windows:** expanding, rolling 8, 6, 5 and 4 years.
- Grid: 7 families × 5 windows × 2 targets × 2 signal sets = **140 setups**.
- The first run (48 setups) was stopped before any score was looked at and before the holdout stage; its saved forecasts are reused (identical code and settings).
- **Two stages, so the holdout is used once:** stage 1 scores every setup on the development period only; stage 2 (`--stage final`) evaluates the finalists on the holdout, and only runs when every setup has been searched.
- More setups make the best development score more optimistic purely by chance (the more you try, the better the luckiest one looks). The holdout result is the honest number; the development score of the winner is not.

### MS5 Consistency rule (student's suggestion; set while stage 1 was running, before any score was seen)
- The development period is cut into 3 blocks: forecasts made Dec 2015 – Nov 2017, Dec 2017 – Nov 2019, Dec 2019 – Dec 2021. Each setup's mean rank IC is computed in each block.
- **Only setups with a positive IC in all 3 blocks qualify.** Among qualifying setups, the best per family (by development IC) goes forward, then the top 3 families. A setup that is good in one lucky stretch is filtered out.
- The holdout is still used once, for the finalists, and their holdout IC is also shown **year by year** to check it holds in every period. It is never used to choose.
- If fewer than 3 families have a consistent setup, that is reported as a finding, not worked around.

### MS6 Round 1 result and round 2 (17 Oct; holdout still unused)
- **Round 1 (140 setups, development only):** 12 passed the 3-block rule (KNN 8, LightGBM 2, MLP 2; none for elastic net, Huber, random forest or extra trees). Median development IC 0.002, best 0.040. **Luck check:** a setup with no skill has a 1-in-8 chance of being positive in 3 blocks, so about 17 of 140 would pass by chance; 12 passed. The round-1 winners are therefore not distinguishable from luck yet; the holdout will decide. The extended signals almost never passed (1 of 70 setups vs 11 of 70 for the base signals).
- **Round 2 (student's request: keep testing, upper-undergraduate models):** 3 more families: principal-component regression (compress the signals into a few combined factors, then a straight-line fit), histogram gradient boosting (scikit-learn's boosted trees, a second implementation of the boosting idea) and kernel ridge regression (a smooth nonlinear model; RBF kernel with a Nyström approximation so it runs in seconds). Plus an **equal-weight average of all family forecasts** for each window/target/signal combination: forecast combination usually reduces noise, and with fixed equal weights nothing is fitted, so it adds no selection bias of its own.
- Same windows, targets and signal sets: 3 × 20 + 20 averages = 80 more setups (220 in total). Same selection rule (MS3 + MS5) over all 220; the holdout is still run once, after round 2.

### MS7 Round 3: broad search with a placebo test (designed 17 Oct, before round 2 results were seen)
- **Student's request:** try as many upper-undergraduate models, signals and settings as possible, including niche ones.
- **New "niche" signal set** (all from daily price/volume/OHLC data up to t; most from the asset-pricing literature): seasonality (average return of the same calendar month in up to 5 past years; Heston & Sadka 2008), overnight and intraday returns over 1 month (Lou, Polk & Skouras 2019), high–low range volatility (Parkinson 1980), price vs 200-day average (trend), momentum acceleration (last 6 months vs the 6 before), idiosyncratic 12–1 momentum (residual vs SPY), share of positive days over 3 months ("frog in the pan", Da, Gurun & Warachka 2014), volume shock (5-day vs 3-month volume), downside beta (Ang, Chen & Xing 2006), co-skewness with the market, sector momentum (Moskowitz & Grinblatt 1999), within-sector 1-month reversal; plus three **market-state** inputs, the same for every stock in a month and left unranked so tree models can use them as context: market volatility (1 month), T-bill level, market 12-month return. Niche set = base 13 + extended 5 + these 16 = 34 inputs.
- **New model styles:** logistic regression and LightGBM **classifiers** that predict whether a stock will beat the month's median (forecast = past average return + a scale fitted on the training data × (probability − ½)), and linear support-vector regression.
- **Wider settings** for the existing families are not added, to keep the run feasible; each family keeps its yearly tuning grid.
- **Placebo test:** the same search, run for the fast families, on data where each month's targets are randomly shuffled across stocks (seed fixed). This keeps the data's structure but removes any real link between signals and returns. The best development IC and the number of 3-block passes in the placebo show what luck alone produces with this many tries; a real finding must clearly exceed them.
- Same selection rule (MS3 + MS5) over every setup of rounds 1–3; holdout once, at the end.

### MS8 Round 4: deeper settings for the families that ever passed (17 Oct; holdout unused)
- **Single-signal check (development only):** of 31 stock-level signals, one has |t| > 2 (within-sector 1-month reversal, IC −0.034, t −2.2); luck alone gives 1–2 of 31. Risk-type signals (1-year beta, downside beta, 1-year max drawdown, volatility, high–low range) keep the same sign in all three development blocks.
- **Round 4 (student's request: keep testing):** the 5 families with at least one consistent setup in rounds 1–2 (KNN, MLP, LightGBM, hist gradient boosting, kernel ridge), with **wider tuning grids**, **retraining every 6 or 12 months**, and three signal sets: base, niche, and a new **"risk" set** chosen from the single-signal check (beta, downside beta, vol 1m/3m, idiosyncratic vol, range vol, max drawdown, distance from 52-week high, within-sector reversal, overnight return).
- **Caveat:** the "risk" set was picked by looking at development data, so its development scores are optimistic (the same years choose the signals and score the models). Only the holdout can tell; this is noted next to every result that uses it.
- 5 families × 5 windows × 2 targets × 3 signal sets × 2 retrain frequencies = 300 setups. Same selection rule; holdout still once, at the end.

### MS9 Families 14 and 15 (17 Oct, student's request; holdout unused)
- **Quantile (median) regression:** a linear model fitted to the median instead of the mean, so extreme months have little pull. **Gaussian naive Bayes classifier:** predicts "beats the month's median" from per-class signal distributions, assuming signals independent; very different assumptions from the other classifiers. Both run over all windows, targets and signal sets (60 setups).
- **Not added: reinforcement learning.** It learns a trading policy by trial and error and needs far more independent episodes than ~190 months provide, so it would mainly fit noise; it is beyond the agreed upper-undergraduate scope and the plan excludes deep models beyond the small MLP.
- **Deeper MLP (student's request):** the round-4 MLP grid also includes 4-hidden-layer networks, (128, 64, 32, 16) and (64, 64, 64, 64). Early stopping on the chronological validation months limits overfitting; with ~9,600 noisy rows a deeper network is not expected to help, but it is tested rather than assumed.

### MS10 Planned bonus stress test: the 2008–2009 crisis (17 Oct, student's request)
- After all search rounds (and weekly), the finalist setups will also be tested on the global financial crisis: download 2000–2009 data separately (own folder; main data untouched), retrain each finalist setup from scratch using only 2002–2007 data with the same walk-forward procedure, and test it month by month over 2008–2009, next to 1/N, SPY and the prior-only Black–Litterman portfolio.
- **Status:** a bonus stress test, in addition to the 2022–2026 holdout, not a replacement for it.
- **Caveats to report:** (1) the setups were chosen on 2016–2021, a later period, so this tests whether the chosen design generalizes to a different era, not a fully blind choice; (2) survivorship bias is stronger in 2008: all 51 stocks survived to 2026 (e.g. JPM is in, Lehman Brothers is not), so every strategy will look better than it would have in reality.

### WU1 Wide-universe forecasting test (designed 17 Oct, before any wide-universe data was downloaded)
- **Student's request:** test the models on a broader, more varied set of stocks (large, mid and small caps; high and low momentum; value), not only 51 large caps.
- **Universe:** current members of the S&P 500, S&P 400 (mid caps) and S&P 600 (small caps), from Wikipedia on 17 Oct 2026 (~1,500 stocks, with GICS sectors), keeping those with complete daily data from Jan 2010 to Aug 2026 under the Phase 0 cleaning rules. Stored separately in `data/wide/`; the main 51-stock data are untouched.
- **What is NOT possible with free data:** book value or other accounting signals. Yahoo provides only ~4 recent years of balance sheets, so a value signal for 2011–2026 would use future information. Value is therefore excluded and stated as a limitation. High/low momentum needs no separate list: it is a signal, and the wide universe contains both kinds of stock.
- **Survivorship bias is stronger** than for the 51 stocks: only companies in the indices today are included, and many small caps that failed or were delisted since 2010 are missing.
- **Scope (compute):** about 30× more rows, so a focused grid: all 15 families × expanding window × both targets × base and niche signals = 60 setups, same walk-forward, same development/holdout split and 3-block rule. Tree ensembles use 100 trees and 30% row subsamples to keep run times to minutes.
- **Reported:** rank IC over the whole wide universe, by size group (S&P 500 / 400 / 600 members), and on the 51 original stocks only (does training on more stocks help forecast our 51?).
- **Portfolio part later, only if forecasts show anything:** with ~1,500 stocks the sample covariance from 252 days is singular, so a portfolio would need Ledoit–Wolf shrinkage (plan item O2).

### WU2 Placebo for the wide universe (18 Oct)
- Same as MS7 for the ~1,050 stocks: the 9 fast families on targets shuffled within each month (36 setups), to show what scores luck alone produces at this size. The wide-universe results are judged against it.

### WU3 Quantile regression on the wide universe uses a 25,000-row subsample (18 Oct)
- Its exact solver (an LP over every row) did not finish one setup in 27 minutes on ~200,000 rows. On the wide universe only, each training set is randomly subsampled to 25,000 rows (seeded) before fitting; prediction is unchanged. The 51-stock search is unaffected (never more than ~9,600 rows).

### MS11 Results of the full development search (18 Oct; holdout still unused)
- **51 stocks, 780 setups (rounds 1–4):** 84 passed the 3-block rule (11%; placebo 20%). Best development IC 0.058 vs placebo best 0.041.
- **Round 4's "risk" signal set produced the only setups clearly above the placebo:** kernel ridge 0.058, KNN 0.054, LightGBM 0.053, each positive in all three blocks (lowest block 0.028–0.050). 30% of risk-set setups passed vs 20% in the placebo. **Caveat (MS8):** the risk set was chosen from development-period single-signal results, so these development scores are optimistic; only the holdout can confirm them.
- **Finalists by the pre-set rule (MS3 + MS5):** kernel ridge (rolling 4y, raw, risk set), KNN (rolling 4y, demeaned, risk set), LightGBM (rolling 4y, demeaned, risk set), all retrained yearly with the wide grids.
- **Wide universe (60 setups, 1,052 stocks):** no setup reaches t > 2 (best 1.56). But the *typical* setup does better than the placebo: median IC 0.006 vs −0.001 overall, and 0.019 vs 0.001 for mid caps, so a small, broad signal in mid/small caps is likely, while no single setup is clearly reliable. The placebo's best t is 2.15, showing that one t > 2 among many tries is expected by luck.
- **Next (student to decide):** weekly experiment (E1), then the single holdout run of the three finalists (`--stage final`), then the 2008 stress test (MS10).

### MS12 Holdout result (22 Oct 2026; run once, after all development searching)
- The weekly experiment (E1) was not run first, at the student's choice; it can no longer influence the choice of finalists.
- **Forecasts made Jan 2022 – Jul 2026, finalists chosen on 2016–2021:**

| Finalist | Dev IC (NW t) | Holdout IC (NW t) | Holdout IC by year 2022 / 23 / 24 / 25 / 26 |
|---|---|---|---|
| Kernel ridge, roll 4y, raw, risk set | 0.058 (1.96) | 0.014 (0.48) | −0.058 / 0.010 / 0.019 / 0.107 / −0.023 |
| KNN, roll 4y, demeaned, risk set | 0.054 (2.10) | 0.021 (0.73) | −0.043 / 0.010 / 0.057 / 0.105 / −0.053 |
| LightGBM, roll 4y, demeaned, risk set | 0.053 (2.52) | 0.002 (0.07) | −0.062 / −0.067 / 0.041 / 0.101 / −0.007 |

  All three have holdout R²_oos below the historical-mean baseline.
- **Portfolios (S6, γ = 5, 10 bps), holdout Sharpe:** kernel ridge 0.54, KNN 0.53, LightGBM 0.53, original Phase 4 S6 0.50, 1/N 0.53, S4 0.30, **BL prior only 0.63**.
- **Conclusion:** after ~900 setups (15 families, 5 windows, 2 targets, 4 signal sets, wide grids, 2 universes), no reliable predictive benefit was detected under the tested conditions. The finalists' development scores were inflated by the number of setups tried and by choosing the risk signal set on the same data (MS8), exactly as the placebo and the design anticipated; the holdout removed the inflation. The main conclusion of Phase 4 is unchanged: Black–Litterman protects the portfolio, and with views that show no reliable predictive benefit, staying near the market prior does best.

### MS13 2008–2009 crisis stress test result (22 Oct 2026)
- 50 of the 51 stocks (GOOGL listed in 2004); models retrained by walk-forward on data from 2001; decisions Dec 2007 – Nov 2009, holding Jan 2008 – Dec 2009; γ = 5, 10 bps.
- **Forecasts during the crisis:** rank IC original elastic net −0.091 (t −1.5), kernel ridge +0.036, KNN +0.002, LightGBM +0.013 (all |t| < 0.6). No reliable predictive benefit in the crisis either.
- **Portfolios (total return / largest daily-path drawdown):** BL with any of the four models −15% / 37.5–37.8%; BL prior only −15% / 40.0%; SPY −20% / 52.3%; Markowitz −23% / 47.2%; 1/N −3% / 45.8%.
- **Reading:** the Black–Litterman portfolios cut the worst fall by about 14 points against SPY and 8 against 1/N; the ML views change almost nothing relative to the prior. 1/N's small total loss is driven by the 2009 rebound of stocks known to have survived (e.g. BAC, WFC): survivorship bias is strongest exactly here, so 1/N's crisis return should not be read as a real advantage.
- Outputs: `report/experiments/crisis_results.csv`, `crisis_wealth.png`.

### WU4 Wide-universe portfolio experiment (designed 22 Oct, before any wide portfolio result)
- **Purpose:** decide, with evidence, whether to move the main project from 51 to ~1,050 stocks.
- **Universe:** the 1,052 wide-universe stocks (WU1), minus any without a share count for market caps.
- **Risk matrix:** Ledoit–Wolf shrinkage of the trailing 252-day daily covariance (shrinks toward a scaled identity), × 21 to monthly. Needed because the sample covariance of 1,052 stocks from 252 days is singular; Ledoit–Wolf is always positive definite (plan item O2).
- **Weight cap 2% per stock** (instead of 10%): with 1,052 stocks, a 10% cap would allow a 10-stock portfolio, which defeats the point; 2% forces at least 50 holdings.
- **Strategies, monthly, Dec 2015 – Jul 2026, 10 bps, γ = 5:** 1/N over all stocks; SPY; minimum variance (Ledoit–Wolf); Black–Litterman prior only; Black–Litterman with Ω-A and views from the wide-search setup chosen by the pre-set rule (MS3 + MS5 on development scores only).
- **Reported:** Sharpe, return, volatility, drawdown, turnover, for the development (2016–2021) and holdout (2022–2026) parts separately, next to the 51-stock results.
- **Decision rule for the student:** propose switching to the advisor only if the wide BL portfolios clearly beat their 51-stock counterparts in both parts.

### MS14 IPCA added after the holdout run (22 Oct; student's request after the literature search)
- **Model:** instrumented PCA (Kelly, Pruitt & Su 2019): r_{i,t+1} = z_{i,t}ᵀ Γ f_{t+1} + ε, where z are the stock's (ranked) signals plus a constant, Γ (L × K) maps signals to K factor loadings, and f are latent factor returns. Estimated by alternating least squares (Γ given f, f given Γ), written from the paper rather than taken from the `ipca` package (keeps the environment unchanged). Forecast for t+1 = z_{i,t}ᵀ Γ λ, with λ the average estimated factor return over the training window (the usual predictive use).
- **Tuning:** K ∈ {1, 2, 3, 5} by validation MSE on the chronological validation months, like every other model; same windows, targets and signal sets as the main grid (30 setups).
- **Status:** because the holdout was already used once (MS12), IPCA's holdout result is reported in full but flagged as obtained after that look; it cannot replace the MS12 conclusion on its own.
- **IPCA result (40 setups: 5 windows × 2 targets × 4 signal sets):** 21 of 40 passed the 3-block rule (52%, against 11% for the main search and 20% for the placebo), the most consistent family tested. Chosen on development data: rolling 5 years, raw target, risk signal set (dev IC ≈ 0.04). **Holdout IC 0.033 (NW t 0.96)**, by year 2022 −0.055, 2023 0.022, 2024 0.079, 2025 0.085, 2026 0.033: the best holdout forecast so far, but not statistically reliable. S6 with IPCA views: Sharpe 1.18 (development), 0.51 (holdout), max drawdown 20%: no better than the other S6 variants or 1/N in the holdout.

### MS15 Round 5: strategies from the literature, beyond large caps (designed 22 Oct, before any result)
Student's request: go beyond large caps, try ML *strategies* (not only models), run the ideas from the literature search, and put everything through the same tests.

**Universes:** (a) the 51 large caps; (b) the 1,052-stock wide universe; (c) mid + small caps only (S&P 400 and 600 members of the wide universe).

**Strategies:**
1. **Learning to rank (LambdaMART; Poh, Lim, Zohren & Roberts 2021):** LightGBM's `lambdarank` objective trained to order stocks within each month (labels = next-month return quintile). The ranking score is turned into a return forecast for Black–Litterman as a + k·score, with a and k fitted on the training data only. Grid: windows {expanding, 5 years}, signal sets {base, niche, risk}, all three universes; plus a placebo on shuffled targets.
2. **Industry rotation (Rapach, Strauss, Tu & Zhou 2019):** forecast next-month returns of the 11 GICS sector portfolios (equal-weighted, built from the wide universe) from last month's returns of all 11 sectors, with elastic net, tuned on chronological validation, walk-forward. Views enter Black–Litterman as **sector views** (P has one row per sector, averaging its stocks), using the general-P form of T6, on the 51-stock portfolio. Placebo: same with shuffled sector returns.
3. **ML volatility forecasting (Christensen, Siggaard & Veliyev 2023; baseline per "HARd to Beat" 2024):** forecast each stock's next-month volatility from its past volatilities (1 week, 1 month, 3 months, 1 year), high–low range volatility and market volatility, with elastic net and LightGBM, against the HAR-style linear baseline and plain trailing 3-month volatility. Used to build Σ = D·R·D (D = forecast volatilities, R = trailing correlation). Tested by forecast accuracy and by the realized volatility, drawdown and risk-target accuracy of the minimum-variance and Black–Litterman portfolios.
4. **Wide-universe portfolio (WU4)** on universes (b) and (c) with Ledoit–Wolf.

**Tests (same as before):** choice on development data only (2016–2021, 3-block rule); placebo where a forecast is learned; holdout (2022–2026) reported in full but flagged **post-holdout**, since that period has already been looked at (MS12, MS14); Black–Litterman portfolio with Ω-A, γ = 5, 10 bps; the 2008–2009 crisis test for the large-cap strategies.
- **Result, strategy 2 (industry rotation, 11 sectors, elastic net per sector):** development rank IC 0.097 (NW t 2.27), positive in all three blocks (0.110, 0.169, 0.017), against −0.031 for the placebo: the first development result above t = 2. **Holdout (post-holdout): −0.026 (t −0.67).** R²_oos against the historical mean −8.6% (development), +1.5% (holdout); top-3 minus bottom-3 sectors +0.01% a month (development), −0.47% (holdout). Black–Litterman with sector views on the 51 stocks: Sharpe 1.16 / 0.65 (development / holdout) vs 1.23 / 0.63 for the prior alone and 1.29 / 0.53 for 1/N, with higher turnover (37% vs 24%). Conclusion: the Rapach et al. (2019) effect appears in 2016–2021 but does not hold up in 2022–2026; no reliable benefit.
- **Result, strategy 3 (ML volatility forecasting, 51 stocks):** forecast error of next-month log volatility (MSE), development / holdout: trailing 63-day 0.194 / 0.119, HAR 0.157 / 0.108, **elastic net 0.149 / 0.097**, LightGBM 0.166 / 0.101. Diebold–Mariano (monthly loss difference, NW t): elastic net beats trailing volatility (t −2.70 / −2.33) and HAR (t −4.60 / −6.13) in both periods. In the 2008–2009 crisis it is also the most accurate (0.138 vs 0.142 HAR, 0.161 naive), but not significantly over 24 months (t −1.35 / −0.54). **This is the first ML result that holds out of sample.**
- **Use for the risk level promised to a user:** for the same minimum-variance portfolio, predicted vs realized next-month volatility (log error): sample Σ bias +0.08, RMSE 0.47; Σ = D·R·D with elastic-net volatilities bias +0.01, RMSE 0.40 (2016–2026); in the crisis RMSE 0.70 → 0.49.
- **Not a better minimum-variance portfolio:** realized volatility 12.5% (D·R·D) vs 11.9% (sample Σ), with about twice the turnover (56% vs 25%); in the crisis 16.1% vs 16.8% and drawdown 33.9% vs 35.5%. Interpretation: better volatilities alone do not beat the sample covariance for choosing weights, but they do make the stated risk of a chosen portfolio more accurate.
- **Result, strategy 5 (Black–Litterman with the ML risk matrix, Σ = D·R·D with elastic-net volatilities; MS15.5):** 2016–2026 (holdout post-holdout), Sharpe development / holdout: prior with sample Σ 1.23 / 0.63, prior with ML Σ 1.21 / 0.53, prior + elastic-net views with sample Σ 1.16 / 0.50, with ML Σ 1.16 / 0.53; turnover roughly doubles with ML Σ (24% → 49%). 2008–2009 crisis (50 stocks): largest drawdown prior 40.0% → 38.0% with ML Σ, with views 37.5% → 36.5%; crisis Sharpe −0.40 → −0.28. Conclusion: the ML risk matrix does not improve normal-period performance but gives modestly smaller losses in the crisis; its main value is the more accurate stated risk level (strategy 3).
- **Result, strategy 1 (learning to rank, LambdaMART):** chosen setups (development, 3-block rule) and holdout IC (post-holdout): large 51, rolling 5y, base signals: 0.013 (t 0.45); wide universe, expanding, base: 0.004 (t 0.23); mid + small, expanding, base: −0.004 (t −0.28). S6 with LTR views (51 stocks): Sharpe 1.01 / 0.52 (development / holdout), drawdown 20.8% / 23.5%. No reliable benefit in any universe.
- **Result, strategy 4 (wide-universe portfolios, Ledoit–Wolf, 2% cap; WU4):** Sharpe development / holdout (post-holdout), largest holdout drawdown: wide (1,047 stocks) — 1/N 0.95 / 0.56, minimum variance 0.95 / 0.59, **BL prior only 0.94 / 0.70 (14.9%)**, BL + ML views 0.89 / 0.26 (turnover 109% a month); mid + small (630) — 1/N 0.85 / 0.51, minimum variance 0.81 / 0.41, BL prior 0.70 / 0.40, BL + views 0.79 / 0.16; SPY 1.02 / 0.66. For comparison, BL prior on the 51 stocks: 1.23 / 0.63.
- **Decision rule (WU4) applied:** the wide-universe BL portfolio is better than the 51-stock one in the holdout (0.70 vs 0.63) but worse in development (0.94 vs 1.23), so it does not clearly beat it in both parts. **Recommendation: keep the 51-stock universe for the main project** and report the wide-universe results as an extension. ML return views on the wide universe hurt the portfolio badly (Sharpe 0.26), consistent with every other return-view result.

### MS16 Volatility targeting with the ML risk forecast (designed 22 Oct, before any result)
- **Idea (Moreira & Muir 2017; critiques: Cederburg et al., Barroso & Detzel):** scale the portfolio's stock exposure each month so its predicted volatility equals a target; the rest sits in T-bills. Turns a better risk forecast into a possible return benefit.
- **Base portfolio:** Black–Litterman prior only (sample Σ), the best normal-period portfolio (51 stocks, γ = 5).
- **Exposure:** s_t = min(1, target / σ̂_t), no leverage (plan: no borrowing); target = 10% a year (the medium-risk level). σ̂_t = the portfolio's predicted next-month volatility from (a) the sample Σ, (b) the ML Σ = D·R·D with elastic-net volatilities, (c) last month's realized portfolio volatility (the original Moreira–Muir rule). Plus the unscaled portfolio.
- **Costs:** 10 bps on the stock trades scaled by s, plus on the change in exposure |s_t − s_{t−1}|. Cash earns the T-bill rate.
- **Reported:** Sharpe, volatility, drawdown, average exposure, and how close realized volatility stays to the 10% target, for development and holdout (post-holdout).
- **Result (MS16):** Sharpe development / holdout (post-holdout), holdout drawdown, average gap between 12-month realized volatility and the 10% target (dev / hold): no scaling 1.22 / 0.63, 19.4%, 4.1% / 2.4%; target with sample Σ 1.07 / 0.61, 14.9%, 2.8% / 1.6%; **target with ML Σ 1.12 / 0.68, 15.7%, 2.5% / 1.4%**; target with last month's realized volatility 1.20 / 0.66, 13.0%, 2.0% / 1.0%. Average stock exposure 82–89%.
- **Reading:** the ML risk forecast makes volatility targeting better than targeting with the sample Σ in both periods, and gives the highest holdout Sharpe of all strategies tested (0.68), with smaller drawdowns than no scaling. But the simple last-month rule is as good or better, unscaled is best in development, and the differences are small and not yet tested for significance. The defensible ML contribution is risk: accurate risk forecasts that keep a user near the chosen risk level with smaller losses.
- **Significance (MS16):** paired Sharpe-difference tests, ML-Σ targeting minus each alternative (block bootstrap p / Memmel p): vs no scaling dev −0.10 (0.26 / 0.45), hold +0.04 (0.49 / 0.52); vs sample-Σ targeting dev +0.05 (0.18 / 0.37), hold +0.06 (0.27 / 0.27), whole period +0.05 (0.13 / 0.19); vs last-month rule dev −0.08, hold +0.02 (all p > 0.16). Holm-adjusted p = 1.0 for every comparison. **None of the return differences is significant.** What remains significant is the volatility-forecast accuracy itself (strategy 3) and the closer tracking of the risk target.

### WU5 Black–Litterman with every model's views on the wide universe, and up/down models (designed 22 Oct, before any result)
- **Student's request:** use the bigger universe, run Black–Litterman with all the models tested before, and check whether up/down (classification) models help the project. Experiment only; decision the same evening.
- **Universe and settings:** as WU4 (1,047 wide-universe stocks with a share count, Ledoit–Wolf Σ, 2% cap, γ = 5, τ = 0.05, Ω-A, 10 bps, Dec 2015 – Jul 2026).
- **Views:** for each of the 15 families, the wide-search setup with the best development rank IC (development data only; no 3-block filter so that every family is represented); the average of those 15; and one new model, a logistic classifier of "excess return > 0" (plain up/down, `ClassifierModel(label="sign")`), base signals, raw target, tuned and trained like every other model. Probabilities become return views as for the other classifiers: a + k·(p − ½), fitted on training data only.
- **Up/down group:** the three "beats the month's median" classifiers (logistic, LightGBM, naive Bayes) and the new plain up/down logistic.
- **Reported:** return, volatility, Sharpe, drawdown, CVaR 95%, turnover for development (2016–21) and holdout (2022–26, post-holdout); up/down accuracy of each model's views (absolute and relative to the month's median) against always guessing "up"; paired block-bootstrap Sharpe tests of every BL + views portfolio against BL prior only, Holm-adjusted over all 34 tests.
- **Decision rule:** a model (or the up/down group) "helps" only if its BL portfolio beats BL prior only in both periods and the holdout difference is significant after Holm; otherwise it is reported as an extension and the main project is unchanged.

### WU6 IPCA and industry rotation on the wide universe (designed 22 Oct, before any result)
- **Student's request:** test the two most promising return feeds (industry rotation: best Black–Litterman return in 2022–26 on the 51 stocks; IPCA: most consistent forecaster) on a bigger universe, "1,300 or so".
- **Universe size:** 1,047 stocks, as WU4/WU5. Only 1,055 of the 1,503 S&P 1500 candidates have prices from Jan 2010; reaching 1,300 would need stocks listed only from 2017–19, which the backtest engine does not support (it assumes every stock is priced over the whole sample) and which would worsen survivorship bias. Not done.
- **IPCA:** windows {expanding, 5 years} × targets {raw, demeaned} × signal sets {base, niche, risk} = 12 setups on the wide panels; K ∈ {1, 2, 3, 5} tuned on validation MSE; choice = 3-block rule, then best development IC (MS14 rule).
- **Industry rotation:** the MS15.2 sector forecasts (already built from the wide universe) as sector views on all 1,047 stocks.
- **Settings, outputs, tests and decision rule:** as WU5.
- **WU5 result (1,047 stocks, post-holdout for 2022–26):** Sharpe development / holdout: BL prior only 0.94 / 0.70 (return 13.6% / 12.3%); SPY 1.02 / 0.66; 1/N 0.95 / 0.56. Every BL + views portfolio is worse than the prior in both periods: best MLP 0.89 / 0.26 (19.9% / 7.2%), KNN 0.71 / 0.30, random forest 0.67 / 0.33; the rest 0.60–0.72 / 0.09–0.29. Turnover 56–111% a month vs 43% for the prior. All 34 Sharpe differences are negative; none significant after Holm (smallest adjusted p 0.67; unadjusted p 0.02–0.05 for the whole period for most). Up/down accuracy of the views in 2022–26: 50–51%, the same as always guessing "up" (51.1%); relative to the month's median 49–50%.
- **Up/down models (WU5):** the three "beats the median" classifiers (holdout Sharpe 0.09–0.18) and the plain up/down logistic (0.21) are no better than the regressors. The plain up/down model predicts "up" for almost every stock (training accuracy 55.6% = share of up months): ranked signals compare stocks with each other and carry no information about the market's direction. **Conclusion: up/down models do not help; not added.**
- **WU6 result:** IPCA on the wide panels: development IC negative for 11 of 12 setups, none passes the 3-block rule (chosen by best development IC anyway: expanding, raw, niche). BL + IPCA 0.56 / 0.73 (return 10.4% / 12.3%), up/down accuracy 50–54%. BL + industry rotation (sector views) 0.61 / 0.68 (10.0% / 12.7%). Both below the prior over the whole period (Sharpe −0.24 and −0.21; unadjusted p 0.08 and 0.05, Holm 0.23 and 0.21).
- **Decision (WU5 + WU6 rule):** no model, the up/down group, IPCA or industry rotation beats BL prior only on the wide universe. Main project unchanged: 51 stocks, BL with the pre-registered elastic-net views (Phase 4). Wide-universe results go in the report as an extension: on 1,047 mostly mid- and small-cap stocks the return views add turnover and noise, and the Ω-A weighting does not shrink them enough to protect the portfolio.

### WU7 Combined views + ML risk matrix (designed 22 Oct, before any result; chosen after the holdout was seen)
- **Student's proposal:** Black–Litterman fed by industry-rotation sector views and LightGBM stock views (maybe also IPCA), with the ML risk matrix as the "AI for risk" part.
- **Method:** all view sets stacked in one pick matrix P (sector rows, then an identity block per stock-level model), Q stacked, Ω diagonal with each view's own Ω-A; general-P posterior of T6. LightGBM = the Phase 3 model (in the plan; not the search finalist, to avoid picking by results). IPCA = the MS14 setup. Σ = sample, or D·R·D with elastic-net volatilities (MS15.3). 51 stocks, γ = 5, τ = 0.05, 10 bps.
- **Variants (fixed in advance):** prior only (sample Σ / ML Σ); rotation only; LightGBM only; rotation + LightGBM; rotation + LightGBM with ML Σ (the proposal); rotation + LightGBM + IPCA with ML Σ. Checks: "rotation only" and "LightGBM only" must reproduce MS15.2 and Phase 4.
- **Caveat:** the components were picked after the 2022–26 results were known, so 2022–26 is post-holdout and cannot confirm the combination; paired tests against the prior, Holm-adjusted, are reported for all periods.
- **WU7 result (51 stocks; 2022–26 post-holdout):** checks pass (rotation only 1.164 / 0.648, LightGBM only 1.189 / 0.498, as before). Return a year development / holdout / all, $10k → end, Sharpe dev / hold, turnover: prior 16.5 / 12.0 / 14.6%, $42.6k, 1.23 / 0.63, 24%; rotation + LightGBM 17.2 / 10.1 / 14.1%, $41.0k, 1.18 / 0.50, 40%; **rotation + LightGBM with ML Σ 18.1 / 11.3 / 15.1%, $44.9k, 1.22 / 0.57, 62%**; + IPCA 19.4 / 10.6 / 15.5%, $46.5k, 1.26 / 0.51, 66%. For reference 1/N 15.8%, $48.0k. No Sharpe difference against the prior is significant (all Holm p = 1.0; smallest raw p 0.09). The extra money of the combined versions comes from 2016–21 and from taking a little more risk; in 2022–26 every combination earns less than the prior.
- **Reading:** the combination is a coherent use of T6 (sector and stock views in one P) and of the ML risk forecast, and it ends with more money than the prior, but it is not a statistically better portfolio and was assembled after seeing the results. Report it as an extension with that caveat; the main model stays as pre-registered.

### BL1 Better use of ML forecasts inside Black–Litterman (designed 22 Oct, before any result)
- **Student's request:** instead of more forecasting models, find ML/AI ideas from the literature that improve Black–Litterman itself, and test them with the same tests.
- **Literature found:** Grinold & Kahn's "alpha = IC × volatility × score" (views from rankings, scaled by measured skill); ranking-based views (Yao, Li & Li 2024, LSTM + BL); confidence-gated views and entropy-based Ω (arXiv 2607.20533, 2026, theoretical only); views whose confidence is set by a tracking-error target (Shu & Mulvey 2024, regime-switching factor views; "active risk targeting" for τ); latent-variable Bayesian BL (ICML 2025; not implementable from the abstract alone, not tested). Market-direction views (time-series momentum) not tested here: with a fully invested long-only portfolio, a market view barely changes the weights.
- **Methods (src/experiment_bl.py):** (1) IC-scaled views: Q = Π + IC·σ·z, z = normal scores of the forecast ranks, σ from Σ, IC = trailing 36-month rank IC shrunk with 12 pseudo-months, floored at 0; Ω = τ·diag(Σ) (He–Litterman). (2) Strong-signal views: (1) only for the top and bottom 20% of z. (3) One relative view: top 20% minus bottom 20% by forecast; Q = mean realized spread of the model's past rankings (36 months), Ω = its squared standard error; no view before 12 months of history. (4) Tracking-error-targeted confidence: Ω rescaled each month (bisection on log scale, 10⁻³–10⁴) so the BL portfolio's predicted tracking error against the prior-only portfolio is 2% a year; applied to Phase 4 Ω-A views and to (1).
- **Forecasts:** Phase 3 elastic net (main model) and LightGBM. **Placebo:** elastic-net forecasts shuffled across stocks each month (seed fixed), all methods.
- **Tests:** 51 stocks, γ = 5, τ = 0.05, 10 bps; development / holdout (post-holdout) / all; paired block-bootstrap Sharpe tests against prior only, Holm over all tests.
- **Decision rule:** a method helps only if, with real forecasts, it beats prior only in both periods, does clearly better than its own placebo, and is significant after Holm.

### WK1 Weekly experiment run (E1 as pre-registered, widened at the student's request; designed 22 Oct before any weekly result)
- **As E1:** 51 stocks; week-end = last trading day of each calendar week; next-week excess-return target; the 13 base features with the same trading-day windows ("months" = blocks of 21 trading days back from each week-end); retrain every 52 weeks, validation 104 weeks, minimum training 104 weeks; Ω-A window 104 weeks with a 52-week warm-up; covariance from 252 days scaled ×5; γ = 5, τ = 0.05, δ = 2.5, 10% cap, 10 bps per trade; decisions from the last week of 2015 to Aug 2026 (557 weeks; a trailing one-day week is dropped). Weekly risk-free return (1 + annual ^IRX yield at the previous week-end)^(1/52) − 1.
- **Widened:** all 15 model families (base signals, raw target, expanding window; the lighter settings used for the wide universe for random forest, extra trees, KNN and quantile regression), each with a placebo (targets shuffled within each week); 3-block rule on development weeks; Black–Litterman portfolios for every family and their average, plus 1/N, SPY, S4 and S5 (elastic net).
- **Tests:** rank IC with Newey–West t; paired block-bootstrap Sharpe tests against BL prior only with 26-week blocks (the same calendar length as the monthly 6-month blocks), annualized ×52, Holm over all tests. 2022–26 is post-holdout.
- **Status (unchanged from E1):** robustness experiment; the monthly results remain the main results.
- **BL1 result (51 stocks; Sharpe development / holdout, $10k → end, turnover):** prior only 1.23 / 0.63, $42.6k, 24%. Elastic net: Ω-A (Phase 4) 1.16 / 0.50, $39.8k, 30%; IC-scaled 1.22 / 0.53, $40.2k; strong-signal 1.22 / 0.54, $40.6k; one relative view 1.09 / 0.41, $33.7k; Ω-A with TE 2% 1.14 / 0.55, $39.0k, 20%; IC-scaled with TE 2% 1.22 / 0.56, $40.6k. LightGBM: Ω-A 1.19 / 0.50; IC-scaled 1.20 / 0.58; strong-signal 1.23 / 0.58, $42.0k; relative 1.10 / 0.40; Ω-A TE 2% 1.18 / 0.56, 18%; IC-scaled TE 2% 1.19 / 0.59. **Placebo (shuffled forecasts):** IC-scaled 1.22 / 0.61, $42.0k; strong-signal 1.23 / 0.62, $42.6k; Ω-A 1.16 / 0.46; relative 1.15 / 0.48. No method beats the prior in either period; nothing is significant after Holm (relative view worse than the prior, raw p 0.003–0.01).
- **Reading:** IC-scaled and strong-signal views, and the tracking-error cap, cut the damage that ML views do (holdout Sharpe 0.50 → 0.53–0.59, lower turnover with the TE cap), but random views do just as well, so the gain comes from the mechanism (views switched off when the model's recent IC ≤ 0, and kept close to the prior), not from information in the forecasts. The single relative view hurts: the past top-minus-bottom spread is a poor guide to the next month's. **No change to the main model.** Worth a sentence in the report: an IC-based "skill switch" is a principled way to stop weak ML views from hurting a BL portfolio.
- **WK1 result, forecasts (rank IC, development / holdout post-holdout):** all 15 families within −0.006 to +0.015 in development (no t above 1.5); 2 of 15 pass the 3-block rule (MLP: holdout −0.004; LightGBM classifier: holdout 0.018, t 1.49). Holdout IC is positive for 13 of 15 families (elastic net 0.053, t 1.98; PCA regression 0.031, t 2.08), against negative for 11 of 15 placebos; but the development ICs of those two were ≈ 0 and inconsistent, so the pre-set rule would not have chosen them. Up/down accuracy relative to the median 49–51%; always "up" 51.8% of holdout weeks.
- **WK1 result, portfolios (weekly; return a year development / holdout / all, $10k → end, Sharpe dev / hold, weekly turnover):** BL prior only 17.5 / 10.6 / 14.5%, $42.5k, 1.20 / 0.53, 11%; BL + elastic net 18.6 / 10.2 / 14.9%, $44.0k, 1.21 / 0.50, 11%; BL + PCA regression 14.9% / 0.50, $44.5k; BL + LightGBM 14.6%, $43.0k; average of 15 14.7%, $43.2k; MLP 13.2%, $37.8k (hold 0.22); 1/N 16.0%, $49.0k; SPY 15.0%; S4 14.6% (hold 0.24); S5 12.3%. No Sharpe difference against the prior is significant (smallest Holm p 0.58). Monthly prior for comparison: 1.23 / 0.63 — weekly rebalancing is worse than monthly for the same strategy (about 5× the trading).
- **WK1 conclusion:** weekly forecasts are slightly livelier in 2022–26 than monthly ones, and BL + elastic net weekly ends level with or a little above the prior ($44.0k vs $42.5k), but nothing is significant, the development period shows no skill, and weekly trading costs make every strategy worse than its monthly version. The monthly results stay the main results (as pre-registered in E1).

### WK2 Weekly test on the wide universe (designed 23 Oct, before any result)
- **Student's request:** run the weekly test (WK1) on the bigger universe, full version (all 15 families).
- **Setup:** as WK1, with forecasts for all 1,052 wide-universe stocks (the lighter wide settings for random forest, extra trees, KNN and quantile regression, as in WU1) and a placebo for each family; portfolios on the 1,047 stocks with a share count, with Ledoit–Wolf Σ and a 2% cap (WU4). Strategies: 1/N, SPY, BL prior only, BL + each family and their average (S4 and S5 not run on this universe). Same tests as WK1; 2022–26 post-holdout.
- **Expectation stated in advance:** on the wide universe the monthly ML views hurt the portfolio (WU5); weekly trades more, so the same or worse is expected.
- **WK2 result, forecasts (rank IC dev / holdout, NW t):** four tree models pass the 3-block rule with t ≈ 2 in both periods: hist gradient boosting 0.014 (2.46) / 0.018 (2.75), random forest 0.013 (2.02) / 0.018 (2.68), LightGBM 0.016 (2.18) / 0.016 (2.24), extra trees 0.010 (1.73) / 0.016 (2.48); their placebos 0.000–0.002 (HGB −0.004). The other 11 families: development 0.00–0.01, holdout 0.01–0.024. Elastic net chose the intercept-only model in 448 of 556 weeks (no ranking). **The first forecast result that holds in development, holdout and against the placebo** (small: IC ≈ 0.015; 4 of 15 families, so some of it may still be selection among models).
- **WK2 result, portfolios (1,047 stocks, weekly, 10 bps; return a year all, $10k → end, Sharpe dev / hold, weekly turnover):** BL prior only 12.8%, $36.2k, 0.95 / 0.51, 20%; 1/N 16.2%, $49.6k; SPY 15.0%, $44.7k. Every BL + views portfolio is worse: best MLP 11.0%, $30.5k, 0.57 / 0.44; tree models 10.0–10.4%, $27.7–28.8k, hold 0.21–0.25; the rest 8.4–10.5%. No difference significant after Holm (smallest 0.62). **Before costs** the views are also worse than the prior (gross 10.4–13.6% vs 13.9%; MLP 14.6%), and costs take a further 1.3–4.0% a year (prior 1.0%).
- **WK2 conclusion:** on 1,047 stocks at a weekly horizon, tree models do rank stocks slightly better than chance, out of sample and beyond the placebo, but the edge is far too small to survive the step into Black–Litterman (each view is weak relative to its Ω, and acting on 1,047 noisy views adds risk and turnover). Report as a finding: "a statistically detectable but economically unusable signal". Main project unchanged.

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

# Decision log

Every choice not fixed by [FYP_PLAN.md](FYP_PLAN.md) is recorded here: what was chosen, what else was considered, and why. Newest entries go at the bottom of each phase.

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

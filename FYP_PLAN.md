# FYP Plan: Risk-Aware Portfolio Recommendation via Mean-Variance Optimization with ML-Informed Black–Litterman Views

**Student:** Saumik Vij — BSc Mathematics (CS track, Extended Major in AI, Business minor), HKUST
**Hard deadline for all work:** Tuesday 1 December 2026 (semester ends 20 December)
**Start:** Wednesday 23 September 2026 (~10 weeks)

---

## 0. Instructions for Claude Code
- Build **one phase at a time** (Section 8). Do not start a phase until the previous phase's tests pass.
- Treat Section 6 (look-ahead rules) as hard requirements. Any function taking a date t must only use data up to t; add a test for it.
- Keep all parameters in `config.yaml`; no magic numbers in code.
- Items marked *optional* are built only when explicitly asked.
- Do not add methods outside this plan (no NLP, no RAG/knowledge graph, no CVaR optimizer, no deep learning beyond the small MLP) unless asked.
- ML stays at undergraduate level: scikit-learn / LightGBM with standard tooling, no custom architectures.
- The mathematical proofs are written by the student; code should include the numerical theorem-checking tests in Section 7.

---

## 1. The idea in one paragraph

A user enters an investment amount, a risk tolerance, and optional sector preferences. The system recommends a simulated portfolio of US large-cap stocks (no real trades). The mathematical core is **mean-variance portfolio optimization**. The known weakness of mean-variance is that it is extremely sensitive to estimated expected returns μ. The project addresses this with the **Black–Litterman model**: a market-equilibrium prior for μ is combined, via Bayesian updating, with **machine-learning return forecasts built from price-based signals**. The ML models' uncertainty (from past forecast errors or ensemble disagreement) sets how much their views are trusted. Everything is evaluated with a **walk-forward backtest** that strictly avoids look-ahead bias, against simple benchmarks. Risk is assessed both by variance and by **tail risk (VaR and CVaR)**, to check whether a variance-based optimizer also controls large losses. The project does not claim to beat the market.

**Pipeline:**

```
Price data → features → ML forecast of next-month returns
                                   │  (views Q, uncertainty Ω from forecast error / ensemble disagreement)
Market-equilibrium prior Π ────────┤
                                   ▼
                       Black–Litterman posterior μ_BL
                                   ▼
  Covariance Σ ──→  Mean-variance optimizer (risk tolerance γ, constraints)
                                   ▼
                      Portfolio weights → walk-forward backtest → metrics
```

**Explicitly out of scope (future work):** NLP/sentiment, RAG / knowledge-graph components (see Section 10), CVaR *optimization* (CVaR is used only as an evaluation metric), risk parity as a method, Monte Carlo simulation, deep learning models (LSTM/transformers), real trading.

**Optional (only if the core is done on time):** O1 — proof of the Sharpe ratio standard error; O2 — Ledoit–Wolf covariance shrinkage (proof + ablation); O3 — Diebold–Mariano test of forecast accuracy. The Sharpe confidence intervals themselves are **not** optional: they are used in evaluation with the formula cited from Lo (2002).

---

## 2. Mathematical framework (the core of the report)

### 2.1 Setup and notation
- n assets, weight vector w ∈ ℝⁿ, expected returns μ ∈ ℝⁿ, covariance Σ ∈ ℝⁿˣⁿ (assumed positive definite), risk-free rate r_f.
- Portfolio return μᵀw, variance wᵀΣw.

### 2.2 Optimization problem (single framework: mean-variance)
Main problem, where γ > 0 is risk aversion (set by the user's risk tolerance):

  maximize  μᵀw − (γ/2) wᵀΣw
  subject to 1ᵀw = 1, 0 ≤ wᵢ ≤ w_max, optional sector constraints

Special cases derived inside the same framework:
- **Global minimum-variance portfolio** (ignore μ).
- **Efficient frontier** (minimize variance for each target return).
- **Tangency / maximum-Sharpe portfolio** (the frontier point with highest Sharpe ratio).

### 2.3 Theorems to state and prove in the report
| # | Result | Proof tools |
|---|--------|-------------|
| T1 | Existence and uniqueness of the optimal portfolio when Σ ≻ 0 | Strict convexity, closed convex feasible set |
| T2 | Closed-form efficient frontier (equality constraints only); frontier is a parabola in (σ², μ) space, using A = 1ᵀΣ⁻¹1, B = 1ᵀΣ⁻¹μ, C = μᵀΣ⁻¹μ | Lagrange multipliers, matrix calculus |
| ↳ Cor. 2.1 | Two-fund separation: every frontier portfolio is an affine combination of two fixed portfolios | Follows from T2 (w* affine in target return) |
| T3 | Closed-form tangency portfolio w ∝ Σ⁻¹(μ − r_f 1) | Maximize Sharpe ratio, homogeneity argument |
| T4 | KKT conditions for the long-only problem are necessary and sufficient; interpretation of multipliers (why some assets get zero weight) | Convexity + Slater's condition |
| T5 | Sensitivity bound: ‖Δw*‖ grows with the condition number κ(Σ) times ‖Δμ‖ — explains why raw mean-variance is unstable. Follow with a numerical check of κ(Σ) on the real data. If the constrained case gets messy, prove the unconstrained/equality-constrained case | Matrix norms, perturbation of linear systems |
| T6 | Black–Litterman posterior mean μ_BL = [(τΣ)⁻¹ + PᵀΩ⁻¹P]⁻¹[(τΣ)⁻¹Π + PᵀΩ⁻¹Q], derived **both** as a Bayesian normal-normal update and as generalized least squares, and shown to agree | Gaussian conjugacy, completing the square, GLS |
| ↳ Cor. 6.1 | BL limiting cases: Ω → ∞ gives μ_BL → Π (views ignored); Ω → 0 makes views hold exactly | Limits of matrix inverses |
| T7 | CVaR is a coherent risk measure (in particular subadditive); VaR is not — explicit two-asset counterexample where VaR(A+B) > VaR(A) + VaR(B). Justifies reporting CVaR alongside variance | Artzner et al. axioms; Rockafellar–Uryasev representation CVaR_α(L) = min_ζ {ζ + E[(L−ζ)⁺]/(1−α)} |
| O1 *(optional)* | Asymptotic standard error of the Sharpe ratio (Lo, 2002). If not proved, the formula is cited and still used for confidence intervals | Delta method, CLT |
| O2 *(optional)* | Ledoit–Wolf shrinkage estimator is positive definite and better conditioned than the sample covariance | Convex combination of PSD/PD matrices, eigenvalue bounds |

**Depth priorities:** spend the most effort on T4 (careful sufficiency proof, Slater's condition, economic meaning of multipliers), T6 (both derivations) and T5 (bound + empirical evidence on real Σ). These three carry the project's argument: optimize → why it is unstable → how BL + ML fixes it.

### 2.4 Black–Litterman specification
- **Prior:** Π = δ Σ w_mkt (reverse optimization). w_mkt = market-cap weights of the universe; δ = market risk aversion (default 2.5, or estimated from market excess return / variance).
- **Views:** absolute view on every stock: P = I, Q = ML forecast of next-month excess return.
- **View uncertainty:** Ω = diag(σ²_err,i), where σ²_err,i is the mean squared out-of-sample forecast error of the ML model for stock i over a trailing window (only past errors, never future ones). Accurate model ⇒ small Ω ⇒ views matter more.
- **τ:** default 0.05; sensitivity analysis over a grid.

### 2.5 Tail-risk definitions (evaluation only)
For a loss L (negative portfolio return) and confidence level α = 0.95:
- **VaR_α(L)** = the α-quantile of L (loss exceeded only 5% of the time).
- **CVaR_α(L)** = expected loss given that the loss is at least VaR_α (average of the worst 5%).
- Estimated historically from the backtest's realized monthly returns (and, as a check, from daily returns).

### 2.6 Risk tolerance mapping
- Low / Medium / High → γ = 10 / 5 / 2 (tunable). Also report the resulting ex-ante volatility so the user sees what the choice means.
- Investment amount only scales weights into dollar and whole-share allocations (leftover as cash).

---

## 3. Data and universe

- **Source:** Yahoo Finance via `yfinance` (daily OHLCV, adjusted close). Bloomberg only if easily accessible — not required.
- **Universe:** ~50 large-cap US stocks from the S&P 500, 4–6 per GICS sector, fixed list. Chosen to be large and liquid.
- **Benchmark index:** SPY. **Risk-free rate:** 13-week T-bill (`^IRX`), converted to monthly.
- **Period:** January 2010 – most recent complete month (≈ Aug/Sep 2026). First ~5 years used only for initial training; out-of-sample backtest from 2016 onward.
- **Frequency:** daily data for features and covariance; **monthly rebalancing and monthly return forecasts**.
- **Known limitations to state in the report:**
  - Survivorship bias: today's large caps are companies that survived. Acknowledge; if time allows, compare with a random subset or note the likely upward bias.
  - Market-cap weights for the BL prior are approximated (price × current shares outstanding); acknowledge the approximation.

---

## 4. ML component (price-based signals only, undergraduate level)

**Design principle:** simple, well-validated, interpretable models over complex architectures. On monthly return data, complex models (LSTM/transformers) mainly overfit; Gu, Kelly & Xiu (2020) find tree ensembles and shallow neural nets work best. The ML is judged on correct validation, honest evaluation, and how its outputs feed the mathematical model.

### 4.1 Target and features
- **Target:** next-month excess return of each stock.
- **Features (~15, computed at each month-end using only data up to that date):** momentum (1m, 3m, 6m, 12-1m), 1-month reversal, realized volatility (1m, 3m), market beta vs SPY (trailing 1y), idiosyncratic volatility (residual vol vs SPY), maximum daily return last month, distance from 52-week high, volume trend / turnover, simple RSI-type indicator, 1-year max drawdown. Cross-sectionally rank-normalized each month.

### 4.2 Models (three families of increasing flexibility + baseline)
1. Historical mean (baseline, no ML)
2. **Elastic net** — regularized linear regression (ridge and lasso as special cases)
3. **Gradient boosted trees** — LightGBM (or sklearn HistGradientBoosting)
4. **Small MLP** — scikit-learn MLPRegressor, 2–3 hidden layers, early stopping

Research question: does nonlinearity improve return forecasts, and does it improve the *portfolio*?

### 4.3 Training and validation
- Walk-forward with an expanding window; retrain every 12 months.
- Hyperparameters tuned by time-ordered validation on the last portion of each training window only (no random k-fold).
- Early stopping for LightGBM and the MLP; fixed random seeds for reproducibility.

### 4.4 Uncertainty → Black–Litterman Ω (the key ML–math link)
Compare two simple ways of turning ML uncertainty into view uncertainty Ω = diag(σ²ᵢ):
- **Ω-A: residual variance** — mean squared out-of-sample forecast error of stock i over a trailing window (past errors only).
- **Ω-B: ensemble disagreement** — train ~10 copies of the model with different seeds / bootstrap samples; σ²ᵢ = variance of their predictions for stock i.

Question answered: which uncertainty measure produces better, more stable BL portfolios?

### 4.5 Forecast evaluation
- **Statistical:** out-of-sample R² (vs zero forecast, Gu–Kelly–Xiu style), monthly Spearman rank IC (mean and t-stat), directional hit rate.
- **Economic:** decile test — each month sort stocks by predicted return; report top-minus-bottom decile return (with ~50 stocks, use quintiles if deciles are too thin).
- **Interpretability:** permutation feature importance (scikit-learn); compare to known findings (momentum, volatility effects).
- *Optional (O3):* Diebold–Mariano test of ML vs historical-mean forecast errors.
- Expect small R² and IC values — normal for monthly returns; discuss honestly.

## 5. Strategies compared

| Label | Description |
|-------|-------------|
| S1 | Equal-weight 1/N |
| S2 | SPY buy-and-hold |
| S3 | Global minimum-variance |
| S4 | Mean-variance with historical-mean μ (classical Markowitz) |
| S5 | Mean-variance with raw ML forecasts as μ (no BL) |
| S6 | **Mean-variance with Black–Litterman μ (ML views) — main model**, run for each ML model and each Ω method |

Key comparisons: S6 vs S4 (does the ML+BL pipeline help?), S6 vs S5 (does BL stabilize ML forecasts?), everything vs S1/S2.

Ablations: τ grid, Ω-A vs Ω-B, Ω scaling, γ levels (low/med/high risk), elastic net vs LightGBM vs MLP, transaction cost levels. *Optional:* sample covariance vs Ledoit–Wolf.

---

## 6. Backtest and evaluation rules (no look-ahead bias)

1. At each month-end t, use **only data available at t** for features, covariance, BL prior, Ω, and model training.
2. Portfolio chosen at t is held over month t+1; its realized return is recorded.
3. No random k-fold CV anywhere — only time-ordered splits.
4. Covariance: trailing 252 trading days of daily returns, scaled to monthly.
5. Transaction costs: 10 bps per unit turnover (sensitivity: 0, 10, 25 bps).
6. Weight cap w_max = 10% per stock (tunable).
7. Unit tests assert that no feature at date t uses data after t.

**Metrics:** annualized return, annualized volatility, Sharpe ratio with confidence interval (Lo 2002; O1), maximum drawdown, **historical VaR and CVaR at 95% (T7)**, average monthly turnover, and cumulative return plots. Report weight stability over time (links to T5).

**Tail-risk discussion:** compare strategy rankings by volatility vs by CVaR. If they disagree (e.g. a lower-volatility portfolio has worse CVaR), that is a finding about the limits of variance as a risk measure and motivates CVaR optimization as future work.

**Framing:** results are reported whether or not the main model outperforms. A valid conclusion can be "BL makes ML-based portfolios more stable with lower turnover" even if returns are similar.

---

## 7. Code structure (for Claude Code)

```
FYP/                         # repo root
├── config.yaml              # universe, dates, γ values, τ, costs, caps
├── data/                    # raw/ and processed/ (git-ignored)
├── src/
│   ├── data.py              # download + clean prices, SPY, T-bill, sectors
│   ├── features.py          # point-in-time feature construction
│   ├── estimators.py        # sample covariance, historical mean, (optional) Ledoit–Wolf
│   ├── optimization.py      # closed-form frontier, tangency, GMV, constrained QP (cvxpy)
│   ├── black_litterman.py   # prior Π, views, Ω, posterior
│   ├── models.py            # historical mean, elastic net, LightGBM, MLP; walk-forward training; ensembles
│   ├── ml_eval.py           # R²_oos, rank IC, hit rate, decile test, permutation importance
│   ├── backtest.py          # monthly walk-forward engine with costs
│   ├── metrics.py           # returns, vol, Sharpe + CI, drawdown, VaR, CVaR, turnover
│   └── recommend.py         # user inputs → current recommended portfolio
├── tests/                   # includes numerical checks of the theorems
├── notebooks/               # experiments and figures
├── app/                     # simple Streamlit or CLI demo
└── report/                  # LaTeX report and figures
```

**Theorem-checking tests (link math to code):**
- Closed-form frontier (T2) matches the cvxpy solution to numerical tolerance.
- Two-fund separation (Cor. 2.1): any frontier portfolio equals the predicted combination of two others.
- Tangency formula (T3) matches numerical Sharpe maximization.
- BL posterior (T6): Bayesian and GLS implementations give identical μ_BL.
- BL limits (Cor. 6.1): huge Ω returns Π; tiny Ω returns Q.
- KKT conditions (T4) hold at the solver's long-only solution.
- VaR non-subadditivity (T7): the report's counterexample reproduces VaR(A+B) > VaR(A) + VaR(B) numerically; CVaR(A+B) ≤ CVaR(A) + CVaR(B) holds on random samples.
- CVaR computed directly (tail average) matches the Rockafellar–Uryasev minimization formula.
- *(Optional)* Ledoit–Wolf estimate is positive definite with smaller condition number than the sample covariance (O2).

**ML leakage tests:** training data for a model used at date t contains no target realized after t; hyperparameter validation split is strictly earlier than the forecast date.

**Libraries:** numpy, pandas, scipy, cvxpy, scikit-learn, lightgbm, yfinance, matplotlib, pytest, streamlit (optional).

---

## 8. Timeline (23 Sep – 1 Dec 2026)

| Phase | Dates | Build | Math / writing |
|-------|-------|-------|----------------|
| **0. Setup & data** | 23 Sep – 29 Sep | Repo, config, `data.py`: download and clean universe, SPY, T-bill, sectors; save processed data | Fix notation; write setup section |
| **1. Optimization core** | 30 Sep – 13 Oct | `estimators.py`, `optimization.py`: GMV, frontier, tangency, long-only constrained QP with sector constraints; theorem tests for T2–T4 and Cor. 2.1; frontier plots | Prove T1–T4 (+ Cor. 2.1); **start early, in parallel with Phase 0** |
| **2. Backtest engine & baselines** | 14 Oct – 27 Oct | `backtest.py`, `metrics.py`: walk-forward engine, costs, all metrics incl. VaR/CVaR, S1–S4 results, look-ahead tests, T7 tests, empirical κ(Σ) check. **Last few days: start `features.py`** | Prove T5, T7; **full draft of math chapter by 27 Oct** — show advisor |
| **3. ML forecasting** | 28 Oct – 13 Nov | `features.py`, `models.py`, `ml_eval.py`: ~15 features, elastic net, LightGBM, MLP, walk-forward training + tuning, ensembles for Ω-B, R²_oos / IC / decile test / permutation importance, leakage tests; S5 results | Write ML methodology section |
| **4. Black–Litterman integration** | 14 Nov – 21 Nov | `black_litterman.py`: prior, Ω-A and Ω-B, posterior; S6 results for each model and Ω method; ablations; theorem tests for T6 and Cor. 6.1. *Optional:* Ledoit–Wolf + O2 test | Prove T6 (+ Cor. 6.1) (*optional:* O1, O2); write results section incl. tail-risk discussion |
| **5. Demo & final report** | 22 Nov – 1 Dec | `recommend.py` + simple Streamlit/CLI demo; final figures; code cleanup and README | Complete report: intro, lit review, discussion, limitations, future work (incl. Section 10); final proofread |

**Checkpoints:** meet the advisor after Phase 2 (math draft + baseline results) and after Phase 4 (main results).

**If running late, cut in this order:** optional items O1–O3 → Streamlit demo (use CLI) → MLP (keep elastic net + LightGBM) → ablations beyond τ, γ and Ω-A vs Ω-B. Never cut: theorems T1–T7 and their corollaries, the walk-forward backtest, the VaR/CVaR metrics, the Ω-A vs Ω-B comparison, or the S4 vs S6 comparison.

---

## 9. Key references
- Markowitz (1952), Portfolio Selection, *J. Finance*.
- Merton (1972), An Analytic Derivation of the Efficient Portfolio Frontier, *JFQA*.
- Black & Litterman (1992), Global Portfolio Optimization, *FAJ*; He & Litterman (2002); Idzorek (2007).
- Best & Grauer (1991); Chopra & Ziemba (1993) — sensitivity to estimation error.
- DeMiguel, Garlappi & Uppal (2009), Optimal Versus Naive Diversification, *RFS*.
- Jagannathan & Ma (2003), Why Imposing the Wrong Constraints Helps, *J. Finance*.
- Ledoit & Wolf (2004), Honey, I Shrunk the Sample Covariance Matrix, *JPM*.
- Gu, Kelly & Xiu (2020), Empirical Asset Pricing via Machine Learning, *RFS*.
- Lo (2002), The Statistics of Sharpe Ratios, *FAJ*.
- Artzner, Delbaen, Eber & Heath (1999), Coherent Measures of Risk, *Mathematical Finance*.
- Rockafellar & Uryasev (2000), Optimization of Conditional Value-at-Risk, *J. Risk*; (2002) CVaR for General Loss Distributions, *J. Banking & Finance*.
- López de Prado (2018), *Advances in Financial Machine Learning* — walk-forward validation, leakage.
- Boyd & Vandenberghe (2004), *Convex Optimization*; Cornuejols & Tütüncü (2007), *Optimization Methods in Finance*.

---

## 10. Possible later extensions (NOT part of the current build)

Do not start these unless the student explicitly asks, and only after Phases 0–5 are complete.

- **RAG / knowledge-graph component.** A graph of company relationships (sector, supply chain, competitors) plus retrieval over financial text. Deferred because it reintroduces NLP (advisor suggested one signal type), needs hard-to-get relationship data, adds no theorem, and is hard to test for look-ahead bias. Possible roles later:
  - inform the covariance structure (e.g. relationship-based shrinkage target) or generate additional Black–Litterman views;
  - a presentation-only feature where an LLM explains the recommended portfolio in plain language (does not touch the maths or backtest). Only consider if everything else is done by ~mid-November.
- NLP/sentiment signals as a second set of BL views.
- CVaR optimization (Rockafellar–Uryasev LP) as an alternative to mean-variance.
- Monte Carlo simulation of forward-looking portfolio risk.


# Proof outlines for T1–T7

Working notes for writing the proofs in the final report. For each result: the precise
statement (with the corrections from the advisor's mid-term feedback, DECISIONS A5), the
steps of the proof, the traps to avoid, and the test that checks it numerically.
Depth priority (plan Section 2.3): **T4, T5, T6** first.

## Conventions (state these once, before T1)

- n = 51 stocks, weights w ∈ ℝⁿ, **μ = expected monthly *excess* returns** (over the T-bill).
- Σ ∈ ℝⁿˣⁿ, symmetric positive definite (Σ ≻ 0): the monthly covariance (daily sample
  covariance of the last 252 trading days × 21). Checked on the real data:
  `tests/test_estimators.py` (Σ ≻ 0 at every rebalance date).
- γ > 0 risk aversion (10 / 5 / 2). Everything is monthly; annualization is only for reporting.
- Sectors: S ∈ {0,1}^{11×n}, S_{k,i} = 1 iff stock i is in sector k; each stock is in exactly
  one sector k(i), so (Sᵀρ)ᵢ = ρ_{k(i)}.
- Main problem, with w_max = 0.10 and s_max = 0.25:

  (P)  maximize f(w) = μᵀw − (γ/2) wᵀΣw  subject to  1ᵀw = 1,  0 ≤ wᵢ ≤ w_max,  Sw ≤ s_max 1.

  Feasible set C. Write the equivalent minimization g(w) = (γ/2) wᵀΣw − μᵀw when using KKT.

---

## T1. Existence and uniqueness

**Statement.** If Σ ≻ 0, γ > 0 and C ≠ ∅, (P) has exactly one optimal portfolio.

**Steps.**
1. *C is non-empty:* the equal-weight portfolio wᵢ = 1/51 ≈ 0.0196 is feasible: it is in
   [0, 0.10], sums to 1, and the largest sector has 6 stocks, so its weight is 6/51 ≈ 0.118 ≤ 0.25.
2. *C is compact:* closed (intersection of finitely many closed half-spaces and one hyperplane)
   and bounded (0 ≤ wᵢ ≤ w_max).
3. *Existence:* f is continuous, so it attains its maximum on the compact set C (Weierstrass).
4. *Uniqueness:* ∇²f = −γΣ ≺ 0, so f is strictly concave. If w₁ ≠ w₂ were both optimal with
   value f*, then w̄ = (w₁ + w₂)/2 ∈ C (C is convex) and f(w̄) > ½f(w₁) + ½f(w₂) = f*, a contradiction.

**Traps.** Uniqueness needs Σ ≻ 0, not just Σ ⪰ 0. Show C ≠ ∅ explicitly (the sector rule could
make it empty; here it does not).

**Test.** `test_T1_unique_optimum_strong_concavity`.

---

## T2. Closed-form efficient frontier (equality constraints only) + Cor. 2.1

**Statement.** Minimize wᵀΣw subject to 1ᵀw = 1 and μᵀw = m (no bounds). Let
A = 1ᵀΣ⁻¹1, B = 1ᵀΣ⁻¹μ, C = μᵀΣ⁻¹μ, D = AC − B². If μ is not a multiple of 1, then D > 0 and

  w*(m) = Σ⁻¹(λ1 + ημ),  λ = (C − Bm)/D,  η = (Am − B)/D,  σ²(m) = (Am² − 2Bm + C)/D,

a parabola in (σ², m) with vertex at the global minimum-variance portfolio
w_GMV = Σ⁻¹1/A, m = B/A, σ² = 1/A.

**Steps.**
1. Lagrangian L = ½wᵀΣw − λ(1ᵀw − 1) − η(μᵀw − m); ∇_w L = 0 gives w = Σ⁻¹(λ1 + ημ).
2. Substitute into both constraints: [A B; B C][λ; η] = [1; m]. Solve the 2×2 system.
3. D > 0: Cauchy–Schwarz in the inner product ⟨x, y⟩ = xᵀΣ⁻¹y gives B² ≤ AC, with equality
   iff μ ∝ 1.
4. σ²(m) = w*ᵀΣw* = λ·1ᵀw* + η·μᵀw* = λ + ηm, then substitute λ, η.
5. Sufficiency: the objective is strictly convex and the constraints are affine, so the
   stationary point is the unique minimizer.

**Cor. 2.1 (two-fund separation).** λ and η are affine in m, so w*(m) = g + m·h with
g = (CΣ⁻¹1 − BΣ⁻¹μ)/D and h = (AΣ⁻¹μ − BΣ⁻¹1)/D. Any two distinct frontier portfolios span
the whole frontier.

**Tests.** `test_T2_closed_form_frontier_matches_solver`, `test_T2_frontier_is_parabola_in_variance_mean_space`,
`test_T2_gmv_is_vertex`, `test_cor21_two_fund_separation`.

---

## T3. Tangency (maximum-Sharpe) portfolio — corrected per advisor

**Statement (μ = excess returns).** Among w with 1ᵀw = 1, the Sharpe ratio
SR(w) = μᵀw / √(wᵀΣw) is maximized by

  w_T = Σ⁻¹μ / (1ᵀΣ⁻¹μ),  provided 1ᵀΣ⁻¹μ > 0,  with SR(w_T) = √(μᵀΣ⁻¹μ).

There is **no r_f term** because μ is already in excess of r_f. (With total returns μ̃, replace
μ by μ̃ − r_f 1; that is the form used in the code, `tangency_closed_form`.)

**Steps.**
1. SR is unchanged by positive scaling, so maximize over all w with SR defined, then rescale.
2. Cauchy–Schwarz: μᵀw = (Σ^{−1/2}μ)ᵀ(Σ^{1/2}w) ≤ ‖Σ^{−1/2}μ‖·‖Σ^{1/2}w‖ = √(μᵀΣ⁻¹μ)·√(wᵀΣw).
3. Equality iff Σ^{1/2}w = cΣ^{−1/2}μ with c > 0, i.e. w = cΣ⁻¹μ.
4. The budget 1ᵀw = 1 fixes c = 1/(1ᵀΣ⁻¹μ), which is positive iff 1ᵀΣ⁻¹μ > 0.
5. If 1ᵀΣ⁻¹μ ≤ 0, no budget-feasible portfolio reaches the bound: no tangency portfolio exists.
   (In total-return terms: r_f ≥ B/A, the GMV mean.)

**Trap.** This is the *unconstrained* tangency; with long-only, cap or sector constraints there is
no closed form (the report uses (P) with γ instead).

**Tests.** `test_T3_tangency_matches_numerical_sharpe_max`, `test_T3_tangency_on_frontier_and_best_sharpe`,
`test_T3_no_tangency_when_rf_above_gmv_mean`.

---

## T4. KKT conditions with stock and sector limits — corrected per advisor (depth priority)

**Statement.** w* solves (P) iff there exist ν ∈ ℝ, λ, η ∈ ℝⁿ₊ and ρ ∈ ℝ¹¹₊ such that

- stationarity: μᵢ − γ(Σw*)ᵢ = ν − λᵢ + ηᵢ + ρ_{k(i)} for every i;
- primal feasibility: w* ∈ C;
- dual feasibility: λ, η, ρ ≥ 0;
- complementary slackness: λᵢwᵢ* = 0, ηᵢ(wᵢ* − w_max) = 0, ρ_k(Σ_{j∈k} wⱼ* − s_max) = 0.

**Steps.**
1. Write (P) as min g(w) = (γ/2)wᵀΣw − μᵀw with affine constraints; form the Lagrangian
   L = g + ν(1ᵀw − 1) − λᵀw + ηᵀ(w − w_max1) + ρᵀ(Sw − s_max1).
2. ∇_w L = 0 gives γΣw − μ + ν1 − λ + η + Sᵀρ = 0, i.e. the stationarity line above.
3. *Necessity:* the constraints are affine, so a constraint qualification holds automatically
   (linear constraints). Alternatively, Slater holds: the equal-weight portfolio is *strictly*
   feasible for every inequality (0 < 1/51 < 0.10, sector weights ≤ 0.118 < 0.25).
4. *Sufficiency:* g is convex and the constraints are affine. For any feasible v, g(v) ≥ g(w*) +
   ∇g(w*)ᵀ(v − w*), and stationarity + complementary slackness + dual feasibility show
   ∇g(w*)ᵀ(v − w*) ≥ 0. (Write this inequality out; it is the heart of the proof.)
5. Uniqueness of w* comes from T1; the multipliers need not be unique.

**Interpretation (the part the advisor asked for).** Let mᵢ = μᵢ − γ(Σw*)ᵢ be stock i's
marginal utility.
- 0 < wᵢ* < w_max: mᵢ = ν + ρ_{k(i)}. Held stocks in the same sector share one marginal
  utility; if the sector cap is slack (ρ_k = 0), all such stocks equal the budget level ν.
- wᵢ* = 0: mᵢ = ν + ρ_{k(i)} − λᵢ ≤ ν + ρ_{k(i)}: not worth buying, which is why some stocks get
  exactly zero weight.
- wᵢ* = w_max: mᵢ = ν + ρ_{k(i)} + ηᵢ: would be bought more if the cap allowed.
- ρ_k > 0 (sector k at 25%): ρ_k is the shadow price of the cap, the rate at which the optimal
  utility would rise if sector k's limit were relaxed.

**Tests.** `test_T4_kkt_hold_at_solver_solution` (with and without sector limits),
`test_T4_multiplier_interpretation`, `test_T4_kkt_on_real_data`.

---

## T5. Sensitivity of the optimal weights to errors in μ — restated (depth priority)

**Statement.**
(a) For (P), or any closed convex feasible set C, and any μ₁, μ₂:
  ‖w*(μ₁) − w*(μ₂)‖₂ ≤ ‖μ₁ − μ₂‖₂ / (γ λ_min(Σ)).
The bound holds even when the active constraints change, so the weights cannot jump.

(b) With no constraints, w* = Σ⁻¹μ/γ and
  ‖Δw*‖/‖w*‖ ≤ κ(Σ)·‖Δμ‖/‖μ‖,  κ(Σ) = λ_max/λ_min,
and the bound is attained (μ on the top eigenvector of Σ, Δμ on the bottom one), so it cannot be improved.

**Steps for (a).**
1. First-order optimality for maximizing a concave function on a convex set:
   ∇f(w*)ᵀ(v − w*) ≤ 0 for all v ∈ C, with ∇f(w) = μ − γΣw.
2. Apply it at w₁ = w*(μ₁) with v = w₂, and at w₂ = w*(μ₂) with v = w₁; add the two inequalities:
   γ(w₁ − w₂)ᵀΣ(w₁ − w₂) ≤ (μ₁ − μ₂)ᵀ(w₁ − w₂).
3. Left side ≥ γλ_min‖w₁ − w₂‖²; right side ≤ ‖μ₁ − μ₂‖·‖w₁ − w₂‖ (Cauchy–Schwarz). Divide.

**Steps for (b).**
1. Δw = Σ⁻¹Δμ/γ, so ‖Δw‖ ≤ ‖Σ⁻¹‖‖Δμ‖/γ.
2. μ = γΣw gives ‖μ‖ ≤ γ‖Σ‖‖w‖, i.e. ‖w‖ ≥ ‖μ‖/(γ‖Σ‖).
3. Divide: ‖Δw‖/‖w‖ ≤ ‖Σ‖‖Σ⁻¹‖·‖Δμ‖/‖μ‖ = κ(Σ)‖Δμ‖/‖μ‖ (2-norms, so ‖Σ‖‖Σ⁻¹‖ = λ_max/λ_min).
4. Tightness: take μ = v_max, Δμ = εv_min.

**Empirical side.** On the trailing covariance, κ(Σ) has median 371 and maximum 1,540 over the
128 rebalance dates (`report/tables/phase2_condition.csv`, figure `condition_number.png`):
errors in μ can be amplified hundreds of times, which is why raw mean–variance (S5) trades so much.

**Tests.** `test_T5_weights_are_lipschitz_in_mu_under_any_constraints` (cases where the active set
changes), `test_T5_unconstrained_relative_error_bounded_by_condition_number` (including tightness),
`test_T5_bound_on_real_data_with_project_constraints`.

---

## T6. Black–Litterman posterior + Cor. 6.1 (depth priority)

**Statement.** Prior μ ~ N(Π, τΣ) with Π = δΣw_mkt; views Q = Pμ + ε, ε ~ N(0, Ω), Ω ≻ 0
(P is k × n). Then μ | Q ~ N(μ_BL, M⁻¹) with M = (τΣ)⁻¹ + PᵀΩ⁻¹P and

  μ_BL = M⁻¹[(τΣ)⁻¹Π + PᵀΩ⁻¹Q] = Π + τΣPᵀ(PτΣPᵀ + Ω)⁻¹(Q − PΠ).

**Derivation 1 (Bayes).**
1. Posterior density ∝ exp(−½[(μ − Π)ᵀ(τΣ)⁻¹(μ − Π) + (Q − Pμ)ᵀΩ⁻¹(Q − Pμ)]).
2. Expand: the exponent is −½[μᵀMμ − 2μᵀb] + const, with b = (τΣ)⁻¹Π + PᵀΩ⁻¹Q.
3. Complete the square: μᵀMμ − 2μᵀb = (μ − M⁻¹b)ᵀM(μ − M⁻¹b) − bᵀM⁻¹b. So μ | Q is normal with
   mean M⁻¹b and covariance M⁻¹ (M ≻ 0 since τΣ ≻ 0).

**Derivation 2 (GLS / mixed estimation).**
1. Stack the two sources of information: y = [Π; Q] = Xμ + e, X = [I; P], e ~ N(0, V),
   V = diag(τΣ, Ω).
2. The GLS estimator μ̂ = (XᵀV⁻¹X)⁻¹XᵀV⁻¹y. Compute XᵀV⁻¹X = M and XᵀV⁻¹y = b: same as Derivation 1.

**Update form.** Woodbury: M⁻¹ = τΣ − τΣPᵀ(PτΣPᵀ + Ω)⁻¹PτΣ; multiply out M⁻¹b and simplify.
This form needs no Ω⁻¹, so it stays valid as Ω → 0 (the code uses it).

**Cor. 6.1.** Ω → ∞ (e.g. Ω = cΩ₀, c → ∞): PᵀΩ⁻¹ → 0, so μ_BL → Π (views ignored).
Ω → 0, with P of full row rank: by the update form, μ_BL → Π + τΣPᵀ(PτΣPᵀ)⁻¹(Q − PΠ), and then
Pμ_BL = Q (views hold exactly). With P = I: μ_BL → Q.

**How far μ_BL moves toward the views (advisor point; for the Results discussion, not a theorem).**
With P = I: μ_BL − Π = τΣ(τΣ + Ω)⁻¹(Q − Π).
- If Ω = cτΣ, the movement is exactly 1/(1 + c) in every direction.
- Ours is not of that form: Ω is diagonal, with median Ω_ii/(τΣ_ii) = 19, which alone would
  suggest 1/20 = 5%. But Σ has a dominant market factor (largest eigenvalue ≈ 17× the average
  variance). Measured over the 128 months (median): 59% of ‖Q − Π‖² lies along that eigenvector,
  where μ moves 55% of the way, and 23% in the other directions, giving the overall
  ‖μ_BL − Π‖/‖Q − Π‖ = 43%.
- The report should define "pull" = ‖μ_BL − Π‖₂/‖Q − Π‖₂ and give this decomposition.

**Tests.** `test_T6_bayesian_and_gls_derivations_agree`, `test_T6_posterior_covariance`,
`test_T6_scalar_case_is_precision_weighted_average`, and the Cor. 6.1 tests in
`tests/test_black_litterman.py`.

---

## T7. CVaR is coherent; VaR is not subadditive

**Statement.** For losses L in L¹ and α ∈ (0, 1),
CVaR_α(L) = min_ζ { ζ + E[(L − ζ)⁺]/(1 − α) } (Rockafellar–Uryasev) is monotone, translation
invariant, positively homogeneous and subadditive. VaR_α can violate subadditivity.

**Steps.**
1. *Representation:* φ(ζ) = ζ + E[(L − ζ)⁺]/(1 − α) is convex in ζ with right derivative
   1 − P(L > ζ)/(1 − α); the minimum is at ζ = VaR_α(L) (cite Rockafellar–Uryasev 2002 for
   distributions with atoms).
2. *Subadditivity:* let ζ₁, ζ₂ attain the minimum for L₁, L₂. Since
   (L₁ + L₂ − ζ₁ − ζ₂)⁺ ≤ (L₁ − ζ₁)⁺ + (L₂ − ζ₂)⁺, evaluating φ for L₁ + L₂ at ζ₁ + ζ₂ gives
   CVaR(L₁ + L₂) ≤ φ(ζ₁ + ζ₂) ≤ CVaR(L₁) + CVaR(L₂).
3. *Others:* monotone, since (·)⁺ is non-decreasing; translation: substitute ζ' = ζ − c;
   homogeneity: substitute ζ' = ζ/λ for λ > 0.
4. *VaR counterexample:* two independent loans, each losing 100 with probability 4% (else 0).
   VaR₉₅ of each is 0 (P(loss > 0) = 4% < 5%); P(at least one default) = 1 − 0.96² = 7.84% > 5%,
   so VaR₉₅(A + B) = 100 > 0 + 0.

**Tests.** `test_T7_var_not_subadditive_counterexample`, `test_T7_cvar_subadditive_on_random_samples`,
`test_T7_cvar_other_coherence_axioms`.

---

## Statistical procedures to state precisely in the report (advisor point 3)

Not theorems, but they must be fully specified (DECISIONS A5.1, item 4):
- **Sharpe CI:** Lo (2002), i.i.d. case: SE(SR) = √((1 + SR²/2)/T) on monthly excess returns,
  annualized by √12; T = 128.
- **Paired Sharpe tests:** circular block bootstrap, the same blocks for both strategies; block
  length 6 months (≈ T^{1/3}); 10,000 resamples; seed 42; p-value from the recentred bootstrap
  distribution. Checks: blocks of 3 and 12 months; Memmel-corrected Jobson–Korkie.
- **Family of 28 (Holm):** {elastic net, LightGBM, MLP} × {Ω-A, Ω-B, Ω-B-cal} × {S6 vs S5 of the
  same model, S6 vs S4, S6 vs S1}, plus the headline S6 vs BL prior only.
- **Newey–West:** Bartlett kernel, lag ⌊4(T/100)^{2/9}⌋ = 4.

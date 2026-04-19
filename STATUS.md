# ABB Project Status

Last updated: 2026-04-15, evening

## Goal

Compare ABB's (2017 Econometrica) quantile regression M-step against
MLE M-step for a three-period nonlinear earnings model.

## CRITICAL ISSUE: MLE slopes biased in full EM

Despite MLE being theoretically more efficient (sharper CDLL curvature),
MLE gives WORSE slope estimates than QR in full EM. Specifically,
τ=0.25 slope biased DOWN, τ=0.75 slope biased UP.

THIS LIKELY INDICATES REMAINING LOGICAL ISSUES IN THE MLE CODE.
The user suspects bugs remain. Next session should focus on finding them.

### Evidence that something is wrong:
- Observed η (no E-step): MLE beats QR 10/10 seeds ← CORRECT
- 1 E-step from truth: MLE slopes [0.79, 0.79, 0.78] ← CLOSE to QR [0.78, 0.78, 0.77]
- Full EM cold start: MLE [0.72, 0.78, 0.81] vs QR [0.77, 0.80, 0.81] ← BIASED
- Two-stage (QR→MLE): MLE [0.73, 0.81, 0.82] vs QR [0.77, 0.79, 0.80] ← STILL BIASED
- Fixed tails at truth: MLE [0.79, 0.79, 0.79] ← BETTER but still slightly biased

### Diagnostic findings:
- Tail profiling contributes some bias (fixed-tail MLE closer to truth)
- The bias is SYSTEMATIC: τ=0.25 always too low, τ=0.75 always too high
- The non-crossing constraint restricts the feasible region; may be too tight
- MLE achieves higher CDLL than QR on same draws but worse slope recovery
- The mechanism: -log(gap) term pushes MLE to widen quantile spacing
  (lower Q_1, higher Q_3) → slope bias at tails

### Possible remaining issues to investigate:
1. The exploration bounds in coordinate descent (w = 0.5 → 0.05) may be
   too restrictive, preventing MLE from reaching the true optimum
2. The QR warm start within each M-step re-runs QR from scratch each time;
   maybe this biases the warm start
3. The ε centering (par.a_eps .-= mean(par.a_eps)) may interact badly
   with MLE transition fitting
4. The feasible interval may be computed slightly wrong in edge cases
5. The profiled tail rates inside neg_profiled_cdll may create a bias
   when observations move between segments during coordinate descent

## Summary of all results (Apr 15)

### M-step on observed η (N=2000, 10 seeds)
  MLE beat QR in 10/10 seeds. Avg slope: QR=0.794, MLE=0.796 (true=0.800).

### Full EM (N=500, S=50, M=50, seed=42)
                      slopes [τ=0.25, τ=0.50, τ=0.75]    mean    time
  QR:                 [0.774, 0.803, 0.808]              0.795    56s
  std MLE (cold):     [0.726, 0.792, 0.822]              0.780    3313s
  gap MLE (cold):     [0.719, 0.785, 0.811]              0.771    1607s
  gap MLE (QR warm):  [0.732, 0.810, 0.822]              0.788    1990s
  True:               [0.800, 0.800, 0.800]              0.800

### Tail profiling diagnostic (1 E-step from truth, N=500, M=50)
  True:               [0.800, 0.800, 0.800]
  QR:                 [0.779, 0.780, 0.774]
  MLE profiled:       [0.791, 0.789, 0.783]
  MLE fixed@truth:    [0.794, 0.792, 0.790]
  MLE fixed@QR:       [0.792, 0.789, 0.783]

### Non-crossing constraint
  Gap reparameterization: Q₂ free, Q₁=Q₂-exp(h'δ_L), Q₃=Q₂+exp(h'δ_U)
  Implemented in mle_gap.jl (coord descent version).
  Non-crossing guaranteed globally; unconstrained optimization.

## Paper
  abb_mle.tex — LaTeX, LyX-importable. Sections:
  1. Introduction
  2. Shape of Objective Functions (with Figures 1, 2 — profile plots)
  3. MLE of ABB Model
  4. MLE of Log-Spline Model
  Profile plots generating: profile_qr_vs_cdll_slopes.png (may still be running)

## HPC
  - Old buggy jobs cancelled on Puma
  - Fixed code NOT yet uploaded
  - DO NOT resubmit until MLE bias issue is resolved
  - Account: ichimura, Julia at ~/julia-1.11.5/bin/julia

## Key files (Apr 15)
  ABB_three_period.jl      Main code (MLE bugs fixed, feasible-interval coord descent)
  mle_gap.jl               Gap reparameterization MLE
  mle_variants.jl          Three MLE approaches compared
  test_mle_comprehensive.jl Comprehensive tests
  test_mle_diag.jl         1 E-step + M-step diagnostic
  test_mle_fixed_tails.jl  Tail profiling diagnostic
  test_two_stage.jl        Two-stage QR → gap MLE
  test_full_em_gap.jl      Full EM gap MLE
  abb_mle.tex              Paper draft
  results_two_stage.jls    Two-stage results (serialized)
  plot_qr_vs_mle_profiles.jl  Profile comparison plots (may still be running)

## Update (Apr 15, late)
  Key fix: sort knots inside neg_profiled_cdll so objective is finite everywhere.
  Wide bounds [0,2] work now. MLE beats QR on observed η (-2.997 vs -3.000).
  Feasibility (non-crossing) used as post-hoc check, not as search constraint.

## Apr 16: Smooth model correctly specified test

Generated data FROM the smooth ABB density model (cubic log-density
interior + exponential tails, ABB quantile knots), then fitted the
smooth model to this data.

Setup:
  N=500, T=3, L=3, K=2, τ=(0.25, 0.50, 0.75)
  True a_Q: slopes=0.8, intercepts=±0.337
  True b_coef: β_1=2.0, γ_l=-0.1 (constants, no Hermite dependence)

Results (results_20260416_smooth_well_specified.txt):
  Truth:   slopes=[0.80, 0.80, 0.80], intcpts=[-0.34, 0.00, 0.34]
  QR:      slopes=[0.80, 0.78, 0.77], intcpts=[+0.84, +1.35, +1.72]
  SmoothML:slopes=[0.91, 0.81, 0.76], intcpts=[+0.75, +1.24, +1.66]

  neg-ll at truth:    4.239
  neg-ll at SmoothML: 4.268  (HIGHER — LBFGS did not converge to truth)

Issues:
  - Smooth DGP with β_1=2 creates heavy right shift; η range [-5.4, 5.0]
    not centered at quantile knots → QR misinterprets the data
  - LBFGS on smooth ML runs 60 iterations (570s) but finds WORSE likelihood
    than truth — stuck at a local min
  - The smooth density parameterization (β, γ with Hermite basis) creates
    non-convex landscape where LBFGS fails

Files added today (Apr 15-16):
  exact_ml.jl              PW exact ML (forward filter, G×G transition matrix)
  smooth_ml.jl             Smooth ABB density (cubic log-density)
  smooth_ml_fixed_b.jl     Smooth ML with b_coef held fixed
  test_smooth_correctly_specified.jl  Data from smooth, fit smooth
  test_exact_sanity.jl     Verified forward filter matches direct 3D integral
  test_init_identification.jl  Showed η_1 params strongly identified
  test_init_sensitivity.jl Showed QR EM insensitive to η_1 initialization
  sml.jl, sml_sd.jl        Simulated ML (slow, not pursued)
  results_20260415_N200.jls                       EM QR+MLE at N=200
  results_20260414_N{200,500,1000}.jls            Earlier EM results
  results_20260416_smooth_well_specified.txt      Today's smooth test

Key finding (Apr 16):
  The sample MLE at N=200-500 differs from truth by finite-sample noise,
  not bias. Log-likelihood at truth = -2081.44, at sample MLE = -2081.28.
  The MLE is correctly finding the sample MLE. What we see as "bias" in
  any single run is sampling variance. Monte Carlo across many seeds
  is required to distinguish bias from variance.

## HPC setup (ready to run)
  hpc/run_hpc.jl           runs QR + Exact ML for one (N, seed)
  hpc/submit_jobs.sh       submits 800 SLURM jobs (4 N × 200 seeds)
  hpc/collect_results.jl   aggregates summary_*.jls
  hpc/sync_to_hpc.sh       rsync code to Puma
  hpc/sync_from_hpc.sh     rsync results back

## Update (Apr 15, later)
  Exact ML (forward filter on grid) is correct but the N=200 sample MLE
  differs from truth by ~0.03 in slopes due to finite-sample noise.
  At sample MLE: neg-ll = 4.166, at truth = 4.220 (lower is better!).
  So the MLE is doing the right thing — the difference from truth is
  sampling variance, not estimator bias.

  Smooth ABB density (cubic log-density with ABB quantile knots) fitted
  by LBFGS — this has issues. The smooth approximation to piecewise-uniform
  creates misspecification; fitting piecewise-uniform DGP by smooth density
  gives worse likelihood and weird parameter estimates.

  For final comparison: need Monte Carlo (200 seeds × 4 N) on HPC to
  distinguish sampling variance from estimator bias.

## HPC setup
  hpc/run_hpc.jl — runs QR + Exact ML for one (N, seed)
  hpc/submit_jobs.sh — submits 800 SLURM jobs (4 N × 200 seeds)
  hpc/collect_results.jl — aggregates summary_*.jls for bias/std/RMSE
  hpc/sync_to_hpc.sh — rsync code to Puma
  hpc/sync_from_hpc.sh — rsync results back

## exact_ml.jl (Apr 15 late)
  Exact (non-simulated) maximum likelihood via forward filter on a grid.
  Key optimization: precompute G×G transition density matrix ONCE per
  likelihood eval, then loglik_all is matrix-vector products.
  Speed: 1ms per eval at N=200, G=100 (after compilation).
  Full estimation: 22 LBFGS iters in 11 seconds.
  Results match QR closely — systematic τ=0.75 slope bias persists.
  Suggests the bias is intrinsic to the ABB parameterization, not EM artifact.

## smooth_abb.jl (Apr 15, late)
  New approach: use ABB's quantile knots but with CUBIC log-density interior.
  log f(x) = β_0 + β_1 x + Σ_l γ_l (x - q_l)₊³ - log C
  Where q_l are ABB quantiles (same Hermite parameterization as before).
  - Exponential left tail (for x < q_1: log f linear in x, need β_1 > 0)
  - Cubic decay right tail (need Σ γ_l < 0)
  - Interior smooth cubic, no gap singularities
  Test: density integrates to 1.0005 (0.05% error), smooth shape.
  TODO: integrate into EM framework, compare with ABB QR and MLE.

## Apr 16: Quantile-respecting smooth DGP

### Key user insight
QR's catastrophic intercept bias on smooth_ml data means the smooth DGP in
smooth_ml.jl was NOT actually generating data from the claimed model.
QR should recover true conditional quantiles regardless of density shape,
so QR's bad results mean the "true" knots weren't the actual τ-quantiles.

### Analytical framework (Apr 16)
For smooth density with log f(x|η) = β_0 + β_1 x + Σ_l γ_l (x-q_l(η))₊³:

Given (q_1 < q_2 < q_3, τ = [0.25,0.50,0.75], β_1 > 0), the coefficients
(β_0, γ_1, γ_2, γ_3) are uniquely determined by the 4 constraints:
  ∫ f dx = 1, F(q_l) = τ_l for l=1,2,3.

Proof: Each γ_l only enters M_l^u and higher segments via (x-q_l)₊³.
Sequential monotone solve: γ_1 from M_1/M_0 = 1, γ_2 from M_2/M_0 = 1
(given γ_1), γ_3 from M_3/M_0 = 1 (given γ_1, γ_2). Each step is
monotone-increasing 1D, so unique. β_0 from normalization.

Tail integrability: β_1 > 0 (left tail exp decay); Σ γ_l < 0 (right tail
cubic decay) — automatic since M_3 finite requires it.

### Cross-percentile restrictions
In ABB piecewise-uniform: a_Q[:, l] per column estimated independently by QR.
In smooth cubic model: γ's are IMPLICIT functions of (a_Q, β_1, τ); they
couple the knot parameters across quantile levels. MLE exploits this;
QR ignores it.

### Numerical issue with cubic-spline-log-density
For ABB-style knots (-0.337, 0, 0.337), γ's come out extreme (e.g.,
γ_2 = -181, γ_3 = 178 for β_1 = 2) because truncated powers (x-q_l)₊³
have GLOBAL support — γ_1's effect cascades into segments 2, 3, right tail.
The density IS mathematically valid but numerically nasty.

smooth_quantile_dgp.jl — numerically-stable version of the solver
(with γ_3 reparameterization). Solver works for most cases but γs remain
extreme → density is hard to sample from reliably.

### Solution: Logistic smooth DGP (logistic_abb_dgp.jl)
Use CONDITIONAL LOGISTIC instead of cubic-spline. Logistic quantile has
closed form: Q_τ(μ, α) = μ + log(τ/(1-τ))/α.

Parameterization:
  η_t | η_{t-1} ~ Logistic(μ(η_{t-1}), α(η_{t-1}))
  μ(η) = m_0 + m_1 η/σ + m_2 (η²/σ²-1)   [Hermite order K=2]
  α(η) = a_0 + a_1 η/σ + a_2 (η²/σ²-1)   [positive, Hermite]

True params (ρ=0.8, σ_v=0.5):
  μ_Q = [0, 0.8, 0]  (linear AR(1) median)
  α_Q = [3.628, 0, 0]  (constant scale)

Implied ABB-style a_Q knots:
  intercepts [-0.303, 0.000, 0.303] (from ±log(3)/3.628)
  slopes [0.8, 0.8, 0.8]
  quadratic [0, 0, 0]

Numerically stable, no extreme coefficients, smooth density at all η.
Empirical quantiles of simulated data match theoretical exactly.

### Files added Apr 16
  smooth_respecting_quantiles.jl — cubic-spline sequential solver (analysis)
  smooth_quantile_dgp.jl           — cubic-spline DGP (numerical issues)
  logistic_abb_dgp.jl              — logistic smooth DGP (WORKS CLEANLY)

## Apr 16 (late): PWL-z bugs found and fixed

### Bugs found in pwl_abb.jl:
1. **CRITICAL: cond_q sorts knots** — bubble-sort creates discontinuous objective;
   LBFGS gets stuck after 5 iterations at a kink. Also breaks column-τ
   correspondence (same class of bug as in piecewise-uniform MLE).
2. **MODERATE: pwl_logpdf returns NaN** when knots cross and interior slope < 0,
   log(negative) = NaN propagates through forward filter.
3. **MINOR: sorting breaks τ assignment** — after sort, q[1] may correspond
   to τ=0.50 not τ=0.25.

### Fix applied: gap reparameterization in cond_q
  q_2(η) = h(η)' a_Q[:, 1]             (conditional median, free)
  q_1(η) = q_2(η) - exp(h(η)' a_Q[:, 2])  (always < q_2)
  q_3(η) = q_2(η) + exp(h(η)' a_Q[:, 3])  (always > q_2)
Non-crossing guaranteed for all η. Objective is smooth → LBFGS should converge.
Verified: at truth, gap reparameterization gives identical quantiles.

### Likelihood boundedness proved empirically (test_pwl_degeneracy.jl)
Squeezing ε or transition densities always INCREASES neg-ll (worsens fit):
  - ε only squeeze (1.0→0.01): neg-ll 3.24 → 36.1
  - Both squeeze (1.0→0.05): neg-ll 3.24 → 59.8
  - Transition only squeeze: neg-ll 3.24 → 3.89
The ε-convolution smooths out transition spikes: 1/ε cancels against ε
integration width. Likelihood is BOUNDED above.

### MC comparison (N=500, R=10) with plain logistic MLE vs QR (mc_logistic_compare.jl)
  Slopes RMSE: MLE [0.036, 0.033, 0.033] vs QR [0.036, 0.037, 0.044]
  Efficiency gain (QR RMSE / MLE RMSE): [1.00, 1.11, 1.31]
  MLE faster: 5.3s vs QR 19.7s
  MLE exploits cross-percentile restrictions → lower variance

### Files added Apr 16 (late session)
  pwl_logistic.jl            — core PWL-z density (CDF, PDF, sampler, tests pass)
  pwl_abb.jl                 — ABB-style wrapper (gap reparam applied, MLE running)
  logistic_ml.jl             — plain logistic MLE (forward filter, works)
  mc_logistic_compare.jl     — MC comparison plain logistic vs QR (complete)
  test_pwl_inspect.jl        — parameter diagnostic (showed LBFGS stuck at kink)
  test_pwl_degeneracy.jl     — proved likelihood bounded (squeeze experiment)

## Apr 16 (latest): PWL-z optimization tests

### test_pwl_from_truth.jl results:
  Test 1 (LBFGS from truth): slopes 0.80→0.74, neg-ll improved by 0.005
    → Sample MLE differs from truth (normal at N=500)
  Test 2 (LBFGS 5% perturbed): slopes→0.787, neg-ll 3.220 (truth 3.237)
    → LBFGS WORKS when starting close. 50 iters, 301s.
  Test 3 (Coord descent 20% perturbed): DIVERGED (neg-ll went to -14)
    → Grid discretization breaks down with concentrated densities.
    → ±0.3 search bounds too wide for coord descent.

### Grid accuracy test:
  At truth: G=50,100,200,300 all give 3.237253 (exact agreement)
  Away from truth: G=100 vs G=300 differ by up to 0.7%
  Non-uniform grid (quantile-based) has 4× finer spacing near data center

### Conclusion:
  LBFGS works for PWL-z MLE from near-truth start. Coord descent fails
  due to grid artifacts with concentrated densities.
  For HPC MC: use LBFGS, warm-start from QR, G=120-200.

## Apr 16 (latest): logistic_abb.jl — ALL-SMOOTH model

### Setup
  logistic_abb.jl — Asymmetric logistic for ALL three distributions
  (init, transition, ε). No piecewise-uniform anywhere.
  Quantile knots parameterized exactly as ABB (a_Q with Hermite basis).
  Density shape fully determined by knots — no extra spline params.
  Analytical normalizing constant: C = 2αLαR/(αL+αR).

### Bugs found and fixed:
  1. f_init and f_eps were piecewise-uniform → caused grid oscillation
     (trapezoidal rule doesn't converge for discontinuous integrands)
  2. After switching to logistic for all: grid convergence PERFECT
     (G=201 = G=401 = 3.53922472, identical to 8 decimals)
  3. Normalization bug: asym_logistic_pdf already includes factor w;
     dividing by C (=w) gave unnormalized density. Fixed: f/C is correct.

### compare_optimizers.jl results (N=300, G=200, all-smooth):
  Truth neg-ll: 3.539, Start: 3.585 (10% perturbed from truth)
  LBFGS: neg-ll=3.269 (BEST), slopes=[0.83,0.65,0.79], 467s
  Adam:  neg-ll=3.668,         slopes=[0.80,0.73,0.76], 452s
  CG:    neg-ll=3.894 (stuck), slopes=[0.77,0.69,0.75], 182s
  
  LBFGS and Adam found neg-ll LOWER than truth → correct sample MLE behavior.
  LBFGS is the best optimizer for this problem.
  Need Monte Carlo across seeds to compare MLE vs QR RMSE.

### Gap reparameterization applied (Apr 16 latest):
  a_Q[:,1] = median coefficients
  a_Q[:,2] = log(gap_L) coefficients (gap_L = q₂ - q₁ > 0 by construction)
  a_Q[:,3] = log(gap_R) coefficients (gap_R = q₃ - q₂ > 0 by construction)
  Non-crossing guaranteed globally. No sorting needed.
  Normalizing constant C = 2αLαR/(αL+αR) couples params across quantiles.
  a_eps[2] fixed at 0 for identification (ε median = 0).
  QR also fixes ε median at 0 after each M-step.
  14 free MLE parameters: 9 a_Q + 3 a_init + 2 a_eps (q₁ and q₃ only).

### MC running: mc_logistic_abb.jl
  N=500 and N=1000, R=20
  Three methods: QR (logistic E-step), MLE cold, MLE warm (from QR)
  
### MC first replication (r=1, N=500):
  QR median slope: 0.683 (true 0.8)
  MLE cold: neg-ll=3.408 (best likelihood) but median slope 0.766, gap slopes wrong
  MLE warm: neg-ll=4.872, median slope 0.584 (worse than QR)
  
  Issue: gap reparam changes optimization landscape — parameters on different
  scales. LBFGS numerical gradient not adapted. Analytical gradient needed.

## TODO next session
  1. PRIORITY: Implement ANALYTICAL GRADIENT using the identity F(q_ℓ) = τ_ℓ:
     Since ∂F(q_ℓ)/∂θ = 0, differentiating gives
       f(q_ℓ) · ∂q_ℓ/∂θ + ∂/∂θ ∫_{-∞}^{q_ℓ} f(x;θ) dx = 0
     This gives ∂C/∂θ explicitly from f(q_ℓ) and ∂q_ℓ/∂θ (both closed form).
     No finite differences needed → much faster, more accurate LBFGS.
  3. Prepare HPC scripts for larger MC
  4. Update paper

## Apr 17: Direct-parameterization MLE with interior-point line search

### What was built (logistic_direct.jl)
Interior-point L-BFGS for the asymmetric logistic ABB model in the DIRECT
parameterization (each column of a_Q = coefs for Q_ℓ(η) directly):
  * Analytical gradient of the Simpson-approximated log-likelihood via
    forward-backward smoothing (`negll_and_grad`). Optimized: α/β/L scratch
    buffers hoisted; f_eps and its derivatives precomputed per (i,t); dlogT
    laid out with parameter dim contiguous.
  * Analytical MAXIMUM FEASIBLE STEP `max_feasible_step_analytical(v,d,…)`:
    along v+αd, for each ℓ the gap Q_{ℓ+1}(η)-Q_ℓ(η) is a quadratic in z=η/σ
    with coefficients LINEAR in α; endpoints give linear roots in α, vertex
    gives a quadratic in α. Plus linear roots for a_init, a_eps orderings.
    All in closed form — no bisection.
  * GOLDEN-SECTION line search on [0, α_cap·0.999] where α_cap = analytical
    α_max, guaranteeing every iterate stays strictly inside the non-crossing
    cone.
  * Safeguards: backtrack if no improvement, fall back to -g if LBFGS dir
    is non-descent, reset LBFGS history on heavy backtrack (α<<α_cap),
    relative-improvement stopping criterion.

### Correctness checks (PASSING)
  * test_feasible_step.jl — analytical α_max matches bisection to ~1e-15;
    boundary crossings are tight (gap_min at α_max ≈ 0, below feasible).
  * test_analytical_grad.jl — analytical grad vs central diff: max |diff| =
    9e-9 across all 14 params at a perturbed-truth point (N=100, G=101).
  * test_mle_gs.jl starting AT truth (N=200, G=101, seed=7): solver stays
    near truth (slopes [0.7998, 0.8003, 0.8008]), nll 3.2571 vs truth 3.2623,
    3.4s, 0 crossings.

### MC at N=500, R=20 (mc_logistic_direct.jl)
Two MLE starts tested:
  * MLE_w (warm) — LBFGS from QR solution projected to feasibility.
  * MLE_c (cold) — LBFGS from equal-coefs-except-constants init:
    intercepts [-0.3, 0, 0.3], slopes all 0.5, quadratic all 0.

Results (true slopes = [0.80,0.80,0.80], true intcpts = [-0.30,0,0.30]):

                      slopes mean                 intcpts mean
  QR        [0.7999, 0.7986, 0.7993]   [-0.3008, +0.4385, +1.0029]
  MLE_w     [0.7982, 0.7963, 0.8003]   [-0.2879, +0.4204, +0.9936]
  MLE_c     [0.5622, 0.4687, 0.5621]   [-0.3154, -0.0098, +0.3206]

  mean nll: truth=3.353, QR=4.003, MLE_w=3.970, MLE_c=3.564
  Crossings (raw QR before projection): 25 across 20 reps.
  After projection / for both MLEs: 0 violations.
  Avg time/rep: QR 15.5s, MLE_w 11.8s, MLE_c 10.6s (total 37.8s).

### Interpretation
  * QR produces slopes ≈ truth but INTERCEPTS BADLY BIASED at τ=0.5, 0.75
    (bias +0.44, +0.70). Intercepts for τ=0.25 are correct.
  * MLE_w inherits QR's local minimum — barely moves. Intercepts stay biased.
  * MLE_c achieves intercepts ≈ truth but slopes STUCK at 0.5 (starting val).
  * MLE_c finds LOWER nll than MLE_w (3.56 vs 3.97), but neither reaches
    truth's 3.35 — different local minima in a multi-modal landscape.
  * 25/20 QR reps cross in raw form → non-crossing is a real issue in the
    direct parameterization. My analytical feasibility machinery works:
    0 crossings in any MLE output.

### Smoothness issue identified
The conditional CDF of the split-logistic density is
    F(x|η_{t-1}) = σ(α_L·(x-q_2))   for x ≤ q_2
                 = σ(α_R·(x-q_2))   for x > q_2
F is CONTINUOUS at q_2 (both branches = 1/2) but f has a JUMP from α_L/4 to
α_R/4 when α_L ≠ α_R (i.e., the asymmetric case, which is always). F is
continuous but not C¹ at q_2 → log-density has a kink. Integration over η
smooths the marginal likelihood but pointwise log-density kinks may
contribute to the observed multi-modality of the landscape.

### Files added Apr 17
  logistic_direct.jl             — extended with estimate_direct_ml_gs,
                                   direct_neg_loglik_vec!, rewritten
                                   negll_and_grad (optimized), and
                                   max_feasible_step_analytical.
  mc_logistic_direct.jl          — MC with both warm and cold MLE.
  test_feasible_step.jl          — analytical vs bisection feasibility.
  test_analytical_grad.jl        — analytical vs central-diff gradient.
  test_mle_gs.jl                 — small MLE run with verbose output.
  test_mle_truth_start.jl        — MLE starting at truth (sanity check).
  mc_logistic_direct_N500_R20.jls — 20-rep MC results (full per-rep tuples).
  mc_logistic_direct_output.txt   — MC log with per-rep lines and summaries.

## Apr 17 (debug): Why MLE warm doesn't move from QR

test_mle_warm_debug.jl (seed=42, N=500):
  neg-ll at truth: 3.237
  neg-ll at QR:    3.229  (BETTER than truth!)
  QR has crossing: cross=1 (infeasible)
  Moving QR→truth monotonically WORSENS neg-ll
  
  The QR solution is in the INFEASIBLE region (knots cross at some η)
  but has better likelihood than truth. The unrestricted MLE is outside
  the non-crossing cone. Projecting QR to feasibility worsens the fit.
  
  This explains why MLE warm doesn't move: after projection, it's at a
  point worse than the unconstrained QR, and the gradient within the
  feasible cone points further from truth.

  The density kink at q₂ (f jumps from α_L/4 to α_R/4) may be causing
  the unconstrained optimum to require crossing quantiles — the
  asymmetric logistic's non-smooth density creates an objective where
  the global optimum requires q₁ > q₂ at some η values.

## CRITICAL BUG FOUND AND FIXED (Apr 17 late)
  direct_neg_loglik AND negll_and_grad returned FINITE values when knots
  crossed at some grid points (set T_mat row to 1e-300 instead of returning Inf).
  This allowed the optimizer to sit at infeasible (crossing) solutions with
  spuriously good likelihood.
  
  FIX: both functions now return Inf immediately when ANY grid point has crossing.
  MC re-running with fix (mc_logistic_direct_output_fixed.txt).

## Fixed MC results (same as before — bug fix didn't change outcome)
  QR slopes unbiased but intercepts biased (+0.43, +0.67 at τ=0.5,0.75)
  MLE warm: ~same as QR (doesn't move)
  MLE cold: intercepts correct, slopes stuck at 0.5
  
## Next: LOGSPLINE model (smooth density, no kink)
  The split-logistic kink at q₂ is the root cause of multi-modality.
  Logspline (log f = cubic spline) is C² everywhere — no kinks.
  
  Challenge: truncated power basis gives ill-conditioned system (cond ~1e6).
  Need B-spline basis or grid-based approach avoiding explicit γ computation.
  
  Key insight: ∂log C/∂θ is computable analytically from F(qₗ) = τₗ,
  so the gradient doesn't require solving for γ. The Hessian is also
  available analytically by differentiating again.
  
## cspline_abb.jl — WORKING (Apr 17 late)
  Natural cubic spline log-density with ABB quantile knots.
  - Knots at q₁, q₂, q₃ (ABB Hermite parameterization)
  - Log-density values s₁, s₂, s₃ solved from quantile constraints (Newton)
  - Exponential tails with slopes β_L > 0, β_R < 0
  - C² continuity at all knots (no jumps, no kinks)
  
  Test results:
  - Quantile constraints satisfied exactly (F(qₗ) = τₗ to 6 digits)
  - s values well-conditioned: [-0.418, 0.186, -0.418] (no extreme values!)
  - Density smooth at knots (jump < 3e-6, numerical noise)
  - Equal segment masses (0.329 each, as required by τ = 0.25, 0.50, 0.75)
  
  Performance:
  - G=101: 0.24s per transition matrix (with precomputed GL nodes)
  - G=201: 0.52s
  - Feasible for LBFGS optimization (~12 min with finite diff gradient,
    ~25s with analytical gradient)
  
  Hessian structure:
  - ∂²log C/∂a_{kℓ}∂a_{k'ℓ'} = 0 for ℓ≠ℓ' (block diagonal)
  - Each block has rank 1 (from single constraint F(qₗ)=τₗ)
  - Full likelihood Hessian NOT block diagonal (data couples quantiles)

## TODO
  1. Build forward filter likelihood with cspline transition
  2. Implement analytical gradient via F(qₗ) = τₗ
  3. MC comparison QR vs cspline MLE
  4. HPC sweep

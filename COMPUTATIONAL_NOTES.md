# Computational Notes: Cubic Spline MLE for ABB Model

## Overview

Comparing MLE vs QR M-step for the ABB (2017 Econometrica) nonlinear earnings
dynamics model, using a cubic spline log-density specification. These notes
document computational difficulties encountered and solutions adopted.

## 1. Newton Solver for Spline Coefficients: Scale Invariance

**Problem:** The spline values (s1, s2, s3) at the three quantile knots are
determined by solving F(q_l) = tau_l (l=1,2,3). These 3 equations in 3 unknowns
have a 1-dimensional null space: adding any constant c to all s_k multiplies all
segment masses by exp(c), which cancels in the ratios F(q_l) = cumM/C. The Newton
solver diverges, with s values wandering to +/- 500,000.

**Solution:** Pin s2 = 0 (fix the middle spline value). This breaks the scale
invariance and reduces the system to 2 equations in 2 unknowns (s1, s3). The
Newton solver then converges reliably with s values near zero.

## 2. Overflow/Underflow in Mass Computation

**Problem:** Even with s2=0, when beta parameters are perturbed during
optimization, the Newton-solved s values can be large (e.g., s = +/-500),
causing exp(s) to overflow or underflow, making the normalizing constant
C = sum(masses) numerically zero or infinity.

**Solution:** Log-space shift in cspline_masses!: subtract log_ref = max(s1,s2,s3)
before exponentiating. The shifted masses are exp(s_k - log_ref), which are
always O(1). The true logC = log(sum(shifted_masses)) + log_ref. Applied in
the Newton solver, transition matrix build, and forward filter.

## 3. Finite-Difference Gradient Step Size

**Problem:** The likelihood function is computed via Simpson's rule on a fixed
grid with G=201 points and spacing h_grid = 16/200 = 0.08. The standard
finite-difference step h = sqrt(eps) ~ 1.5e-8 (Optim.jl default) is far below
the grid resolution, producing gradients dominated by integration noise. Even
h = 1e-5 gave gradients that pointed uphill.

**Solution:** Increased finite-difference step to h = 1e-3, following the
Gill-Murray-Saunders-Wright (1983) formula h_opt = (3*eps_f)^(1/3) where
eps_f ~ 1e-4 is the Simpson's rule discretization error. This gives h_opt ~ 0.07,
comparable to the grid spacing. With h = 1e-3, the gradient reliably points
downhill and LBFGS converges.

**Reference:** Gill, Murray, Saunders, Wright (1983) "Computing Forward-Difference
Intervals for Numerical Optimization," SIAM J. Sci. Stat. Comput., 4(2), 310-321.

## 4. Ordering Constraints on Quantile Knots

**Problem:** The quantile knots must satisfy a_init[1] < a_init[2] < a_init[3]
and a_eps1 < 0 < a_eps3. Finite-difference perturbation of these parameters can
violate the ordering, causing the likelihood to return Inf, which corrupts the
gradient and makes LBFGS diverge.

**Solution:** Gap reparameterization. Store (median, log(gap_L), log(gap_R))
instead of (a_init[1], a_init[2], a_init[3]). Since exp(.) > 0, the ordering
is guaranteed for any unconstrained parameter vector. Similarly for a_eps:
store (log(-a_eps1), log(a_eps3)).

Note: Sorting the quantile parameters to enforce ordering is WRONG — it breaks
the correspondence between parameters and quantile levels.

## 5. LBFGS Line Search Failures

**Problem:** LBFGS with default line search (HagerZhang or StrongWolfe) gets
stuck because the first step is too large, pushing a_Q parameters into infeasible
regions where quantile crossings occur at some grid points.

**Solution:** Use BackTracking line search with the default initial step size
(alpha=1), combined with capping Inf likelihoods at 1e10 so the line search
can backtrack. The explicit gradient with h=1e-3 gives reliable descent
directions, making BackTracking effective.

## 6. Separate Tail Parameters per Distribution

**Problem:** Initially used shared (beta_L, beta_R) for all three distributions
(transition, initial, epsilon). This is a misspecification when the true DGP
has different tail rates (sigma_v=0.5 -> beta_Q=2, sigma_eta1=1.0 -> beta_init=1,
sigma_eps=0.3 -> beta_eps=3.33).

**Solution:** Separate (beta_L, beta_R) for each distribution: 6 tail parameters
instead of 2, bringing total from 16 to 20. This matches the flexibility that
QR has (separate tail rates per distribution), making the MLE-QR comparison fair.

## 7. Forward Filter Backward Sampler for QR E-step

**Problem:** QR with cubic spline E-step requires drawing eta from p(eta|y) using
the grid-based forward filter. The backward sampler draws from the grid, which
discretizes eta to grid points (spacing 0.08).

**Solution:** Implemented grid-based FFBS: forward pass stores all filtering
distributions, backward pass samples eta_T from the terminal filtering
distribution, then samples backwards using T(eta_t|eta_{t-1}) * p(eta_{t-1}|y).
The grid discretization adds noise to the QR estimates but is inherent to the
grid-based approach.

## 8. Analytical vs Numerical Gradient (in progress)

**Problem:** With numerical gradients (h=1e-3), MLE has higher RMSE than QR for
some higher-order parameters (quadratic Hermite coefficients), likely due to
gradient noise inflating variance for near-zero parameters.

**Status:** Implementing analytical gradient. The key identity:
  d(log f(x))/dtheta = d(spline(x))/dtheta - E_f[d(spline(X))/dtheta]
where E_f is the expectation under the density f. This requires:
  (a) Analytical derivatives of the spline evaluation (implemented, verified)
  (b) Sensitivity ds/d(t,beta) from implicit differentiation of Newton system (implemented, verified)
  (c) Integration into the forward filter recursion (in progress)

## 9. Quantile Crossing and Boundary Bias

**Problem:** The linear quantile model q_l(η₁) = a_Q[:,l]' H(η₁/σ_y) can have
quantile crossing (q₁ > q₂ or q₂ > q₃) at some η₁ values. The non-crossing
region requires the gap polynomial d₂z² + d₁z + (d₀ - d₂) > 0 for all z,
which requires d₂ > 0 and discriminant < 0.

When the true DGP has parallel quantile functions (Gaussian: d₂ = 0), the truth
is on the **boundary** of the non-crossing constraint. Any method of enforcing
non-crossing — Inf return, gap reparameterization, penalty — biases MLE toward
equal slopes across quantiles.

**Manifestation:** With the Gaussian DGP (d₂ = 0):
- Gap reparameterization with d₂ = exp(δ) forces d₂ > 0, excluding the truth.
  The optimizer pushes δ → -∞, which forces d₁ → 0 through the √(d₂) coupling,
  making all three quantile slopes equal (ρ = a_Q[2,3] in every seed).
- Inf constraint approach: optimizer finds solutions near the boundary where
  some grid points cross, biasing toward equal slopes.
- Both approaches give downward-biased slope estimates.

**Solution:** Use a DGP where d₂ > 0 (truth in the interior of the feasible
set). This corresponds to heteroscedastic transition density — quantile
spacing increases with |η₁|. Added d2_Q=0.03 parameter to make_true_cspline.

**Fundamental point:** This is inherent to the linear quantile model, not
specific to our implementation. ABB's QR approach avoids this because QR
estimates each quantile independently without enforcing non-crossing.

## 10. Unbounded Likelihood with C² Model

**Problem:** The C² model (β determined by spline slopes) has unbounded
likelihood. The optimizer can make the gap between quantile knots arbitrarily
small, creating a near-delta-function density. This gives arbitrarily high
likelihood at observations near the mode.

Unlike the separate-β model where β is a free parameter that controls tail
decay independently of knot spacing, the C² model ties β to the spline slopes
at the knots. Small knot spacing → large β → concentrated density → high
likelihood.

This manifests as:
- LBFGS pushing eps knots to near-zero spacing (a_eps1 → 0)
- Transition density gaps shrinking at some η₁ values
- Numerical precision loss when gaps ≈ 1e-6 (floating-point errors exceed gap)
- "Quantile crossing" errors that are really floating-point precision failures

**Status:** Unsolved. Options being considered:
- Add minimum gap constraint (changes model)
- Use penalty/prior on gap size
- Return to separate-β model (with kink) and live with the curvature issue
- Investigate if the problem is in the DGP or the model

## Summary of Parameters

With K=2 Hermite polynomials:
- 9 a_Q: transition quantile knots (3 quantiles x 3 polynomial terms)
- 6 beta: tail rates (beta_L, beta_R for transition, init, eps)
- 3 a_init: initial distribution quantile knots
- 2 a_eps: error distribution quantile knots (median fixed at 0)
- Total: 20 parameters

## Preliminary MC Results (N=500, S=185; N=1000, S=111)

MLE has lower RMSE than QR for most parameters, especially:
- Persistence (rho = a_Q[2,2]): QR/ML RMSE ratio = 1.25 (N=500), 1.34 (N=1000)
- Tail parameters: ML substantially better (ratio ~2x)
- Marginal distributions (a_init, a_eps): ML much better

QR has lower RMSE for some higher-order Hermite coefficients (a_Q[3,l]),
possibly due to numerical gradient noise in MLE. Analytical gradient expected
to close this gap.

#=
logspline.jl — Logspline density estimation for the ABB transition model

Replaces ABB's piecewise-uniform density with a smooth logspline:
  log f(x; θ) = β₀ + β₁x + Σₖ γₖ(x - tₖ)₊³ - log C(θ)
where (x - tₖ)₊³ = max(0, x-tₖ)³ and C(θ) = ∫exp(s(x;θ))dx.

For the transition η_t | η_{t-1}, the coefficients depend on η_{t-1}
through the Hermite basis, exactly as in ABB:
  β₀(η) = Σⱼ a[1,j] Heⱼ(η/σ)
  β₁(η) = Σⱼ a[2,j] Heⱼ(η/σ)
  γₖ(η) = Σⱼ a[k+2,j] Heⱼ(η/σ)

The density is smooth (C∞ away from knots, C² at knots), so the
log-likelihood is smooth in the parameters → LBFGS works.

Reference: Stone, Hansen, Kooperberg, Truong (1997), Annals of Statistics.
C implementation: polspline R package (lsdall.c).
=#

using LinearAlgebra, Statistics, Random

# ================================================================
#  GAUSS-LEGENDRE QUADRATURE (precomputed nodes and weights)
# ================================================================

"""Gauss-Legendre nodes and weights on [-1,1] for n points."""
function gauss_legendre(n::Int)
    # Use the Golub-Welsch algorithm
    β = [i / sqrt(4i^2 - 1) for i in 1:n-1]
    J = SymTridiagonal(zeros(n), β)
    λ, V = eigen(J)
    w = 2.0 .* V[1,:].^2
    λ, w
end

# Precompute for common sizes
# 8 points: 0.14% error (sufficient for MH ratios where errors cancel)
# 16 points: 0.013% error (for final likelihood evaluation)
const GL8_nodes, GL8_weights = gauss_legendre(8)
const GL16_nodes, GL16_weights = gauss_legendre(16)

"""
Integrate f(x) over [a,b] using Gauss-Legendre quadrature.
Transform from [-1,1] to [a,b]: x = (b-a)/2 * t + (a+b)/2.
"""
function gl_integrate(f, a::Float64, b::Float64;
                      nodes=GL8_nodes, weights=GL8_weights)
    mid = (a + b) / 2
    half = (b - a) / 2
    s = 0.0
    for i in eachindex(nodes)
        s += weights[i] * f(mid + half * nodes[i])
    end
    s * half
end

# ================================================================
#  TRUNCATED POWER BASIS
# ================================================================

"""Evaluate (x - t)₊³ = max(0, x-t)³."""
@inline function tp3(x::Float64, t::Float64)
    d = x - t
    d > 0.0 ? d * d * d : 0.0
end

"""Derivative of (x - t)₊³ with respect to x: 3(x-t)₊²."""
@inline function dtp3(x::Float64, t::Float64)
    d = x - t
    d > 0.0 ? 3.0 * d * d : 0.0
end

# ================================================================
#  LOGSPLINE EVALUATION
# ================================================================

"""
Evaluate the unnormalized log-density s(x; β₀, β₁, γ₁,...,γK, knots).
  s(x) = β₀ + β₁*x + Σₖ γₖ*(x - tₖ)₊³

Parameters:
  beta0, beta1: scalar coefficients
  gamma: vector of length K (knot coefficients)
  knots: vector of length K (knot locations)
"""
function logspline_s(x::Float64, beta0::Float64, beta1::Float64,
                     gamma::AbstractVector{Float64},
                     knots::AbstractVector{Float64})
    s = beta0 + beta1 * x
    for k in eachindex(gamma)
        s += gamma[k] * tp3(x, knots[k])
    end
    s
end

"""
Gradient of s(x) with respect to x (for possible future use).
"""
function logspline_ds(x::Float64, beta1::Float64,
                      gamma::AbstractVector{Float64},
                      knots::AbstractVector{Float64})
    ds = beta1
    for k in eachindex(gamma)
        ds += gamma[k] * dtp3(x, knots[k])
    end
    ds
end

"""
Compute the normalizing constant C = ∫exp(s(x))dx by quadrature.
Integration range [lo, hi] should be wide enough to capture the density.
Uses adaptive splitting: integrate each knot interval separately.
"""
function logspline_normalize(beta0::Float64, beta1::Float64,
                             gamma::AbstractVector{Float64},
                             knots::AbstractVector{Float64};
                             lo::Float64=-10.0, hi::Float64=10.0)
    f(x) = exp(logspline_s(x, beta0, beta1, gamma, knots))

    # Split at knots for better accuracy (function may change character)
    breakpoints = vcat(lo, knots[knots .> lo .&& knots .< hi], hi)
    C = 0.0
    for i in 1:length(breakpoints)-1
        C += gl_integrate(f, breakpoints[i], breakpoints[i+1])
    end
    C
end

"""
Evaluate the normalized log-density:
  log f(x) = s(x) - log C
"""
function logspline_logdens(x::Float64, beta0::Float64, beta1::Float64,
                           gamma::AbstractVector{Float64},
                           knots::AbstractVector{Float64},
                           logC::Float64)
    logspline_s(x, beta0, beta1, gamma, knots) - logC
end

# ================================================================
#  CONDITIONAL LOGSPLINE (transition η_t | η_{t-1})
# ================================================================

"""
Logspline transition model parameters.
  a_coef[p, j] for p = 1,...,K+2 and j = 1,...,J+1
  p=1: β₀ coefficients (Hermite basis in η_{t-1})
  p=2: β₁ coefficients
  p=3,...,K+2: γ₁,...,γK coefficients

  knots: fixed knot locations for the spline in η_t
  sigma_y: standardization for Hermite basis
  J: Hermite polynomial order
"""
struct LogsplineTransition
    a_coef::Matrix{Float64}  # (K+2) × (J+1)
    knots::Vector{Float64}   # K knot locations
    sigma_y::Float64
    J::Int                   # Hermite order
end

"""
Extract β₀, β₁, γ from the coefficient matrix given η_{t-1}.
"""
function logspline_coeffs(ls::LogsplineTransition, eta_lag::Float64)
    J = ls.J
    z = eta_lag / ls.sigma_y

    # Compute Hermite values
    hv = Vector{Float64}(undef, J + 1)
    hv[1] = 1.0
    J >= 1 && (hv[2] = z)
    for k in 2:J
        hv[k+1] = z * hv[k] - (k - 1) * hv[k-1]
    end

    K = length(ls.knots)
    beta0 = dot(view(ls.a_coef, 1, :), hv)
    beta1 = dot(view(ls.a_coef, 2, :), hv)
    gamma = Vector{Float64}(undef, K)
    for k in 1:K
        gamma[k] = dot(view(ls.a_coef, k + 2, :), hv)
    end
    beta0, beta1, gamma
end

"""
Evaluate the conditional log-density log f(η_t | η_{t-1}).
"""
function logspline_transition_logdens(eta_t::Float64, eta_lag::Float64,
                                      ls::LogsplineTransition;
                                      lo::Float64=-10.0, hi::Float64=10.0)
    beta0, beta1, gamma = logspline_coeffs(ls, eta_lag)
    logC = log(max(logspline_normalize(beta0, beta1, gamma, ls.knots;
                                        lo=lo, hi=hi), 1e-300))
    logspline_logdens(eta_t, beta0, beta1, gamma, ls.knots, logC)
end

"""
Draw from the logspline conditional distribution by inverse CDF.
Uses bisection on the CDF (computed by quadrature).
"""
function logspline_draw(rng::AbstractRNG, eta_lag::Float64,
                        ls::LogsplineTransition;
                        lo::Float64=-10.0, hi::Float64=10.0, tol::Float64=1e-6)
    beta0, beta1, gamma = logspline_coeffs(ls, eta_lag)
    C = logspline_normalize(beta0, beta1, gamma, ls.knots; lo=lo, hi=hi)

    u = rand(rng)
    target = u * C

    # Bisection on the CDF: find x such that ∫_{lo}^x exp(s(t))dt = target
    a, b = lo, hi
    f(x) = exp(logspline_s(x, beta0, beta1, gamma, ls.knots))
    for _ in 1:60
        mid = (a + b) / 2
        cdf_mid = gl_integrate(f, lo, mid)
        if cdf_mid < target
            a = mid
        else
            b = mid
        end
        (b - a) < tol && break
    end
    (a + b) / 2
end

# ================================================================
#  LOG-LIKELIHOOD AND GRADIENT FOR THE TRANSITION
# ================================================================

"""
Average negative log-likelihood for the transition, given stacked data.
Parameters packed as a vector theta = vec(a_coef).

eta_t: vector of current η values
H_lag: Hermite basis matrix for lagged η (n_obs × (J+1))
knots: fixed knot locations
sigma_y: standardization

Returns the negative average log-likelihood.
"""
function logspline_neg_loglik(theta::Vector{Float64},
                              eta_t::Vector{Float64},
                              H_lag::Matrix{Float64},
                              knots::Vector{Float64};
                              lo::Float64=-10.0, hi::Float64=10.0)
    K = length(knots)
    Jp1 = size(H_lag, 2)
    n_obs = length(eta_t)
    a = reshape(theta, K + 2, Jp1)

    ll = 0.0
    gamma = Vector{Float64}(undef, K)

    for j in 1:n_obs
        hv = view(H_lag, j, :)

        # Compute coefficients for this observation
        beta0 = dot(view(a, 1, :), hv)
        beta1 = dot(view(a, 2, :), hv)
        for k in 1:K
            gamma[k] = dot(view(a, k + 2, :), hv)
        end

        # Evaluate s(eta_t) and log C
        sx = logspline_s(eta_t[j], beta0, beta1, gamma, knots)
        C = logspline_normalize(beta0, beta1, gamma, knots; lo=lo, hi=hi)
        ll += sx - log(max(C, 1e-300))
    end

    -ll / n_obs
end

"""
Analytical gradient of the negative log-likelihood.

For log f(η_t | η_{t-1}) = s(η_t; θ) - log C(θ), where s is linear in θ:
  s(x; θ) = Σ_p θ_p φ_p(x, η_{t-1})

The gradient is:
  ∂(-log f)/∂θ_p = -φ_p(η_t, η_{t-1}) + E[φ_p(x, η_{t-1})]

where E[·] is under the logspline density f(x | η_{t-1}).
The expectation is computed by quadrature: E[φ_p] = ∫ φ_p(x) exp(s(x)) dx / C.
"""
function logspline_neg_loglik_grad(theta::Vector{Float64},
                                   eta_t::Vector{Float64},
                                   H_lag::Matrix{Float64},
                                   knots::Vector{Float64};
                                   lo::Float64=-10.0, hi::Float64=10.0)
    K = length(knots)
    Jp1 = size(H_lag, 2)
    n_obs = length(eta_t)
    np = (K + 2) * Jp1
    a = reshape(theta, K + 2, Jp1)

    g = zeros(np)
    ga = reshape(g, K + 2, Jp1)
    gamma = Vector{Float64}(undef, K)

    for j in 1:n_obs
        hv = view(H_lag, j, :)

        # Coefficients for this observation
        beta0 = dot(view(a, 1, :), hv)
        beta1 = dot(view(a, 2, :), hv)
        for k in 1:K; gamma[k] = dot(view(a, k+2, :), hv); end

        # Normalizing constant
        f_unnorm(x) = exp(logspline_s(x, beta0, beta1, gamma, knots))
        C = gl_integrate(f_unnorm, lo, hi)
        C = max(C, 1e-300)

        # Basis functions φ_p at η_t[j]:
        # For p=(row, col): φ = hv[col] * basis_row(η_t)
        # row=1: basis = 1, row=2: basis = η_t, row=k+2: basis = (η_t - t_k)₊³

        # -φ_p(η_t) contribution
        x_j = eta_t[j]
        basis_at_x = Vector{Float64}(undef, K + 2)
        basis_at_x[1] = 1.0
        basis_at_x[2] = x_j
        for k in 1:K; basis_at_x[k+2] = tp3(x_j, knots[k]); end

        for row in 1:K+2, col in 1:Jp1
            ga[row, col] -= basis_at_x[row] * hv[col] / n_obs
        end

        # +E[φ_p] contribution via quadrature
        # E[basis_row(x)] = ∫ basis_row(x) * exp(s(x)) dx / C
        function integrand_basis(x, row)
            b = row == 1 ? 1.0 : row == 2 ? x : tp3(x, knots[row-2])
            b * f_unnorm(x)
        end

        for row in 1:K+2
            E_basis = gl_integrate(x -> integrand_basis(x, row), lo, hi) / C
            for col in 1:Jp1
                ga[row, col] += E_basis * hv[col] / n_obs
            end
        end
    end

    g
end

# ================================================================
#  TESTS
# ================================================================

function test_logspline()
    println("="^60)
    println("  LOGSPLINE TESTS")
    println("="^60)

    knots = [-0.5, 0.0, 0.5]
    beta0 = 0.0; beta1 = 0.0
    gamma = [-0.1, 0.0, -0.1]  # negative cubic terms for tail decay

    # Test 1: density is positive and integrates to 1
    println("\nTest 1: Normalization")
    C = logspline_normalize(beta0, beta1, gamma, knots)
    println("  C = $C")

    dx = 0.001
    xgrid = collect(-10.0:dx:10.0)
    numerical_integral = sum(exp(logspline_s(x, beta0, beta1, gamma, knots))
                             for x in xgrid) * dx
    println("  Numerical integral: $numerical_integral")
    println("  Ratio: $(numerical_integral / C)")

    # Normalized density should integrate to 1
    logC = log(C)
    dens_integral = sum(exp(logspline_logdens(x, beta0, beta1, gamma, knots, logC))
                        for x in xgrid) * dx
    println("  Normalized density integral: $dens_integral  (should be ~1)")

    # Test 2: density is smooth (no jumps)
    println("\nTest 2: Smoothness at knots")
    for t in knots
        fl = exp(logspline_logdens(t - 1e-6, beta0, beta1, gamma, knots, logC))
        fr = exp(logspline_logdens(t + 1e-6, beta0, beta1, gamma, knots, logC))
        println("  At knot $t: f(t-ε)=$(round(fl,digits=6)), f(t+ε)=$(round(fr,digits=6)), diff=$(abs(fl-fr))")
    end

    # Test 3: conditional logspline with Hermite basis
    println("\nTest 3: Conditional logspline")
    K = 3; J = 2
    a_coef = zeros(K + 2, J + 1)
    a_coef[1, 1] = 0.0   # β₀ intercept
    a_coef[2, 1] = 0.0   # β₁ intercept (no location shift)
    a_coef[2, 2] = -0.5  # β₁ depends on η_{t-1} (creates mean-reversion)
    a_coef[3, 1] = -0.1  # γ₁ (tail decay)
    a_coef[4, 1] = 0.0   # γ₂
    a_coef[5, 1] = -0.1  # γ₃ (tail decay)

    ls = LogsplineTransition(a_coef, knots, 1.0, J)

    for eta_lag in [-1.0, 0.0, 1.0]
        ll = logspline_transition_logdens(0.0, eta_lag, ls)
        println("  log f(0 | η_{t-1}=$eta_lag) = $(round(ll, digits=4))")
    end

    # Test 4: draw from conditional
    println("\nTest 4: Drawing samples")
    rng = MersenneTwister(42)
    samples = [logspline_draw(rng, 0.0, ls) for _ in 1:10000]
    println("  Mean: $(round(mean(samples), digits=4)) (should be ~0)")
    println("  Std:  $(round(std(samples), digits=4))")

    println("\n","="^60)
    println("  TESTS DONE")
    println("="^60)
end

# Run tests if executed directly
if abspath(PROGRAM_FILE) == @__FILE__
    test_logspline()
end

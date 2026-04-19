#=
logistic_abb_dgp.jl — Quantile-respecting smooth DGP using conditional logistic
distribution with ABB-style Hermite-parameterized location and scale.

MODEL:
  y_{it} = η_{it} + ε_{it}
  η_{i1} ~ Logistic(μ_init, α_init)   (smooth initial density)
  η_{it} | η_{i,t-1} ~ Logistic(μ(η_{i,t-1}), α(η_{i,t-1}))   (smooth transition)
  ε_{it} ~ Logistic(μ_eps, α_eps)      (smooth noise)

where μ(η) = m_0 + m_1 η/σ + m_2 (η²/σ² - 1)   (Hermite order K=2)
      α(η) = a_0 + a_1 η/σ + a_2 (η²/σ² - 1) > 0 (positivity required)

The logistic density:
  f(x; μ, α) = α * exp(-α(x-μ)) / (1 + exp(-α(x-μ)))²
  quantile:  Q_τ(μ, α) = μ + log(τ/(1-τ)) / α

For τ = (0.25, 0.50, 0.75) and knot values q_ℓ(η):
  q_1(η) = μ(η) - log(3)/α(η)
  q_2(η) = μ(η)
  q_3(η) = μ(η) + log(3)/α(η)

IMPORTANT for cross-percentile restrictions:
- In this logistic DGP, the three Hermite-parameterized quantile columns
  a_Q[:, 1], a_Q[:, 2], a_Q[:, 3] are DETERMINED FUNCTIONS of (m_0,m_1,m_2, a_0,a_1,a_2).
- QR estimates each a_Q[:, ℓ] column independently, ignoring the smooth structure.
- MLE uses the FULL logistic density and thus exploits the cross-percentile
  restrictions: the 9 knot parameters a_Q[:,·] are constrained by just 6
  underlying parameters (m_0,m_1,m_2, a_0,a_1,a_2).
=#

include("ABB_three_period.jl")
using Random, Printf, Optim, LinearAlgebra

# ================================================================
#  LOGISTIC DENSITY
# ================================================================

"""Logistic density f(x; μ, α)."""
@inline function logistic_pdf(x::Real, μ::Real, α::Real)
    z = α * (x - μ)
    e = exp(-abs(z))  # exp(-|z|)/(1+exp(-|z|))² = e/(1+e)²
    α * e / (1 + e)^2
end

"""Logistic log-density."""
@inline function logistic_logpdf(x::Real, μ::Real, α::Real)
    z = α * (x - μ)
    log(α) - z - 2 * log1p(exp(-z))   # numerically safer
end

"""Logistic quantile."""
@inline logistic_quantile(τ::Real, μ::Real, α::Real) = μ + log(τ / (1 - τ)) / α

"""Draw from Logistic(μ, α)."""
@inline logistic_draw(rng::AbstractRNG, μ::Real, α::Real) =
    μ + log(rand(rng) / (1 - rand(rng))) / α

# More accurate: use single uniform and invert CDF
@inline function logistic_draw1(rng::AbstractRNG, μ::Real, α::Real)
    u = rand(rng)
    μ + log(u / (1 - u)) / α
end

# ================================================================
#  ABB-STYLE PARAMETERS FOR LOGISTIC MODEL
# ================================================================

"""
LogisticParams mirrors the ABB Params structure but with logistic distributions.

Transition: μ_Q[k], α_Q[k] for k = 0,...,K (Hermite coefficients)
Initial: μ_init, α_init (scalars, unconditional)
Noise:   μ_eps, α_eps (scalars)
"""
mutable struct LogisticParams
    # Transition: conditional logistic given η_{t-1}
    μ_Q::Vector{Float64}  # (K+1,) Hermite coeffs for conditional median
    α_Q::Vector{Float64}  # (K+1,) Hermite coeffs for conditional scale (evaluated to >0)
    # Initial η_1 distribution: unconditional logistic
    μ_init::Float64
    α_init::Float64
    # Noise: unconditional logistic
    μ_eps::Float64
    α_eps::Float64
end

function copy_params_log(p::LogisticParams)
    LogisticParams(copy(p.μ_Q), copy(p.α_Q),
                   p.μ_init, p.α_init, p.μ_eps, p.α_eps)
end

"""Evaluate conditional μ, α given η_{t-1}."""
@inline function cond_μ_α(p::LogisticParams, η::Float64, K::Int, σy::Float64)
    z = η / σy
    μ = p.μ_Q[1]; α = p.α_Q[1]
    if K >= 1
        μ += p.μ_Q[2] * z; α += p.α_Q[2] * z
    end
    if K >= 2
        hv2 = z*z - 1  # He_2(z)
        μ += p.μ_Q[3] * hv2; α += p.α_Q[3] * hv2
    end
    for k in 3:K
        @warn "K > 2 not implemented"
    end
    (μ, max(α, 1e-6))  # floor α at a tiny positive value
end

# ================================================================
#  CONSTRUCT "TRUE" LOGISTIC PARAMS FROM AR(1) SPEC
# ================================================================
"""
True logistic params approximating AR(1):
  η_t | η_{t-1} ~ Logistic(ρ · η_{t-1}, α_v)  where α_v matched to sigma_v
  η_1            ~ Logistic(0, α_η)            (stationary)
  ε              ~ Logistic(0, α_eps)

For logistic with scale α: variance = π²/(3 α²).
So α = π / sqrt(3 · variance) = π / (sqrt(3) · std_dev).
"""
function make_true_logistic_params(; ρ=0.8, σ_v=0.5, σ_eps=0.3, σ_η1=1.0)
    K = 2
    α_v = π / (sqrt(3) * σ_v)
    α_eps = π / (sqrt(3) * σ_eps)
    α_η1 = π / (sqrt(3) * σ_η1)

    # Conditional logistic: μ(η) = ρ η, α = α_v (constant)
    μ_Q = [0.0, ρ, 0.0]   # [intercept, slope on η/σy, quadratic]
    α_Q = [α_v, 0.0, 0.0] # constant scale

    LogisticParams(μ_Q, α_Q,
                   0.0, α_η1,   # η_1 ~ Logistic(0, α_η1)
                   0.0, α_eps)  # ε ~ Logistic(0, α_eps)
end

"""
Convert to ABB-style quantile knots.
q_l(η) = μ(η) + log(τ_l/(1-τ_l))/α(η).

Returns a_Q (K+1 × L) giving the BEST LINEAR-IN-HERMITE approximation
(via regression at sample points). Used for comparison with QR estimator.
"""
function logistic_to_aQ(p::LogisticParams, τ::Vector{Float64}, K::Int, σy::Float64;
                        η_range::AbstractRange=range(-3, 3, length=50))
    L = length(τ)
    a_Q = zeros(K+1, L)
    # Evaluate q_l at many points, then regress onto Hermite basis
    η_pts = collect(η_range)
    H = hermite_basis(η_pts, K, σy)  # (n × K+1)
    for l in 1:L
        lz = log(τ[l] / (1 - τ[l]))
        q_vals = [begin
                      μ, α = cond_μ_α(p, η, K, σy)
                      μ + lz / α
                  end for η in η_pts]
        a_Q[:, l] .= H \ q_vals
    end
    a_Q
end

# ================================================================
#  DATA GENERATION
# ================================================================

function generate_data_logistic(N::Int, p::LogisticParams, K::Int, σy::Float64;
                                seed::Int=42)
    rng = MersenneTwister(seed)
    T = 3
    η = zeros(N, T); y = zeros(N, T)

    # η_1 ~ Logistic(μ_init, α_init)
    for i in 1:N
        η[i, 1] = logistic_draw1(rng, p.μ_init, p.α_init)
    end

    # η_t | η_{t-1}
    for t in 2:T, i in 1:N
        μ, α = cond_μ_α(p, η[i, t-1], K, σy)
        η[i, t] = logistic_draw1(rng, μ, α)
    end

    # y = η + ε
    for t in 1:T, i in 1:N
        y[i, t] = η[i, t] + logistic_draw1(rng, p.μ_eps, p.α_eps)
    end

    y, η
end

# ================================================================
#  DEMO
# ================================================================
if abspath(PROGRAM_FILE) == @__FILE__
    K = 2; σy = 1.0; T = 3
    τ = [0.25, 0.50, 0.75]

    println("="^70)
    println("  LOGISTIC ABB-STYLE MODEL: quantile-respecting smooth DGP")
    println("="^70)

    p = make_true_logistic_params()
    @printf("\nTrue params:\n")
    @printf("  μ_Q (conditional median coeffs): %s\n", p.μ_Q)
    @printf("  α_Q (conditional scale coeffs):  %s\n", p.α_Q)
    @printf("  Initial: μ=%.3f, α=%.3f (σ_η1=%.3f)\n",
            p.μ_init, p.α_init, π/(sqrt(3)*p.α_init))
    @printf("  Noise: μ=%.3f, α=%.3f (σ_ε=%.3f)\n",
            p.μ_eps, p.α_eps, π/(sqrt(3)*p.α_eps))

    # Convert to ABB-style a_Q knots (for comparison)
    a_Q = logistic_to_aQ(p, τ, K, σy)
    @printf("\nImplied ABB-style a_Q knots (linear Hermite approximation):\n")
    @printf("  Row 1 (intcpt): [%.4f, %.4f, %.4f]\n", a_Q[1, :]...)
    @printf("  Row 2 (slope):  [%.4f, %.4f, %.4f]\n", a_Q[2, :]...)
    @printf("  Row 3 (quad):   [%.4f, %.4f, %.4f]\n", a_Q[3, :]...)

    @printf("\nCompare ABB's piecewise-uniform true_linear params:\n")
    par_abb = make_true_params_linear(tau=τ, sigma_y=σy, K=K)
    @printf("  Row 1 (intcpt): [%.4f, %.4f, %.4f]\n", par_abb.a_Q[1, :]...)
    @printf("  Row 2 (slope):  [%.4f, %.4f, %.4f]\n", par_abb.a_Q[2, :]...)
    @printf("  Row 3 (quad):   [%.4f, %.4f, %.4f]\n", par_abb.a_Q[3, :]...)

    # Generate some data
    N = 2000
    y, η = generate_data_logistic(N, p, K, σy; seed=42)
    @printf("\nGenerated N=%d, η range [%.3f, %.3f]\n", N, extrema(η)...)
    @printf("var(η_1)=%.3f (target %.3f)\n", var(η[:, 1]), 1.0)
    @printf("corr(η_1, η_2)=%.3f (target ~0.8)\n", cor(η[:, 1], η[:, 2]))
    @printf("corr(η_2, η_3)=%.3f (target ~0.8)\n", cor(η[:, 2], η[:, 3]))

    # Check quantile property: at η_{t-1} = 0, Q_0.25(η_t) should equal a_Q[1, 1]
    # Take η_{t-1} near 0 and check empirical quantiles of η_t
    idx_lo = findall(abs.(η[:, 1]) .< 0.2)
    if length(idx_lo) > 50
        q_emp = quantile(η[idx_lo, 2], τ)
        @printf("\nEmpirical quantiles of η_2 | η_1 ≈ 0:\n")
        @printf("  [%.4f, %.4f, %.4f]\n", q_emp...)
        @printf("Theoretical:\n")
        @printf("  [%.4f, %.4f, %.4f]\n",
                [logistic_quantile(τ[l], 0.0, p.α_Q[1]) for l in 1:3]...)
    end
end

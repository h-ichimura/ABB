#=
pwl_logistic.jl — Piecewise-linear-z Logistic CDF

Model:
  F(x) = σ(z(x))
where σ(t) = 1/(1+e^{-t}) is the logistic CDF and z(x) is piecewise LINEAR
in x with knots at q_1 < q_2 < q_3, such that
  z(q_ℓ) = logit(τ_ℓ)  for  ℓ = 1, 2, 3.

Free parameters:
  - Three knot locations (q_1, q_2, q_3)   — the conditional τ_ℓ quantiles
  - Two tail slopes α_L, α_R > 0           — control tail decay rates
Interior slopes α_1, α_2 are DETERMINED by knot continuity:
  α_ℓ = (logit(τ_{ℓ+1}) - logit(τ_ℓ)) / (q_{ℓ+1} - q_ℓ).

Density:
  f(x) = σ(z)·(1-σ(z))·z'(x)
where z'(x) = α_L, α_1, α_2, α_R in the four segments.
Density jumps at knots (like ABB's piecewise-uniform) but the OUTER CDF
is the smooth logistic — so the CDF is continuous and monotone.

For τ = [0.25, 0.50, 0.75]:
  logit(τ) = [-log 3, 0, log 3]
=#

# ================================================================
#  LOGISTIC UTILS
# ================================================================
@inline sigm(t) = 1 / (1 + exp(-t))
@inline logit(p) = log(p / (1 - p))

# Log of σ(z)(1-σ(z)) — stable version
@inline function log_sigmprime(z)
    # σ(z)(1-σ(z)) = e^{-|z|} / (1 + e^{-|z|})^2
    az = abs(z)
    -az - 2 * log1p(exp(-az))
end

# ================================================================
#  CORE PWL z, CDF, DENSITY
# ================================================================

"""
Compute z(x) and z'(x) at a given x.
q::NTuple{3,Float64}  sorted knot locations
τ::NTuple{3,Float64}  quantile levels (logit precomputed in logit_τ)
logit_τ::NTuple{3,Float64}  = logit.(τ)
α_L, α_R::Float64  tail slopes (positive)
Returns (z, z').
"""
@inline function pwl_z_and_slope(x::Float64,
                                  q::NTuple{3,Float64},
                                  logit_τ::NTuple{3,Float64},
                                  α_L::Float64, α_R::Float64)
    if x <= q[1]
        return (logit_τ[1] + α_L * (x - q[1]),  α_L)
    elseif x > q[3]
        return (logit_τ[3] + α_R * (x - q[3]),  α_R)
    elseif x <= q[2]
        α_int = (logit_τ[2] - logit_τ[1]) / (q[2] - q[1])
        return (logit_τ[1] + α_int * (x - q[1]),  α_int)
    else
        α_int = (logit_τ[3] - logit_τ[2]) / (q[3] - q[2])
        return (logit_τ[2] + α_int * (x - q[2]),  α_int)
    end
end

@inline pwl_cdf(x, q, logit_τ, αL, αR) = sigm(first(pwl_z_and_slope(x, q, logit_τ, αL, αR)))

@inline function pwl_logpdf(x::Float64, q::NTuple{3,Float64},
                             logit_τ::NTuple{3,Float64},
                             α_L::Float64, α_R::Float64)
    z, zp = pwl_z_and_slope(x, q, logit_τ, α_L, α_R)
    log_sigmprime(z) + log(zp)
end

@inline pwl_pdf(x, q, logit_τ, αL, αR) = exp(pwl_logpdf(x, q, logit_τ, αL, αR))

"""
Draw one sample by inverse CDF: given uniform u, find x.
Because the CDF is monotone and z is piecewise linear, inversion is closed-form.
"""
@inline function pwl_draw(u::Float64,
                           q::NTuple{3,Float64},
                           logit_τ::NTuple{3,Float64},
                           α_L::Float64, α_R::Float64,
                           τ::NTuple{3,Float64})
    if u <= τ[1]
        # Left tail: z(x) = logit_τ[1] + α_L (x - q[1]) and σ(z) = u
        z_target = logit(u)
        return q[1] + (z_target - logit_τ[1]) / α_L
    elseif u >= τ[3]
        z_target = logit(u)
        return q[3] + (z_target - logit_τ[3]) / α_R
    elseif u <= τ[2]
        z_target = logit(u)
        α_int = (logit_τ[2] - logit_τ[1]) / (q[2] - q[1])
        return q[1] + (z_target - logit_τ[1]) / α_int
    else
        z_target = logit(u)
        α_int = (logit_τ[3] - logit_τ[2]) / (q[3] - q[2])
        return q[2] + (z_target - logit_τ[2]) / α_int
    end
end

# ================================================================
#  QUICK TESTS
# ================================================================
using Printf, Random, Statistics

if abspath(PROGRAM_FILE) == @__FILE__

    println("="^70)
    println("  TEST: piecewise-linear-z logistic density")
    println("="^70)

    τ = (0.25, 0.50, 0.75)
    logit_τ = (logit(0.25), logit(0.50), logit(0.75))

    # Test 1: symmetric equally-spaced knots → should be plain logistic
    println("\nTest 1: q=(-0.303, 0, 0.303), α_L=α_R=3.628 → plain logistic?")
    q = (-0.303, 0.0, 0.303)
    α = 3.628  # matches σ_v = 0.5 via α = π/(√3·σ)

    # Verify CDF at knots equals τ
    for ℓ in 1:3
        F = pwl_cdf(q[ℓ], q, logit_τ, α, α)
        @printf("  F(q_%d = %.4f) = %.6f (want %.4f)\n", ℓ, q[ℓ], F, τ[ℓ])
    end

    # Interior slopes automatically equal α because knots are logit-spaced
    α1 = (logit_τ[2] - logit_τ[1]) / (q[2] - q[1])
    α2 = (logit_τ[3] - logit_τ[2]) / (q[3] - q[2])
    @printf("  Interior slopes: α_1=%.4f, α_2=%.4f (both should equal α_L=α_R=%.4f for plain logistic)\n",
            α1, α2, α)

    # Draw samples
    rng = MersenneTwister(42)
    N = 20000
    samples = [pwl_draw(rand(rng), q, logit_τ, α, α, τ) for _ in 1:N]
    @printf("  Empirical quantiles at τ: [%.4f, %.4f, %.4f] (want [%.4f, %.4f, %.4f])\n",
            quantile(samples, collect(τ))..., q...)
    @printf("  Sample mean=%.4f (should be 0), std=%.4f (should be %.4f)\n",
            mean(samples), std(samples), π/(sqrt(3)*α))

    # Test 2: ASYMMETRIC quantile spacing — plain logistic can't match this
    println("\nTest 2: q=(-1.0, 0.0, 0.3) asymmetric → only PWL z can fit!")
    q = (-1.0, 0.0, 0.3)
    α_L, α_R = 2.0, 5.0  # different tail slopes

    α1 = (logit_τ[2] - logit_τ[1]) / (q[2] - q[1])
    α2 = (logit_τ[3] - logit_τ[2]) / (q[3] - q[2])
    @printf("  Interior slopes: α_1=%.4f, α_2=%.4f (DIFFERENT — asymmetric spacing)\n", α1, α2)

    samples = [pwl_draw(rand(rng), q, logit_τ, α_L, α_R, τ) for _ in 1:N]
    @printf("  Empirical quantiles: [%.4f, %.4f, %.4f] (want [%.4f, %.4f, %.4f])\n",
            quantile(samples, collect(τ))..., q...)
    @printf("  Skewness (sample): %.4f  (plain logistic would give 0)\n",
            mean((samples .- mean(samples)).^3) / std(samples)^3)

    # Test 3: log-density and sampling consistency
    println("\nTest 3: Verify that density integrates to 1")
    # Split integral over 4 segments
    q = (-0.5, 0.0, 0.7); α_L = 1.5; α_R = 3.0
    # Numerical integration on grid
    dx = 0.0001
    xs = collect(-20:dx:20)
    integral = sum(pwl_pdf(x, q, logit_τ, α_L, α_R) for x in xs) * dx
    @printf("  ∫f dx (numerical, wide grid) = %.6f (should be 1.000)\n", integral)

    # Test 4: CDF at various τ
    println("\nTest 4: F at arbitrary τ — verify numerical quantile matches PWL prediction")
    for u in [0.1, 0.35, 0.5, 0.65, 0.9]
        x = pwl_draw(u, q, logit_τ, α_L, α_R, τ)
        F = pwl_cdf(x, q, logit_τ, α_L, α_R)
        @printf("  u=%.2f → x=%.4f, F(x)=%.6f (should equal u)\n", u, x, F)
    end
end

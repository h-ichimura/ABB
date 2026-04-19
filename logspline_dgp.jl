#=
logspline_dgp.jl — Generate data from a logspline transition model.

Strategy: Start from Gaussian AR(1) draws, then use the logspline
infrastructure to define a non-Gaussian transition that has the
right moments by construction.

Step 1: Generate η from Gaussian AR(1) with desired (ρ, σ_v)
Step 2: Apply a smooth monotone transformation g(·) to make it non-Gaussian
Step 3: The logspline MLE should be able to recover the transformed density
Step 4: Compare QR vs logspline MLE
=#

include("logspline.jl")
using Printf, Statistics, Random

# ================================================================
#  NON-GAUSSIAN TRANSITION VIA TRANSFORMATION
# ================================================================

"""
Smooth monotone transformation that introduces non-Gaussianity.
  g(x; κ) = x + κ * (x³/3)   for small κ
This preserves the sign and ordering, adds skewness/kurtosis.
For κ=0, identity (Gaussian). For κ>0, heavier right tail.
"""
function transform_ng(x::Float64, kappa::Float64)
    x + kappa * x^3 / 3.0
end

"""Inverse of g (numerical, by Newton's method)."""
function inv_transform_ng(y::Float64, kappa::Float64; tol=1e-12, maxiter=50)
    kappa == 0.0 && return y
    x = y  # initial guess
    for _ in 1:maxiter
        fx = x + kappa * x^3 / 3.0 - y
        fpx = 1.0 + kappa * x^2
        x -= fx / fpx
        abs(fx) < tol && break
    end
    x
end

# ================================================================
#  DATA GENERATION
# ================================================================

"""
Generate data from a non-Gaussian AR(1) model:
  v_t ~ N(0, σ_v²)                     (Gaussian innovation)
  η*_t = ρ * η*_{t-1} + v_t             (latent Gaussian AR(1))
  η_t = g(η*_t; κ)                      (non-Gaussian transformation)
  y_t = η_t + ε_t, ε ~ N(0, σ_ε²)

The transformation g introduces non-Gaussianity while preserving
the basic dynamic structure. The logspline should be able to capture
the resulting transition density.
"""
function generate_data_ngar1(N::Int, T::Int;
                              rho=0.8, sigma_v=0.5, sigma_eps=0.3,
                              kappa=0.1, seed=42)
    rng = MersenneTwister(seed)
    sigma_eta = sigma_v / sqrt(1 - rho^2)

    eta_star = zeros(N, T)  # latent Gaussian
    eta = zeros(N, T)       # transformed (non-Gaussian)
    y = zeros(N, T)

    for i in 1:N
        eta_star[i, 1] = sigma_eta * randn(rng)
        eta[i, 1] = transform_ng(eta_star[i, 1], kappa)
    end

    for t in 2:T, i in 1:N
        eta_star[i, t] = rho * eta_star[i, t-1] + sigma_v * randn(rng)
        eta[i, t] = transform_ng(eta_star[i, t], kappa)
    end

    for t in 1:T, i in 1:N
        y[i, t] = eta[i, t] + sigma_eps * randn(rng)
    end

    y, eta, eta_star
end

# ================================================================
#  REPORT
# ================================================================

function report_moments(y, eta; label="")
    N, T = size(y)
    println("  $label")
    println("  Observable moments (from y):")
    for t in 1:T
        @printf("    var(y_%d) = %.4f\n", t, var(y[:, t]))
    end
    for t in 2:T
        @printf("    corr(y_%d, y_%d) = %.4f\n", t-1, t, cor(y[:, t-1], y[:, t]))
    end
    if T >= 3
        @printf("    corr(y_1, y_3) = %.4f\n", cor(y[:, 1], y[:, 3]))
    end

    println("  Latent moments (from η):")
    for t in 1:T
        @printf("    mean(η_%d)=%+.4f  std(η_%d)=%.4f\n", t, mean(eta[:,t]), t, std(eta[:,t]))
    end
    for t in 2:T
        @printf("    corr(η_%d, η_%d) = %.4f\n", t-1, t, cor(eta[:, t-1], eta[:, t]))
        b = cov(eta[:,t-1], eta[:,t]) / var(eta[:,t-1])
        @printf("    slope(η_%d on η_%d) = %.4f\n", t, t-1, b)
    end
    for t in 2:T
        resid = eta[:, t] .- mean(eta[:, t])
        sk = mean(resid.^3) / std(eta[:, t])^3
        ku = mean(resid.^4) / std(eta[:, t])^4
        @printf("    η_%d: skewness=%.3f, kurtosis=%.3f (Gaussian: 0, 3)\n", t, sk, ku)
    end
end

# ================================================================
#  MAIN
# ================================================================

if abspath(PROGRAM_FILE) == @__FILE__
    N = 5000; T = 3
    rho = 0.8; sigma_v = 0.5; sigma_eps = 0.3

    println("="^60)
    println("  Non-Gaussian AR(1) via transformation")
    println("="^60)

    for kappa in [0.0, 0.05, 0.1, 0.2]
        println("\n--- κ = $kappa ---")
        y, eta, eta_star = generate_data_ngar1(N, T; rho=rho, sigma_v=sigma_v,
                                                sigma_eps=sigma_eps, kappa=kappa, seed=42)

        if any(abs.(eta) .> 100)
            println("  EXPLOSION at κ=$kappa")
            continue
        end

        report_moments(y, eta; label="κ=$kappa")
    end
end

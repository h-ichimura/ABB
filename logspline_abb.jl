#=
logspline_abb.jl — Logspline ABB model with smooth density

log f(x | η_{t-1}) = β₀ + β₁x + Σₗ γₗ(x - qₗ)₊³ - log C

where:
  qₗ(η_{t-1}) = Σₖ a_{kℓ} He_k(η_{t-1}/σ)   (ABB quantile knots)
  β₁(η_{t-1}) = Σₖ b_k He_k(η_{t-1}/σ)       (left tail rate)

  (β₀, γ₁, γ₂, γ₃) determined by 4 constraints:
    ∫f = 1,  F(q₁) = τ₁,  F(q₂) = τ₂,  F(q₃) = τ₃

The density is C² at knots, C∞ elsewhere. No jumps.

Parameters: a_Q (9) + b (3) + a_init (3) + a_eps (2) = 17 total

Gradient of log-likelihood uses:
  ∂/∂θ log f(x) = ∂s/∂θ - (1/C)∂C/∂θ
  The quantile constraint ∂F(qₗ)/∂θ = 0 gives ∂C/∂θ analytically.
=#

include("ABB_three_period.jl")
using Optim, Printf, LinearAlgebra

# ================================================================
#  TRUNCATED POWER BASIS
# ================================================================

@inline tp3(x, t) = (d = x - t; d > 0 ? d*d*d : 0.0)
@inline dtp3(x, t) = (d = x - t; d > 0 ? 3*d*d : 0.0)

# ================================================================
#  LOGSPLINE UNNORMALIZED LOG-DENSITY
# ================================================================

"""
Evaluate s(x) = β₀ + β₁x + Σₗ γₗ(x-qₗ)₊³
"""
function logspline_s(x::Float64, β0::Float64, β1::Float64,
                     γ::Vector{Float64}, q::Vector{Float64})
    s = β0 + β1 * x
    for l in eachindex(γ)
        s += γ[l] * tp3(x, q[l])
    end
    s
end

# ================================================================
#  SOLVE FOR (β₀, γ₁, γ₂, γ₃) GIVEN (q₁, q₂, q₃, β₁, τ)
#
#  4 equations:
#    ∫ exp(s(x)) dx = 1      (normalization)
#    ∫_{-∞}^{q₁} exp(s) dx / C = τ₁    (quantile 1)
#    ∫_{-∞}^{q₂} exp(s) dx / C = τ₂    (quantile 2)
#    ∫_{-∞}^{q₃} exp(s) dx / C = τ₃    (quantile 3)
#
#  4 unknowns: β₀, γ₁, γ₂, γ₃
#
#  Solve by Newton's method on the residuals.
# ================================================================

"""
Compute segment masses M₀,...,M₃,M₄ where:
  M₀ = ∫_{-∞}^{q₁} exp(s) dx   (left tail)
  M₁ = ∫_{q₁}^{q₂} exp(s) dx   (segment 1)
  M₂ = ∫_{q₂}^{q₃} exp(s) dx   (segment 2)
  M₃ = ∫_{q₃}^{∞} exp(s) dx    (right tail)

Left tail (x < q₁): s = β₀ + β₁x (all cubics zero),
  M₀ = exp(β₀ + β₁q₁) / β₁  (if β₁ > 0)

Interior segments: numerical integration (Gauss-Legendre)

Right tail (x > q₃): s has all cubic terms active.
  Need Σγₗ < 0 for integrability (cubic dominates).
  Integrate numerically from q₃ to some large cutoff.
"""
function segment_masses(β0::Float64, β1::Float64, γ::Vector{Float64},
                        q::Vector{Float64}; n_gl::Int=16, tail_cut::Float64=15.0)
    L = length(q)

    # Left tail: ∫_{-∞}^{q₁} exp(β₀ + β₁x) dx = exp(β₀ + β₁q₁)/β₁
    if β1 <= 0; return fill(Inf, L+1); end
    M0 = exp(β0 + β1 * q[1]) / β1

    # Gauss-Legendre nodes/weights on [-1,1]
    # Use pre-allocated or compute on the fly
    β_gl = [i / sqrt(4i^2 - 1) for i in 1:n_gl-1]
    J = SymTridiagonal(zeros(n_gl), β_gl)
    nodes, V = eigen(J)
    weights = 2.0 .* V[1,:].^2

    function gl_int(a, b)
        mid = (a+b)/2; half = (b-a)/2
        s = 0.0
        for i in 1:n_gl
            x = mid + half * nodes[i]
            s += weights[i] * exp(logspline_s(x, β0, β1, γ, q))
        end
        s * half
    end

    masses = zeros(L + 1)
    masses[1] = M0
    for l in 1:L-1
        masses[l+1] = gl_int(q[l], q[l+1])
    end
    # Right tail
    masses[L+1] = gl_int(q[L], tail_cut)

    masses
end

"""
Residual: given (β₀, γ₁, γ₂, γ₃), compute [R₁, R₂, R₃, R₄] where
  R₁ = M₀/C - τ₁
  R₂ = (M₀+M₁)/C - τ₂
  R₃ = (M₀+M₁+M₂)/C - τ₃
  R₄ = C - 1  (normalization, or equivalently log C)
where C = Σ masses.
"""
function logspline_residual(β0::Float64, β1::Float64, γ::Vector{Float64},
                            q::Vector{Float64}, τ::Vector{Float64})
    masses = segment_masses(β0, β1, γ, q)
    any(isinf, masses) && return fill(Inf, 4)
    C = sum(masses)
    cumM = cumsum(masses)
    # R₁ = cumM[1]/C - τ[1], R₂ = cumM[2]/C - τ[2], R₃ = cumM[3]/C - τ[3]
    R = zeros(4)
    for l in 1:3
        R[l] = cumM[l] / C - τ[l]
    end
    R[4] = log(C)  # want log C = 0 (i.e., C = 1)
    R
end

"""
Solve for (β₀, γ₁, γ₂, γ₃) given (q, β₁, τ) by Newton's method.
"""
function solve_logspline_coeffs(q::Vector{Float64}, β1::Float64,
                                τ::Vector{Float64};
                                maxiter::Int=50, tol::Float64=1e-10)
    L = length(q)
    # Initial guess: β₀ from normalization of left tail alone,
    # γ = larger negative to create sufficient tail decay
    β0 = -log(β1) - β1 * q[1]  # so exp(β₀ + β₁q₁)/β₁ ≈ 1/L
    γ = fill(-5.0, L)

    x = vcat(β0, γ)  # 4-vector: [β₀, γ₁, γ₂, γ₃]

    for iter in 1:maxiter
        β0_c = x[1]; γ_c = x[2:4]
        R = logspline_residual(β0_c, β1, γ_c, q, τ)
        any(isinf, R) && break
        norm(R) < tol && break

        # Jacobian by finite differences
        J = zeros(4, 4)
        h = 1e-7
        for j in 1:4
            xp = copy(x); xp[j] += h
            xm = copy(x); xm[j] -= h
            Rp = logspline_residual(xp[1], β1, xp[2:4], q, τ)
            Rm = logspline_residual(xm[1], β1, xm[2:4], q, τ)
            J[:, j] = (Rp .- Rm) ./ (2h)
        end

        # Newton step
        Δ = J \ (-R)
        # Damped step
        α = 1.0
        for _ in 1:10
            x_new = x .+ α .* Δ
            R_new = logspline_residual(x_new[1], β1, x_new[2:4], q, τ)
            if !any(isinf, R_new) && norm(R_new) < norm(R)
                x .= x_new
                break
            end
            α *= 0.5
        end
    end

    β0_out = x[1]; γ_out = x[2:4]
    (β0_out, γ_out)
end

# ================================================================
#  TEST
# ================================================================

function test_logspline_abb()
    println("="^60)
    println("  LOGSPLINE-ABB TEST")
    println("="^60)

    q = [-0.337, 0.0, 0.337]
    β1 = 2.0
    τ = [0.25, 0.50, 0.75]

    println("\nSolving for (β₀, γ) given q=$q, β₁=$β1...")
    β0, γ = solve_logspline_coeffs(q, β1, τ)
    @printf("  β₀ = %.6f\n", β0)
    @printf("  γ  = [%.6f, %.6f, %.6f]\n", γ...)

    # Verify: compute masses and check quantiles
    masses = segment_masses(β0, β1, γ, q)
    C = sum(masses)
    cumM = cumsum(masses)
    @printf("\n  C = %.6f (should be 1)\n", C)
    @printf("  F(q₁) = %.6f (should be 0.25)\n", cumM[1]/C)
    @printf("  F(q₂) = %.6f (should be 0.50)\n", cumM[2]/C)
    @printf("  F(q₃) = %.6f (should be 0.75)\n", cumM[3]/C)

    # Check density is smooth at knots
    println("\n  Density at knots (should be continuous):")
    for qi in q
        fl = exp(logspline_s(qi - 1e-6, β0, β1, γ, q)) / C
        fr = exp(logspline_s(qi + 1e-6, β0, β1, γ, q)) / C
        @printf("    f(%+.4f⁻) = %.6f,  f(%+.4f⁺) = %.6f,  jump = %.2e\n",
                qi, fl, qi, fr, abs(fr-fl))
    end

    # Density values
    println("\n  Density values:")
    for x in [-1.0, -0.5, -0.337, -0.1, 0.0, 0.1, 0.337, 0.5, 1.0]
        f = exp(logspline_s(x, β0, β1, γ, q)) / C
        @printf("    f(%.3f) = %.6f\n", x, f)
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    test_logspline_abb()
end

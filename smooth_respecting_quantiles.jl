#=
smooth_respecting_quantiles.jl — Smooth density where q_l(η) are the TRUE
τ_l-conditional-quantiles by construction.

log f(x|η) = β_0 + β_1 x + Σ_l γ_l (x - q_l(η))₊³ - log C(η)

Given (q_1, q_2, q_3, τ = [0.25, 0.50, 0.75], β_1 > 0), solve sequentially for
(γ_1, γ_2, γ_3) such that the segment masses match τ intervals. β_0 is then
determined by normalization.

Uses the fact that M_l^unnorm (segment mass) is monotone increasing in γ_l
(holding previous γ's fixed), so each 1D root-find is monotone and unique.
=#

include("ABB_three_period.jl")
using LinearAlgebra, Printf

# ================================================================
#  GAUSS-LEGENDRE QUADRATURE (32-point, high accuracy)
# ================================================================
function _gl_nodes(n)
    β = [i / sqrt(4i^2 - 1) for i in 1:n-1]
    J = SymTridiagonal(zeros(n), β)
    λ, V = eigen(J)
    λ, 2.0 .* V[1, :].^2
end

const GL32_n, GL32_w = _gl_nodes(32)
const GL64_n, GL64_w = _gl_nodes(64)

function gl32(f, a::Float64, b::Float64)
    mid = (a + b) / 2; half = (b - a) / 2
    s = 0.0
    for i in eachindex(GL32_n); s += GL32_w[i] * f(mid + half * GL32_n[i]); end
    s * half
end

function gl64(f, a::Float64, b::Float64)
    mid = (a + b) / 2; half = (b - a) / 2
    s = 0.0
    for i in eachindex(GL64_n); s += GL64_w[i] * f(mid + half * GL64_n[i]); end
    s * half
end

# ================================================================
#  UNNORMALIZED SEGMENT MASSES (functions of γ, given β_1 and q)
# ================================================================

"""
M_0 (left tail): ∫_{-∞}^{q_1} exp(β_1 x) dx = exp(β_1 q_1) / β_1.
Closed-form, requires β_1 > 0.
"""
M0_unnorm(β1::Float64, q1::Float64) = exp(β1 * q1) / β1

"""
M_1 = ∫_{q_1}^{q_2} exp(β_1 x + γ_1 (x-q_1)³) dx. Depends only on γ_1.
Monotone increasing in γ_1.
"""
function M1_unnorm(γ1::Float64, β1::Float64, q1::Float64, q2::Float64)
    gl32(x -> exp(β1 * x + γ1 * (x - q1)^3), q1, q2)
end

"""
M_2 = ∫_{q_2}^{q_3} exp(β_1 x + γ_1 (x-q_1)³ + γ_2 (x-q_2)³) dx.
Depends on γ_1, γ_2. Monotone increasing in γ_2.
"""
function M2_unnorm(γ1::Float64, γ2::Float64, β1::Float64,
                    q1::Float64, q2::Float64, q3::Float64)
    gl32(x -> exp(β1 * x + γ1 * (x - q1)^3 + γ2 * (x - q2)^3), q2, q3)
end

"""
M_3 = ∫_{q_3}^{∞} exp(β_1 x + Σ γ_l (x-q_l)³) dx.
Depends on γ_1, γ_2, γ_3. Requires Σ γ < 0 (for cubic decay).
Monotone increasing in γ_3 (holding γ_1, γ_2 fixed) when γ_3 is such that
Σ γ < 0 (so integral is finite).

Use adaptive truncation: integrate to q_3 + 30/max(|Σγ|^(1/3), 1).
"""
function M3_unnorm(γ1::Float64, γ2::Float64, γ3::Float64, β1::Float64,
                    q1::Float64, q2::Float64, q3::Float64)
    sγ = γ1 + γ2 + γ3
    sγ >= -1e-10 && return Inf   # improper
    # Right tail decays like exp((Σγ) x³); find x_cap such that log-density -50
    # (Σγ)(x_cap - q_3)³ ≈ -50 (ignoring lower-order terms)
    x_cap = q3 + max( (50.0 / -sγ)^(1/3), 2.0 )
    # Split integral into a few pieces for accuracy
    n_split = 4
    xs = range(q3, x_cap, length=n_split+1)
    s = 0.0
    for i in 1:n_split
        s += gl64(x -> exp(β1 * x + γ1 * (x - q1)^3 + γ2 * (x - q2)^3 + γ3 * (x - q3)^3),
                   xs[i], xs[i+1])
    end
    s
end

# ================================================================
#  SEQUENTIAL SOLVER (monotone 1D root finds)
# ================================================================

"""
Solve monotonically increasing f(γ) = target on ℝ using bracketing + bisection
followed by a few Newton refinements.

Takes advantage of the known monotonicity to bracket the root reliably.
"""
function monotone_solve(f::Function, target::Float64;
                         γ_init::Float64=0.0, tol::Float64=1e-10,
                         max_iter::Int=100)
    # Initial bracket: find γ_lo with f(γ_lo) < target and γ_hi with f(γ_hi) > target
    γ_lo = γ_init
    γ_hi = γ_init
    f_at_init = f(γ_init)
    if f_at_init < target
        γ_hi = γ_init + 1.0
        while f(γ_hi) < target
            γ_lo = γ_hi
            γ_hi += abs(γ_hi) + 1.0
            γ_hi > 1e6 && error("monotone_solve: upper bracket exceeded 1e6")
        end
    elseif f_at_init > target
        γ_lo = γ_init - 1.0
        while f(γ_lo) > target
            γ_hi = γ_lo
            γ_lo -= abs(γ_lo) + 1.0
            γ_lo < -1e6 && error("monotone_solve: lower bracket exceeded -1e6")
        end
    else
        return γ_init
    end
    # Bisection
    for _ in 1:max_iter
        mid = (γ_lo + γ_hi) / 2
        fm = f(mid)
        if fm < target
            γ_lo = mid
        else
            γ_hi = mid
        end
        abs(γ_hi - γ_lo) < tol && return mid
    end
    (γ_lo + γ_hi) / 2
end

"""
Given (q_1, q_2, q_3) sorted, τ = [0.25, 0.50, 0.75], and β_1 > 0,
solve sequentially for (β_0, γ_1, γ_2, γ_3) such that:
  F(q_1) = τ_1, F(q_2) = τ_2, F(q_3) = τ_3, ∫ f = 1.

Returns (β_0, γ_1, γ_2, γ_3).
"""
function solve_smooth_coeffs(q::Vector{Float64}, τ::Vector{Float64}, β1::Float64)
    @assert length(q) == 3 "Currently L=3 only"
    @assert length(τ) == 3
    @assert issorted(q)
    @assert β1 > 0 "β_1 must be positive for integrable left tail"

    q1, q2, q3 = q[1], q[2], q[3]
    τ1, τ2, τ3 = τ[1], τ[2], τ[3]

    # Segment masses
    m_0 = τ1
    m_1 = τ2 - τ1
    m_2 = τ3 - τ2
    m_3 = 1 - τ3

    # Reference: left-tail mass (depends only on β_1, q_1)
    M0 = M0_unnorm(β1, q1)
    # Target ratios: M_k^u / M_0 = m_k / m_0
    r1 = m_1 / m_0
    r2 = m_2 / m_0
    r3 = m_3 / m_0

    # Step 1: solve for γ_1 such that M_1^u(γ_1) / M_0 = r_1
    γ1 = monotone_solve(γ -> M1_unnorm(γ, β1, q1, q2) / M0, r1)

    # Step 2: solve for γ_2 such that M_2^u(γ_1, γ_2) / M_0 = r_2
    γ2 = monotone_solve(γ -> M2_unnorm(γ1, γ, β1, q1, q2, q3) / M0, r2)

    # Step 3: solve for γ_3 such that M_3^u(γ_1, γ_2, γ_3) / M_0 = r_3
    # γ_3 must make Σ γ < 0. Since γ_1 + γ_2 may be positive, need γ_3 < -(γ_1 + γ_2).
    # Monotone in γ_3: smaller γ_3 → faster right-tail decay → smaller M_3^u.
    # As γ_3 → -∞, M_3^u → 0. As γ_3 → -(γ_1 + γ_2)⁻, M_3^u → ∞.
    # So monotone decreasing-in-γ_3 from ∞ to 0 on the feasible range.
    # Wait — let me recheck. The integrand exp(γ_3 (x-q_3)³) for x > q_3:
    #   larger γ_3 (closer to 0 from below) → slower decay → larger M_3^u
    #   so M_3^u is INCREASING in γ_3 (in the feasible range).
    γ3 = monotone_solve(γ -> M3_unnorm(γ1, γ2, γ, β1, q1, q2, q3) / M0, r3;
                         γ_init=-(γ1 + γ2) - 0.5)

    # Step 4: normalization → β_0
    # Total unnormalized mass (including M_0): M_0 + M_1 + M_2 + M_3
    # Z = exp(β_0) * total, want Z = 1 → β_0 = -log(total)
    M1v = M1_unnorm(γ1, β1, q1, q2)
    M2v = M2_unnorm(γ1, γ2, β1, q1, q2, q3)
    M3v = M3_unnorm(γ1, γ2, γ3, β1, q1, q2, q3)
    total = M0 + M1v + M2v + M3v
    β0 = -log(total)

    β0, γ1, γ2, γ3
end

# ================================================================
#  DENSITY EVALUATION
# ================================================================
function smooth_logf(x::Float64, β0::Float64, β1::Float64, γ::NTuple{3,Float64},
                     q::NTuple{3,Float64})
    s = β0 + β1 * x
    @inbounds for l in 1:3
        d = x - q[l]
        d > 0 && (s += γ[l] * d^3)
    end
    s
end

function smooth_cdf(x::Float64, β0::Float64, β1::Float64, γ::NTuple{3,Float64},
                     q::NTuple{3,Float64})
    # Piecewise integration from -∞ to x
    if x <= q[1]
        # Left tail: exp(β_0 + β_1 t) integrated to x
        return exp(β0 + β1 * x) / β1
    end
    acc = exp(β0 + β1 * q[1]) / β1
    h(t) = exp(smooth_logf(t, β0, β1, γ, q))
    if x <= q[2]
        acc += gl64(h, q[1], x); return acc
    end
    acc += gl64(h, q[1], q[2])
    if x <= q[3]
        acc += gl64(h, q[2], x); return acc
    end
    acc += gl64(h, q[2], q[3])
    acc += gl64(h, q[3], x); return acc
end

# ================================================================
#  TEST
# ================================================================
if abspath(PROGRAM_FILE) == @__FILE__
    println("="^70)
    println("  TEST: sequential solve_smooth_coeffs")
    println("="^70)

    τ = [0.25, 0.50, 0.75]

    test_cases = [
        ("symmetric mild",   [-0.67, 0.0, 0.67],   0.5),
        ("ABB-style, β_1=1", [-0.337, 0.0, 0.337], 1.0),
        ("asymmetric",        [-1.0, 0.2, 1.5],     0.3),
        ("β_1 small (near PU)", [-0.337, 0.0, 0.337], 0.05),
        ("β_1 large",         [-0.337, 0.0, 0.337], 5.0),
    ]

    for (name, q, β1) in test_cases
        println("\n--- $name: q=$q, β_1=$β1 ---")
        try
            β0, γ1, γ2, γ3 = solve_smooth_coeffs(q, τ, β1)
            @printf("  β_0=%.4f, γ=[%.4f, %.4f, %.4f]\n", β0, γ1, γ2, γ3)
            @printf("  Σγ = %.4f (should be <0 for integrability)\n", γ1+γ2+γ3)

            γt = (γ1, γ2, γ3); qt = (q[1], q[2], q[3])
            F1 = smooth_cdf(q[1], β0, β1, γt, qt)
            F2 = smooth_cdf(q[2], β0, β1, γt, qt)
            F3 = smooth_cdf(q[3], β0, β1, γt, qt)
            # Total mass
            Fmax = smooth_cdf(q[3] + 30.0, β0, β1, γt, qt)
            @printf("  F(q_1)=%.6f (want %.4f)  err=%.2e\n", F1, τ[1], F1-τ[1])
            @printf("  F(q_2)=%.6f (want %.4f)  err=%.2e\n", F2, τ[2], F2-τ[2])
            @printf("  F(q_3)=%.6f (want %.4f)  err=%.2e\n", F3, τ[3], F3-τ[3])
            @printf("  F(∞ approx)=%.6f (want 1.0000)\n", Fmax)
        catch e
            println("  FAILED: $e")
        end
    end
end

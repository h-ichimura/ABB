#=
smooth_quantile_dgp.jl — Quantile-respecting smooth DGP.

MODEL (same as ABB but with smooth density between knots):
  y_{it}   = η_{it} + ε_{it}
  η_{it} | η_{i,t-1} ~ smooth density with knots q_l(η_{t-1}) at τ = (0.25,0.5,0.75)
  where q_l(η) = h(η)' a_Q[:, l]  (Hermite basis, same as ABB)

TRANSITION DENSITY: unique smooth density such that
  - log density is cubic spline with knots q_1 < q_2 < q_3
  - exp(β_1 x) left tail, rate β_1 > 0
  - β_0, γ_1, γ_2, γ_3 determined by quantile constraints F(q_l) = τ_l and ∫f=1
  - β_1 is a shape parameter (scalar)

Given (a_Q, β_1), the per-η density is fully determined (no free γ's —
 γ's are IMPLICIT FUNCTIONS of (q_l(η), β_1)).

IMPORTANT: this imposes cross-percentile restrictions that QR ignores.
QR estimates each column a_Q[:, l] independently minimizing check-loss;
MLE uses the FULL density, including the γ's implied by (a_Q, β_1).

For ε and initial η_1, keep piecewise-uniform (ABB form).
=#

include("ABB_three_period.jl")
using LinearAlgebra, Printf, Random

# ================================================================
#  GAUSS-LEGENDRE
# ================================================================
function _gl_nodes(n)
    β = [i / sqrt(4i^2 - 1) for i in 1:n-1]
    J = SymTridiagonal(zeros(n), β)
    λ, V = eigen(J)
    λ, 2.0 .* V[1, :].^2
end
const GLn, GLw = _gl_nodes(64)

function gl64(f, a::Float64, b::Float64)
    mid = (a + b) / 2; half = (b - a) / 2
    s = 0.0
    @inbounds for i in eachindex(GLn); s += GLw[i] * f(mid + half * GLn[i]); end
    s * half
end

# ================================================================
#  MASS FUNCTIONS (unnormalized, given β_1 and q)
# ================================================================
# Unnormalized density at x given (β_1, γ, q):
@inline function _h(x, β1, γ, q)
    s = β1 * x
    @inbounds for l in 1:3
        d = x - q[l]
        d > 0 && (s += γ[l] * d^3)
    end
    exp(s)
end

# M_0 = ∫_{-∞}^{q_1} exp(β_1 x) dx = exp(β_1 q_1)/β_1   (closed form)
M0u(β1, q1) = exp(β1 * q1) / β1

# M_k depend only on γ_1, ..., γ_k (and β_1, q). Only lower-indexed γ matter in segment k.
function M1u(γ1, β1, q1, q2)
    gl64(x -> _h(x, β1, (γ1, 0.0, 0.0), (q1, 1e30, 2e30)), q1, q2)
end
function M2u(γ1, γ2, β1, q1, q2, q3)
    gl64(x -> _h(x, β1, (γ1, γ2, 0.0), (q1, q2, 3e30)), q2, q3)
end
function M3u(γ1, γ2, γ3, β1, q1, q2, q3)
    sγ = γ1 + γ2 + γ3
    sγ >= -1e-10 && return Inf
    # Right tail: density ~ exp(Σγ · x³) → cubic decay; integrate to large x where log-density -50
    x_cap = q3 + max((50.0 / (-sγ))^(1/3), 2.0)
    n_split = 6  # split for accuracy
    xs = range(q3, x_cap, length=n_split+1)
    s = 0.0
    for i in 1:n_split
        s += gl64(x -> _h(x, β1, (γ1, γ2, γ3), (q1, q2, q3)), xs[i], xs[i+1])
    end
    s
end

# ================================================================
#  MONOTONE 1D ROOT-FIND (robust bracket + bisection + Newton polish)
# ================================================================
function monotone_solve(f::Function, target::Float64;
                         γ_init::Float64=0.0, tol::Float64=1e-10,
                         γ_min::Float64=-1e4, γ_max::Float64=1e4,
                         max_iter::Int=200)
    # Phase 1: find bracket [γ_lo, γ_hi] with f(γ_lo) < target < f(γ_hi)
    f0 = f(γ_init)
    if abs(f0 - target) < tol * max(1.0, abs(target))
        return γ_init
    end
    if f0 < target
        γ_lo = γ_init; γ_hi = γ_init + max(abs(γ_init), 1.0)
        while true
            γ_hi >= γ_max && error("monotone_solve: upper bracket γ_max=$γ_max exceeded")
            fh = f(γ_hi)
            fh >= target && break
            γ_lo = γ_hi
            γ_hi = min(γ_hi * 2 + 1.0, γ_max)
        end
    else  # f0 > target
        γ_hi = γ_init; γ_lo = γ_init - max(abs(γ_init), 1.0)
        while true
            γ_lo <= γ_min && error("monotone_solve: lower bracket γ_min=$γ_min exceeded")
            fl = f(γ_lo)
            fl <= target && break
            γ_hi = γ_lo
            γ_lo = max(γ_lo * 2 - 1.0, γ_min)
        end
    end

    # Phase 2: bisection
    for _ in 1:max_iter
        mid = (γ_lo + γ_hi) / 2
        fm = f(mid)
        if abs(fm - target) < tol * max(1.0, abs(target))
            return mid
        end
        if fm < target
            γ_lo = mid
        else
            γ_hi = mid
        end
        γ_hi - γ_lo < tol * max(1.0, abs(mid)) && return mid
    end
    (γ_lo + γ_hi) / 2
end

# ================================================================
#  SOLVE (β_0, γ_1, γ_2, γ_3) given (q, τ, β_1)
# ================================================================
"""
Given q_1 < q_2 < q_3 (vector), τ = [τ_1, τ_2, τ_3], and β_1 > 0,
solve for (β_0, γ_1, γ_2, γ_3) such that:
  ∫ f(x) dx = 1  (normalization)
  F(q_l) = τ_l  for l = 1, 2, 3

Returns (β_0, (γ_1, γ_2, γ_3)).

Requires β_1 > 0 (left tail) and yields γ_1 + γ_2 + γ_3 < 0 (right tail).
"""
function solve_coeffs(q::AbstractVector{<:Real}, τ::AbstractVector{<:Real},
                       β1::Real; verbose::Bool=false)
    @assert length(q) == 3 && length(τ) == 3
    @assert q[1] < q[2] < q[3]
    @assert β1 > 0
    q1, q2, q3 = Float64(q[1]), Float64(q[2]), Float64(q[3])
    τ1, τ2, τ3 = Float64(τ[1]), Float64(τ[2]), Float64(τ[3])
    β1_f = Float64(β1)

    # Required segment masses (normalized): m_k (summing to 1)
    m0 = τ1; m1 = τ2 - τ1; m2 = τ3 - τ2; m3 = 1 - τ3

    # Target: M_k^u / M_0^u = m_k / m_0 for k = 1, 2, 3
    M0 = M0u(β1_f, q1)
    T1 = m1 / m0; T2 = m2 / m0; T3 = m3 / m0  # target ratios

    # Step 1: γ_1 such that M_1^u(γ_1)/M_0 = T_1
    γ1 = monotone_solve(γ -> M1u(γ, β1_f, q1, q2) / M0, T1)
    verbose && @printf("  γ_1 = %.4f  (M_1/M_0 target %.4f, got %.4f)\n",
                       γ1, T1, M1u(γ1, β1_f, q1, q2) / M0)

    # Step 2: γ_2 such that M_2^u(γ_1, γ_2)/M_0 = T_2
    γ2 = monotone_solve(γ -> M2u(γ1, γ, β1_f, q1, q2, q3) / M0, T2)
    verbose && @printf("  γ_2 = %.4f  (M_2/M_0 target %.4f, got %.4f)\n",
                       γ2, T2, M2u(γ1, γ2, β1_f, q1, q2, q3) / M0)

    # Step 3: γ_3 such that M_3^u(γ_1, γ_2, γ_3)/M_0 = T_3
    # γ_3 < -(γ_1 + γ_2) for right-tail integrability.
    # Reparameterize: γ_3 = -(γ_1 + γ_2) - exp(t), t ∈ ℝ
    #   t → +∞: γ_3 → -∞ (M_3 → 0)
    #   t → -∞: γ_3 → -(γ_1+γ_2)⁻ (M_3 → ∞)
    # So M_3 is DECREASING in t. Use monotone_solve with target = -T_3 on -M_3(t)... or just negate.
    neg_shift = -(γ1 + γ2)  # this is the asymptote
    # Find t such that M_3(γ_3 = neg_shift - exp(t)) = T_3 * M_0
    function f_t(t)
        γ = neg_shift - exp(t)
        M3u(γ1, γ2, γ, β1_f, q1, q2, q3) / M0
    end
    # f_t is DECREASING in t (since larger t → more negative γ_3 → smaller M_3)
    # We want f_t(t) = T_3 = 1. Bracket: at t = 0, γ_3 = neg_shift - 1 (moderate).
    # Find bracket: expand t upward if f_t(0) > T_3, downward if f_t(0) < T_3.
    function solve_decreasing(f::Function, target::Float64; t0::Float64=0.0, tol::Float64=1e-10)
        f0 = f(t0)
        if abs(f0 - target) < tol * max(1.0, abs(target)); return t0; end
        if f0 > target  # need larger t to decrease f
            t_lo = t0; t_hi = t0 + 1.0
            while f(t_hi) > target
                t_lo = t_hi
                t_hi += max(abs(t_hi), 1.0)
                t_hi > 500 && error("solve_decreasing: upper t exceeded")
            end
        else  # f0 < target, need smaller t
            t_hi = t0; t_lo = t0 - 1.0
            while f(t_lo) < target
                t_hi = t_lo
                t_lo -= max(abs(t_lo), 1.0)
                t_lo < -500 && error("solve_decreasing: lower t exceeded")
            end
        end
        # Bisection (f decreasing → standard bisection with target)
        for _ in 1:200
            mid = (t_lo + t_hi) / 2
            fm = f(mid)
            if abs(fm - target) < tol * max(1.0, abs(target)); return mid; end
            if fm > target
                t_lo = mid
            else
                t_hi = mid
            end
            t_hi - t_lo < 1e-12 && return mid
        end
        (t_lo + t_hi) / 2
    end
    t_opt = solve_decreasing(f_t, T3)
    γ3 = neg_shift - exp(t_opt)
    verbose && @printf("  γ_3 = %.4f  (M_3/M_0 target %.4f, got %.4f)\n",
                       γ3, T3, M3u(γ1, γ2, γ3, β1_f, q1, q2, q3) / M0)

    # Normalization: β_0 = -log(total unnormalized mass)
    M1v = M1u(γ1, β1_f, q1, q2)
    M2v = M2u(γ1, γ2, β1_f, q1, q2, q3)
    M3v = M3u(γ1, γ2, γ3, β1_f, q1, q2, q3)
    total = M0 + M1v + M2v + M3v
    β0 = -log(total)

    β0, (γ1, γ2, γ3)
end

# ================================================================
#  DENSITY EVALUATION
# ================================================================
function smooth_logdens(x::Float64, β0::Float64, β1::Float64,
                         γ::NTuple{3,Float64}, q::NTuple{3,Float64})
    s = β0 + β1 * x
    @inbounds for l in 1:3
        d = x - q[l]
        d > 0 && (s += γ[l] * d^3)
    end
    s
end

smooth_dens(x, β0, β1, γ, q) = exp(smooth_logdens(x, β0, β1, γ, q))

# ================================================================
#  INVERSE-CDF SAMPLER
# ================================================================
"""
Draw one sample from the smooth density with parameters (β_0, β_1, γ, q)
using inverse CDF (bisection).
"""
function smooth_draw(rng::AbstractRNG, β0::Float64, β1::Float64,
                      γ::NTuple{3,Float64}, q::NTuple{3,Float64};
                      x_cap::Float64=Inf)
    u = rand(rng)

    # CDF piecewise:
    # F(x) for x ≤ q_1: exp(β_0 + β_1 x) / β_1
    # etc.
    # Bracket via checking CDF at knots.
    F_q1 = exp(β0 + β1 * q[1]) / β1

    if u <= F_q1
        # In left tail: x = q_1 + log(u β_1 / exp(β_0 + β_1 q_1)) / β_1
        #             = q_1 + (log(u) + log(β_1) - β_0 - β_1 q_1) / β_1
        return q[1] + (log(u) - log(F_q1)) / β1
    end

    # Else numerical bisection in [q_1, x_cap]
    # Compute CDF via segmented integration
    cdf(x) = begin
        if x <= q[1]
            return exp(β0 + β1 * x) / β1
        end
        acc = F_q1
        if x <= q[2]
            acc += gl64(t -> smooth_dens(t, β0, β1, γ, q), q[1], x)
            return acc
        end
        acc += gl64(t -> smooth_dens(t, β0, β1, γ, q), q[1], q[2])
        if x <= q[3]
            acc += gl64(t -> smooth_dens(t, β0, β1, γ, q), q[2], x)
            return acc
        end
        acc += gl64(t -> smooth_dens(t, β0, β1, γ, q), q[2], q[3])
        acc += gl64(t -> smooth_dens(t, β0, β1, γ, q), q[3], x)
        return acc
    end

    sγ = γ[1] + γ[2] + γ[3]
    x_hi = isfinite(x_cap) ? x_cap : q[3] + (50.0 / (-sγ))^(1/3) + 2.0
    x_lo = q[1]

    # Make sure bracket contains u
    while cdf(x_hi) < u
        x_hi += 1.0
        x_hi > 1e4 && break
    end

    for _ in 1:100
        mid = (x_lo + x_hi) / 2
        F_mid = cdf(mid)
        if F_mid < u
            x_lo = mid
        else
            x_hi = mid
        end
        x_hi - x_lo < 1e-8 && break
    end
    (x_lo + x_hi) / 2
end

# ================================================================
#  FULL DGP: generate (y, η) from quantile-respecting smooth model
# ================================================================
"""
Generate panel data from the quantile-respecting smooth ABB model.

Arguments:
  N::Int        — number of individuals
  par::Params   — ABB parameters: a_Q (3×3), a_init, a_eps, b1_init, bL_init, b1_eps, bL_eps
  β1::Real      — shape parameter for smooth transition density
  tau           — quantile levels (length L=3)
  sigma_y::Real — standardization
  K::Int        — Hermite order (must match par.a_Q)
  seed::Int     — RNG seed

Returns (y::Matrix, η::Matrix).

η_{i,1} drawn from piecewise-uniform initial dist (ABB form)
η_{i,t} | η_{i,t-1} drawn from smooth transition (quantile-respecting)
ε_{i,t} drawn from piecewise-uniform ε dist (ABB form)
"""
function generate_data_smooth_qr(N::Int, par::Params, β1::Real, tau, sigma_y, K::Int;
                                  seed::Int=42)
    rng = MersenneTwister(seed)
    T = 3; L = length(tau)
    η = zeros(N, T); y = zeros(N, T)
    qbuf = zeros(L)

    # Initial η_1
    for i in 1:N
        η[i, 1] = pw_draw(rng, par.a_init, tau, par.b1_init, par.bL_init)
    end

    # Transitions
    for t in 2:T, i in 1:N
        transition_quantiles!(qbuf, η[i, t-1], par.a_Q, K, sigma_y)
        sort!(qbuf)  # ensure sorted (should be, but guard)
        β0, γ = solve_coeffs(qbuf, tau, β1)
        η[i, t] = smooth_draw(rng, β0, Float64(β1), γ, (qbuf[1], qbuf[2], qbuf[3]))
    end

    # Observations
    for t in 1:T, i in 1:N
        y[i, t] = η[i, t] + pw_draw(rng, par.a_eps, tau, par.b1_eps, par.bL_eps)
    end

    y, η
end

# ================================================================
#  TEST
# ================================================================
if abspath(PROGRAM_FILE) == @__FILE__
    println("="^70)
    println("  TEST: quantile-respecting smooth density")
    println("="^70)

    τ = [0.25, 0.50, 0.75]

    test_cases = [
        ("β_1 = 2.0, ABB knots",  [-0.337, 0.0, 0.337], 2.0),
        ("β_1 = 3.0, ABB knots",  [-0.337, 0.0, 0.337], 3.0),
        ("β_1 = 1.0, ABB knots",  [-0.337, 0.0, 0.337], 1.0),
        ("β_1 = 2.0, asymmetric", [-0.5, 0.1, 0.8],     2.0),
        ("β_1 = 2.0, wide",       [-1.0, 0.0, 1.0],     2.0),
    ]

    for (name, q, β1) in test_cases
        println("\n--- $name ---")
        try
            β0, γ = solve_coeffs(q, τ, β1; verbose=true)
            @printf("  Result: β_0=%.4f, γ=(%.4f, %.4f, %.4f)\n", β0, γ...)

            qt = (q[1], q[2], q[3])
            # Verify quantile conditions via actual CDF
            cdf_at(x) = begin
                if x <= q[1]; return exp(β0 + β1 * x) / β1; end
                F = exp(β0 + β1 * q[1]) / β1
                F += gl64(t -> smooth_dens(t, β0, β1, γ, qt), q[1], min(x, q[2]))
                x <= q[2] && return F
                F += gl64(t -> smooth_dens(t, β0, β1, γ, qt), q[2], min(x, q[3]))
                x <= q[3] && return F
                F += gl64(t -> smooth_dens(t, β0, β1, γ, qt), q[3], x)
                return F
            end
            F1 = cdf_at(q[1])
            F2 = cdf_at(q[2])
            F3 = cdf_at(q[3])
            sγ = γ[1] + γ[2] + γ[3]
            F_inf = cdf_at(q[3] + max((50.0 / -sγ)^(1/3), 2.0))
            @printf("  F(q_1)=%.6f (want %.4f)\n", F1, τ[1])
            @printf("  F(q_2)=%.6f (want %.4f)\n", F2, τ[2])
            @printf("  F(q_3)=%.6f (want %.4f)\n", F3, τ[3])
            @printf("  F(∞)  =%.6f (want 1.0)\n", F_inf)
        catch e
            println("  FAILED: $e")
        end
    end
end

#=
cspline_abb.jl — Cubic spline log-density with ABB quantile knots

The log-density of η_t given η_{t-1} is a NATURAL CUBIC SPLINE with knots
at the ABB quantile locations q₁(η_{t-1}) < q₂(η_{t-1}) < q₃(η_{t-1}).

Setup:
  Knots: t₁ = q₁, t₂ = q₂, t₃ = q₃  (from Hermite basis, same as ABB)
  Log-density at knots: s₁, s₂, s₃
  Left tail (x < t₁):  log f = s₁ + β_L(x - t₁)  (exponential, β_L > 0)
  Right tail (x > t₃): log f = s₃ + β_R(x - t₃)  (exponential, β_R < 0)
  Interior [t₁,t₂] and [t₂,t₃]: natural cubic spline matching (sₗ, s'ₗ)

Natural cubic spline: s''(t₁) = 0 and s''(t₃) = 0.
This gives C² continuity everywhere (including at knots).

Parameters:
  Transition: a_Q (K+1)×3 for knot locations + β_L, β_R coefficients
  The values s₁, s₂, s₃ are DETERMINED by the 4 constraints:
    F(t₁) = τ₁ = 0.25
    F(t₂) = τ₂ = 0.50
    F(t₃) = τ₃ = 0.75
    ∫f dx = 1

  So the free transition parameters are: a_Q (9) + β_L Hermite (K+1) + β_R Hermite (K+1) = 15
  Or if β_L and β_R are constants: 9 + 2 = 11 (same count as ABB!)

For now: start with β_L, β_R as constants (not η-dependent). Total = 11 + 5 (marginals) = 16.
=#

include("ABB_three_period.jl")
using Printf, LinearAlgebra

# ================================================================
#  CUBIC SPLINE ON 3 KNOTS WITH FREE ENDPOINT CURVATURES
#
#  Given knots t₁ < t₂ < t₃, values s₁, s₂, s₃,
#  and endpoint curvatures M₁ = s''(t₁), M₃ = s''(t₃).
#
#  M₂ = s''(t₂) is determined by C¹ continuity at t₂:
#    M₁h₁ + 2M₂(h₁+h₂) + M₃h₂ = 6[(s₃-s₂)/h₂ - (s₂-s₁)/h₁]
#
#  Tails are quadratic (Gaussian-like decay):
#    Left:  log f(x) = s₁ + β_L(x-t₁) + ½M₁(x-t₁)²   for x < t₁
#    Right: log f(x) = s₃ + β_R(x-t₃) + ½M₃(x-t₃)²   for x > t₃
#  where β_L = S'(t₁⁺), β_R = S'(t₃⁻).
#  Integrability requires M₁ < 0 and M₃ < 0.
#
#  Natural spline is the special case M₁ = M₃ = 0.
# ================================================================

# Compute M₂ from C¹ continuity at t₂
function cspline_M2(t::Vector{Float64}, s::Vector{Float64},
                    M1::Float64, M3::Float64)
    h1 = t[2] - t[1]; h2 = t[3] - t[2]
    (6.0*((s[3]-s[2])/h2 - (s[2]-s[1])/h1) - M1*h1 - M3*h2) / (2.0*(h1+h2))
end

# Compute β_L, β_R from the spline slopes at boundaries
function cspline_implied_beta(t::Vector{Float64}, s::Vector{Float64},
                              M1::Float64, M3::Float64)
    h1 = t[2] - t[1]; h2 = t[3] - t[2]
    M2 = cspline_M2(t, s, M1, M3)
    β_L = (s[2]-s[1])/h1 - h1*(2*M1+M2)/6   # S'(t₁⁺)
    β_R = (s[3]-s[2])/h2 + h2*(M2+2*M3)/6    # S'(t₃⁻)
    β_L, β_R
end

# Backward compatibility: natural spline (M1=M3=0)
function cspline_implied_beta(t::Vector{Float64}, s::Vector{Float64})
    cspline_implied_beta(t, s, 0.0, 0.0)
end

function cspline_eval(x::Float64, t::Vector{Float64}, s::Vector{Float64},
                      β_L::Float64, β_R::Float64, M1::Float64, M3::Float64)
    if x <= t[1]
        dx = x - t[1]
        return s[1] + β_L * dx + 0.5 * M1 * dx * dx
    end
    if x >= t[3]
        dx = x - t[3]
        return s[3] + β_R * dx + 0.5 * M3 * dx * dx
    end

    h1 = t[2] - t[1]; h2 = t[3] - t[2]
    M2 = cspline_M2(t, s, M1, M3)

    if x <= t[2]
        # Segment [t₁, t₂]: second derivatives M₁ and M₂
        a = t[2] - x; b = x - t[1]
        return M1*a^3/(6*h1) + M2*b^3/(6*h1) + (s[1]/h1 - M1*h1/6)*a + (s[2]/h1 - M2*h1/6)*b
    else
        # Segment [t₂, t₃]: second derivatives M₂ and M₃
        a = t[3] - x; b = x - t[2]
        return M2*a^3/(6*h2) + M3*b^3/(6*h2) + (s[2]/h2 - M2*h2/6)*a + (s[3]/h2 - M3*h2/6)*b
    end
end

# Backward compatible: linear tails (M1=M3=0, explicit β)
function cspline_eval(x::Float64, t::Vector{Float64}, s::Vector{Float64},
                      β_L::Float64, β_R::Float64)
    cspline_eval(x, t, s, β_L, β_R, 0.0, 0.0)
end

# Solve for (s₁, s₃, δ) from 3 mass constraints, s₂=0 pinned.
# κ_mean = (κ₁+κ₃)/2 is the model parameter (given).
# δ = (κ₃−κ₁)/2 is solved so both curvatures move simultaneously:
#   κ₁ = κ_mean − δ,  κ₃ = κ_mean + δ
# Integrability: κ₁ < 0 and κ₃ < 0  ⟺  |δ| < |κ_mean|.
# All other quantities (κ₂, β_L, β_R) determined analytically.
# Residuals: R_k = masses[k]/C − 0.25 for k=1,2,3.
struct SplineSolverBuffers
    masses::Vector{Float64}  # 4
    mp::Vector{Float64}      # 4
    mm::Vector{Float64}      # 4
    sp::Vector{Float64}      # 3
    sm::Vector{Float64}      # 3
end
SplineSolverBuffers() = SplineSolverBuffers(zeros(4), zeros(4), zeros(4), zeros(3), zeros(3))

# Backward compatible alias
const C1SolverBuffers = SplineSolverBuffers

# Convenience wrapper (allocates buffers)
function solve_cspline_c2!(s::Vector{Float64}, βL_out::Ref{Float64}, βR_out::Ref{Float64},
                            κ1_out::Ref{Float64}, κ3_out::Ref{Float64},
                            t::Vector{Float64}, τ::Vector{Float64}, κ_mean::Float64;
                            maxiter::Int=100, tol::Float64=1e-10)
    solve_cspline_c2!(s, βL_out, βR_out, κ1_out, κ3_out, t, τ, κ_mean, SplineSolverBuffers();
                       maxiter=maxiter, tol=tol)
end

# Main solver: 3×3 Newton for (s₁, s₃, δ) given κ_mean.
function solve_cspline_c2!(s::Vector{Float64}, βL_out::Ref{Float64}, βR_out::Ref{Float64},
                            κ1_out::Ref{Float64}, κ3_out::Ref{Float64},
                            t::Vector{Float64}, τ::Vector{Float64},
                            κ_mean::Float64, buf::SplineSolverBuffers;
                            maxiter::Int=100, tol::Float64=1e-10)
    s[1] = 0.0; s[2] = 0.0; s[3] = 0.0
    δ = 0.0  # initial guess: symmetric (κ₁=κ₃=κ_mean)

    masses = buf.masses; mp = buf.mp; mm = buf.mm
    s_tmp = buf.sp
    h_fd = 1e-7
    target = τ[2] - τ[1]  # = 0.25

    # Evaluate residuals given (s₁, s₃, δ) where κ₁=κ_mean−δ, κ₃=κ_mean+δ
    @inline function eval_residuals!(x_s1, x_s3, x_δ, m_buf)
        s_tmp[1] = x_s1; s_tmp[2] = 0.0; s_tmp[3] = x_s3
        κ1 = κ_mean - x_δ; κ3 = κ_mean + x_δ
        βL, βR = cspline_implied_beta(t, s_tmp, κ1, κ3)
        lr = max(s_tmp[1], s_tmp[2], s_tmp[3])
        cspline_masses!(m_buf, t, s_tmp, βL, βR, κ1, κ3, lr)
        C = m_buf[1]+m_buf[2]+m_buf[3]+m_buf[4]
        C < 1e-300 && return (Inf, Inf, Inf, C)
        (m_buf[1]/C - target, m_buf[2]/C - target, m_buf[3]/C - target, C)
    end

    abs_κ = abs(κ_mean)  # bound for |δ|

    for iter in 1:maxiter
        R1, R2, R3, C = eval_residuals!(s[1], s[3], δ, masses)
        isinf(R1) && break
        Rnorm = sqrt(R1*R1 + R2*R2 + R3*R3)
        Rnorm < tol && break

        # 3×3 Jacobian by central differences w.r.t. (s₁, s₃, δ)
        R1p, R2p, R3p, _ = eval_residuals!(s[1]+h_fd, s[3], δ, mp)
        R1m, R2m, R3m, _ = eval_residuals!(s[1]-h_fd, s[3], δ, mm)
        J11 = (R1p-R1m)/(2h_fd); J21 = (R2p-R2m)/(2h_fd); J31 = (R3p-R3m)/(2h_fd)

        R1p, R2p, R3p, _ = eval_residuals!(s[1], s[3]+h_fd, δ, mp)
        R1m, R2m, R3m, _ = eval_residuals!(s[1], s[3]-h_fd, δ, mm)
        J12 = (R1p-R1m)/(2h_fd); J22 = (R2p-R2m)/(2h_fd); J32 = (R3p-R3m)/(2h_fd)

        R1p, R2p, R3p, _ = eval_residuals!(s[1], s[3], δ+h_fd, mp)
        R1m, R2m, R3m, _ = eval_residuals!(s[1], s[3], δ-h_fd, mm)
        J13 = (R1p-R1m)/(2h_fd); J23 = (R2p-R2m)/(2h_fd); J33 = (R3p-R3m)/(2h_fd)

        # Solve 3×3 by cofactor (Cramer's rule)
        det = J11*(J22*J33-J23*J32) - J12*(J21*J33-J23*J31) + J13*(J21*J32-J22*J31)
        abs(det) < 1e-30 && break

        Δ1 = ((-R1)*(J22*J33-J23*J32) - J12*((-R2)*J33-J23*(-R3)) + J13*((-R2)*J32-J22*(-R3))) / det
        Δ2 = (J11*((-R2)*J33-J23*(-R3)) - (-R1)*(J21*J33-J23*J31) + J13*(J21*(-R3)-(-R2)*J31)) / det
        Δ3 = (J11*(J22*(-R3)-(-R2)*J32) - J12*(J21*(-R3)-(-R2)*J31) + (-R1)*(J21*J32-J22*J31)) / det

        (isfinite(Δ1) && isfinite(Δ2) && isfinite(Δ3)) || break

        # Line search with integrability guard: |δ_new| < |κ_mean|
        α = 1.0
        for _ in 1:20
            s1_new = s[1] + α*Δ1; s3_new = s[3] + α*Δ2; δ_new = δ + α*Δ3
            if abs(δ_new) < abs_κ  # ensures κ₁<0 and κ₃<0
                R1n, R2n, R3n, _ = eval_residuals!(s1_new, s3_new, δ_new, mp)
                if !isinf(R1n) && sqrt(R1n*R1n + R2n*R2n + R3n*R3n) < Rnorm
                    s[1] = s1_new; s[3] = s3_new; δ = δ_new
                    break
                end
            end
            α *= 0.5
        end
    end

    s[2] = 0.0
    κ1 = κ_mean - δ; κ3 = κ_mean + δ
    β_L, β_R = cspline_implied_beta(t, s, κ1, κ3)
    βL_out[] = β_L; βR_out[] = β_R
    κ1_out[] = κ1; κ3_out[] = κ3
    s
end

# Convenience: callers that don't need κ₁,κ₃ output (backward compatible signature)
function solve_cspline_c2!(s::Vector{Float64}, βL_out::Ref{Float64}, βR_out::Ref{Float64},
                            t::Vector{Float64}, τ::Vector{Float64},
                            κ_mean::Float64, buf::SplineSolverBuffers;
                            maxiter::Int=100, tol::Float64=1e-10)
    κ1_out = Ref(0.0); κ3_out = Ref(0.0)
    solve_cspline_c2!(s, βL_out, βR_out, κ1_out, κ3_out, t, τ, κ_mean, buf;
                       maxiter=maxiter, tol=tol)
    s
end

# Convenience: no buffer, no κ output
function solve_cspline_c2!(s::Vector{Float64}, βL_out::Ref{Float64}, βR_out::Ref{Float64},
                            t::Vector{Float64}, τ::Vector{Float64}, κ_mean::Float64;
                            maxiter::Int=100, tol::Float64=1e-10)
    solve_cspline_c2!(s, βL_out, βR_out, t, τ, κ_mean, SplineSolverBuffers();
                       maxiter=maxiter, tol=tol)
end

"""Density from cubic spline log-density (unnormalized)."""
cspline_dens(x, t, s, β_L, β_R) = exp(cspline_eval(x, t, s, β_L, β_R))

# ================================================================
#  ANALYTICAL DERIVATIVES OF SPLINE
#
#  Returns (val, ds1, ds3, dβL, dβR) where dsₖ = ∂val/∂sₖ, etc.
#  Note: s₂=0 is pinned, so derivatives are w.r.t. s₁ and s₃ only.
#  Derivatives w.r.t. t (knot positions) are also needed for a_Q.
# ================================================================

function cspline_eval_derivs(x::Float64, t::Vector{Float64}, s::Vector{Float64},
                              β_L::Float64, β_R::Float64)
    # Returns (val, ∂val/∂s₁, ∂val/∂s₃, ∂val/∂β_L, ∂val/∂β_R,
    #          ∂val/∂t₁, ∂val/∂t₂, ∂val/∂t₃)
    if x <= t[1]
        val = s[1] + β_L * (x - t[1])
        return (val, 1.0, 0.0, x - t[1], 0.0, -β_L, 0.0, 0.0)
    end
    if x >= t[3]
        val = s[3] + β_R * (x - t[3])
        return (val, 0.0, 1.0, 0.0, x - t[3], 0.0, 0.0, -β_R)
    end

    h1 = t[2] - t[1]; h2 = t[3] - t[2]
    H = h1 + h2

    # M2 = 3[(s₃-s₂)/h₂ - (s₂-s₁)/h₁] / H, with s₂=0:
    # M2 = 3[s₃/h₂ + s₁/h₁] / H
    M2 = 3.0 * (s[3]/h2 + s[1]/h1) / H
    # ∂M2/∂s₁ = 3/(h₁H), ∂M2/∂s₃ = 3/(h₂H)
    dM2_ds1 = 3.0 / (h1 * H)
    dM2_ds3 = 3.0 / (h2 * H)
    # ∂M2/∂β_L = ∂M2/∂β_R = 0 (M2 doesn't depend on β)
    # ∂M2/∂t depends on h1, h2 which depend on t

    if x <= t[2]
        # Segment [t₁, t₂]: S(x) = M2*b³/(6h₁) + (s₁/h₁)*a + (s₂/h₁ - M2*h₁/6)*b
        # with a = t₂-x, b = x-t₁, s₂=0
        a = t[2] - x; b = x - t[1]
        val = M2*b^3/(6*h1) + (s[1]/h1)*a + (0.0/h1 - M2*h1/6)*b

        # ∂val/∂s₁: through direct s₁ term and through M2
        # Direct: a/h₁
        # Through M2: ∂M2/∂s₁ × [b³/(6h₁) - h₁b/6]
        dval_dM2 = b^3/(6*h1) - h1*b/6
        dval_ds1 = a/h1 + dM2_ds1 * dval_dM2
        dval_ds3 = dM2_ds3 * dval_dM2
        dval_dβL = 0.0
        dval_dβR = 0.0

        # ∂val/∂t₁: a = t₂-x, b = x-t₁, h₁ = t₂-t₁
        # ∂a/∂t₁ = 0, ∂b/∂t₁ = -1, ∂h₁/∂t₁ = -1
        # ∂M2/∂t₁ = 3[s₃/h₂ × 0 + s₁/h₁ × (1/h₁)] / H - M2 × (-1)/H
        #          ... this gets complicated. Use numerical for t derivatives.
        # Actually, for a_Q derivatives, we need ∂val/∂t. Let me compute them.
        # val = M2*b³/(6h₁) + s₁*a/h₁ - M2*h₁*b/6
        # ∂val/∂t₁: (∂val/∂b)(∂b/∂t₁) + (∂val/∂a)(∂a/∂t₁) + (∂val/∂h₁)(∂h₁/∂t₁) + (∂val/∂M2)(∂M2/∂t₁)
        # ∂b/∂t₁ = -1, ∂a/∂t₁ = 0, ∂h₁/∂t₁ = -1
        # ∂val/∂b = M2*b²/(2h₁) - M2*h₁/6
        # ∂val/∂a = s₁/h₁
        # ∂val/∂h₁ = -M2*b³/(6h₁²) - s₁*a/h₁² - M2*b/6
        # ∂M2/∂t₁: M2 = 3(s₃/h₂ + s₁/h₁)/H, h₁=t₂-t₁, H=h₁+h₂
        #   ∂M2/∂t₁ = 3[s₁×(1/h₁²)]/H - M2/H × (-1) = 3s₁/(h₁²H) + M2/H
        dvdb = M2*b^2/(2*h1) - M2*h1/6
        dvda = s[1]/h1
        dvdh1 = -M2*b^3/(6*h1^2) - s[1]*a/h1^2 - M2*b/6
        dM2_dt1 = 3.0*s[1]/(h1^2*H) + M2/H
        dval_dt1 = dvdb*(-1) + dvda*0 + dvdh1*(-1) + dval_dM2*dM2_dt1

        # ∂val/∂t₂: ∂a/∂t₂ = 1, ∂b/∂t₂ = 0, ∂h₁/∂t₂ = 1
        # ∂M2/∂t₂: ∂h₁/∂t₂=1, ∂h₂/∂t₂=-1, ∂H/∂t₂=0
        #   M2 = 3(s₃/h₂ + s₁/h₁)/H
        #   ∂M2/∂t₂ = 3(s₃/h₂² - s₁/h₁²)/H
        dM2_dt2 = 3.0*(s[3]/h2^2 - s[1]/h1^2)/H
        dval_dt2 = dvdb*0 + dvda*1 + dvdh1*1 + dval_dM2*dM2_dt2

        # ∂val/∂t₃: ∂a/∂t₃ = 0, ∂b/∂t₃ = 0, ∂h₁/∂t₃ = 0
        # ∂M2/∂t₃: ∂h₂/∂t₃=1, ∂H/∂t₃=1
        #   ∂M2/∂t₃ = 3(-s₃/h₂²)/H - M2/H = -(3s₃/(h₂²H) + M2/H)
        dM2_dt3 = -(3.0*s[3]/(h2^2*H) + M2/H)
        dval_dt3 = dval_dM2*dM2_dt3

        return (val, dval_ds1, dval_ds3, dval_dβL, dval_dβR, dval_dt1, dval_dt2, dval_dt3)
    else
        # Segment [t₂, t₃]: S(x) = M2*a³/(6h₂) + (s₂/h₂ - M2*h₂/6)*a + s₃*b/h₂
        # with a = t₃-x, b = x-t₂, s₂=0
        a = t[3] - x; b = x - t[2]
        val = M2*a^3/(6*h2) + (-M2*h2/6)*a + s[3]*b/h2

        dval_dM2 = a^3/(6*h2) - h2*a/6
        dval_ds1 = dM2_ds1 * dval_dM2
        dval_ds3 = b/h2 + dM2_ds3 * dval_dM2
        dval_dβL = 0.0
        dval_dβR = 0.0

        # ∂val/∂t₁: only through M2
        dM2_dt1 = 3.0*s[1]/(h1^2*H) + M2/H
        dval_dt1 = dval_dM2*dM2_dt1

        # ∂val/∂t₂: ∂b/∂t₂=-1, ∂a/∂t₂=0, ∂h₂/∂t₂=-1
        dvdb = s[3]/h2
        dvda = M2*a^2/(2*h2) - M2*h2/6
        dvdh2 = -M2*a^3/(6*h2^2) - M2*a/6 - s[3]*b/h2^2
        dM2_dt2 = 3.0*(s[3]/h2^2 - s[1]/h1^2)/H
        dval_dt2 = dvdb*(-1) + dvda*0 + dvdh2*(-1) + dval_dM2*dM2_dt2

        # ∂val/∂t₃: ∂a/∂t₃=1, ∂b/∂t₃=0, ∂h₂/∂t₃=1
        dM2_dt3 = -(3.0*s[3]/(h2^2*H) + M2/H)
        dval_dt3 = dvdb*0 + dvda*1 + dvdh2*1 + dval_dM2*dM2_dt3

        return (val, dval_ds1, dval_ds3, dval_dβL, dval_dβR, dval_dt1, dval_dt2, dval_dt3)
    end
end

# ================================================================
#  C² ANALYTICAL GRADIENT INFRASTRUCTURE
#
#  Total derivative dS(x)/dθ for θ ∈ {t₁,t₂,t₃,κ_mean}.
#  Chain: θ → (s₁,s₃,δ) [via IFT] → (βL,βR,κ₁,κ₃) → S(x).
# ================================================================

# Partial derivatives of S w.r.t. (s₁, s₃, κ₁, κ₃, t₁, t₂, t₃)
# at evaluation point x, with βL, βR treated as given.
# Returns (val, ds1, ds3, dκ1, dκ3, dt1, dt2, dt3, dβL, dβR)
function cspline_eval_partials(x::Float64, t::Vector{Float64}, s::Vector{Float64},
                                βL::Float64, βR::Float64, κ1::Float64, κ3::Float64)
    if x <= t[1]
        dx = x - t[1]
        val = s[1] + βL*dx + 0.5*κ1*dx*dx
        # ∂/∂s₁=1, ∂/∂s₃=0, ∂/∂κ₁=½dx², ∂/∂κ₃=0
        # ∂/∂t₁=-βL-κ₁dx, ∂/∂t₂=0, ∂/∂t₃=0
        # ∂/∂βL=dx, ∂/∂βR=0
        return (val, 1.0, 0.0, 0.5*dx*dx, 0.0, -βL-κ1*dx, 0.0, 0.0, dx, 0.0)
    end
    if x >= t[3]
        dx = x - t[3]
        val = s[3] + βR*dx + 0.5*κ3*dx*dx
        return (val, 0.0, 1.0, 0.0, 0.5*dx*dx, 0.0, 0.0, -βR-κ3*dx, 0.0, dx)
    end

    h1 = t[2]-t[1]; h2 = t[3]-t[2]; H = h1+h2
    # M₂ = [6(s₃/h₂+s₁/h₁) - κ₁h₁ - κ₃h₂] / [2H]
    M2 = (6.0*(s[3]/h2 + s[1]/h1) - κ1*h1 - κ3*h2) / (2.0*H)

    # ∂M₂/∂s₁ = 3/(h₁H), ∂M₂/∂s₃ = 3/(h₂H)
    dM2_ds1 = 3.0/(h1*H); dM2_ds3 = 3.0/(h2*H)
    # ∂M₂/∂κ₁ = -h₁/(2H), ∂M₂/∂κ₃ = -h₂/(2H)
    dM2_dκ1 = -h1/(2.0*H); dM2_dκ3 = -h2/(2.0*H)

    if x <= t[2]
        a = t[2]-x; b = x-t[1]
        val = κ1*a^3/(6*h1) + M2*b^3/(6*h1) + (s[1]/h1-κ1*h1/6)*a + (s[2]/h1-M2*h1/6)*b

        dval_dM2 = b^3/(6*h1) - h1*b/6
        # Through s₁ direct + M₂
        dval_ds1 = a/h1 + dM2_ds1*dval_dM2
        dval_ds3 = dM2_ds3*dval_dM2
        # Through κ₁ direct (as M₁) + M₂
        dval_dκ1 = a^3/(6*h1) - h1*a/6 + dM2_dκ1*dval_dM2
        dval_dκ3 = dM2_dκ3*dval_dM2
        # βL, βR don't appear in interior
        dval_dβL = 0.0; dval_dβR = 0.0

        # ∂/∂t₁: ∂b/∂t₁=-1, ∂a/∂t₁=0, ∂h₁/∂t₁=-1
        dvdb = M2*b^2/(2*h1) - M2*h1/6
        dvda = κ1*a^2/(2*h1) - κ1*h1/6 + s[1]/h1
        dvdh1 = -κ1*a^3/(6*h1^2) - M2*b^3/(6*h1^2) - s[1]*a/h1^2 - κ1*a/6 - M2*b/6
        dM2_dt1 = 3.0*s[1]/(h1^2*H) + M2/H + κ1/(2*H)
        dval_dt1 = dvdb*(-1) + dvdh1*(-1) + dval_dM2*dM2_dt1

        # ∂/∂t₂: ∂a/∂t₂=1, ∂b/∂t₂=0, ∂h₁/∂t₂=1
        dM2_dt2 = 3.0*(s[3]/h2^2 - s[1]/h1^2)/H + (κ3-κ1)/(2*H)
        dval_dt2 = dvda*1 + dvdh1*1 + dval_dM2*dM2_dt2

        # ∂/∂t₃: ∂a/∂t₃=0, ∂b/∂t₃=0, ∂h₁/∂t₃=0
        dM2_dt3 = -(3.0*s[3]/(h2^2*H) + M2/H + κ3/(2*H))
        dval_dt3 = dval_dM2*dM2_dt3

        return (val, dval_ds1, dval_ds3, dval_dκ1, dval_dκ3, dval_dt1, dval_dt2, dval_dt3, dval_dβL, dval_dβR)
    else
        a = t[3]-x; b = x-t[2]
        val = M2*a^3/(6*h2) + κ3*b^3/(6*h2) + (s[2]/h2-M2*h2/6)*a + (s[3]/h2-κ3*h2/6)*b

        dval_dM2 = a^3/(6*h2) - h2*a/6
        dval_ds1 = dM2_ds1*dval_dM2
        dval_ds3 = b/h2 + dM2_ds3*dval_dM2
        dval_dκ1 = dM2_dκ1*dval_dM2
        dval_dκ3 = b^3/(6*h2) - h2*b/6 + dM2_dκ3*dval_dM2
        dval_dβL = 0.0; dval_dβR = 0.0

        # ∂/∂t₁
        dM2_dt1 = 3.0*s[1]/(h1^2*H) + M2/H + κ1/(2*H)
        dval_dt1 = dval_dM2*dM2_dt1

        # ∂/∂t₂: ∂b/∂t₂=-1, ∂a/∂t₂=0, ∂h₂/∂t₂=-1
        dvdb = κ3*b^2/(2*h2) - κ3*h2/6 + s[3]/h2
        dvda = M2*a^2/(2*h2) - M2*h2/6
        dvdh2 = -M2*a^3/(6*h2^2) - κ3*b^3/(6*h2^2) - M2*a/6 - κ3*b/6 - s[3]*b/h2^2
        dM2_dt2 = 3.0*(s[3]/h2^2 - s[1]/h1^2)/H + (κ3-κ1)/(2*H)
        dval_dt2 = dvdb*(-1) + dvdh2*(-1) + dval_dM2*dM2_dt2

        # ∂/∂t₃: ∂a/∂t₃=1, ∂b/∂t₃=0, ∂h₂/∂t₃=1
        dM2_dt3 = -(3.0*s[3]/(h2^2*H) + M2/H + κ3/(2*H))
        dval_dt3 = dvda*1 + dvdh2*1 + dval_dM2*dM2_dt3

        return (val, dval_ds1, dval_ds3, dval_dκ1, dval_dκ3, dval_dt1, dval_dt2, dval_dt3, dval_dβL, dval_dβR)
    end
end

# Derivatives of βL, βR w.r.t. (s₁, s₃, κ₁, κ₃, t₁, t₂, t₃)
# βL = (s₂-s₁)/h₁ - h₁(2κ₁+M₂)/6,  βR = (s₃-s₂)/h₂ + h₂(M₂+2κ₃)/6
# with s₂=0.
function cspline_beta_derivs(t::Vector{Float64}, s::Vector{Float64},
                              κ1::Float64, κ3::Float64)
    h1 = t[2]-t[1]; h2 = t[3]-t[2]; H = h1+h2
    M2 = (6.0*(s[3]/h2 + s[1]/h1) - κ1*h1 - κ3*h2) / (2.0*H)

    dM2_ds1 = 3.0/(h1*H); dM2_ds3 = 3.0/(h2*H)
    dM2_dκ1 = -h1/(2.0*H); dM2_dκ3 = -h2/(2.0*H)

    # βL = -s₁/h₁ - h₁(2κ₁+M₂)/6
    dβL_ds1 = -1.0/h1 - h1/6*dM2_ds1
    dβL_ds3 = -h1/6*dM2_ds3
    dβL_dκ1 = -h1/3 - h1/6*dM2_dκ1  # -2h₁/6 + ...
    dβL_dκ3 = -h1/6*dM2_dκ3

    # βR = s₃/h₂ + h₂(M₂+2κ₃)/6
    dβR_ds1 = h2/6*dM2_ds1
    dβR_ds3 = 1.0/h2 + h2/6*dM2_ds3
    dβR_dκ1 = h2/6*dM2_dκ1
    dβR_dκ3 = h2/3 + h2/6*dM2_dκ3  # 2h₂/6 + ...

    # t derivatives are complex; compute numerically via dM₂/dt
    dM2_dt1 = 3.0*s[1]/(h1^2*H) + M2/H + κ1/(2*H)
    dM2_dt2 = 3.0*(s[3]/h2^2 - s[1]/h1^2)/H + (κ3-κ1)/(2*H)
    dM2_dt3 = -(3.0*s[3]/(h2^2*H) + M2/H + κ3/(2*H))

    # βL = -s₁/h₁ - h₁(2κ₁+M₂)/6
    # ∂βL/∂t₁: ∂h₁/∂t₁=-1 → s₁/h₁²·(-1)·(-1) = -s₁/h₁²... wait
    # βL = -s₁/h₁ - h₁(2κ₁+M₂)/6
    # ∂βL/∂t₁ = s₁/h₁² + (2κ₁+M₂)/6 - h₁/6·dM₂/dt₁  (since ∂h₁/∂t₁=-1)
    dβL_dt1 = -s[1]/h1^2 + (2*κ1+M2)/6 - h1/6*dM2_dt1   # ∂h₁/∂t₁=-1
    dβL_dt2 = s[1]/h1^2 - (2*κ1+M2)/6 - h1/6*dM2_dt2    # ∂h₁/∂t₂=+1
    dβL_dt3 = -h1/6*dM2_dt3

    # βR = s₃/h₂ + h₂(M₂+2κ₃)/6
    # ∂βR/∂t₂: ∂h₂/∂t₂=-1
    dβR_dt1 = h2/6*dM2_dt1
    dβR_dt2 = s[3]/h2^2 - (M2+2*κ3)/6 + h2/6*dM2_dt2
    dβR_dt3 = -s[3]/h2^2 + (M2+2*κ3)/6 + h2/6*dM2_dt3  # ∂h₂/∂t₃=+1

    # Return as named tuple
    (dβL_ds1=dβL_ds1, dβL_ds3=dβL_ds3, dβL_dκ1=dβL_dκ1, dβL_dκ3=dβL_dκ3,
     dβL_dt1=dβL_dt1, dβL_dt2=dβL_dt2, dβL_dt3=dβL_dt3,
     dβR_ds1=dβR_ds1, dβR_ds3=dβR_ds3, dβR_dκ1=dβR_dκ1, dβR_dκ3=dβR_dκ3,
     dβR_dt1=dβR_dt1, dβR_dt2=dβR_dt2, dβR_dt3=dβR_dt3)
end

# Solver IFT: compute d(s₁,s₃,δ)/d(t₁,t₂,t₃,κ_mean).
# The solver satisfies F(s₁,s₃,δ; t,κ_mean) = 0 (3 mass residual equations).
# IFT: d(s₁,s₃,δ)/dθ = -J⁻¹ · ∂F/∂θ, where J = ∂F/∂(s₁,s₃,δ).
# J and ∂F/∂θ are computed by finite differences of mass residuals.
# Returns ds_dt[2,3], dδ_dt[3], ds_dκ[2], dδ_dκ as scalar.
# Analytical derivatives of shifted segment masses w.r.t. (s₁, s₃, κ₁, κ₃, t₁, t₂, t₃).
# dm[seg, param] where seg=1..4, param=1..7 (s₁,s₃,κ₁,κ₃,t₁,t₂,t₃).
# Interior segments: dm/dθ = half × Σ w_i exp(S-lr) × ∂S/∂θ  (+ boundary/width terms for t)
# Tails: analytical derivatives of _half_gaussian_integral.
function cspline_mass_derivs(t::Vector{Float64}, s::Vector{Float64},
                              βL::Float64, βR::Float64, κ1::Float64, κ3::Float64,
                              log_ref::Float64)
    dm = zeros(4, 7)  # dm[seg, param_idx]
    h1 = t[2]-t[1]; h2 = t[3]-t[2]

    # Beta derivatives (needed for tail mass derivatives)
    bd = cspline_beta_derivs(t, s, κ1, κ3)

    # ---- Left tail: m₁ = exp(s₁-lr) × I(βL, κ₁) ----
    I_L = _half_gaussian_integral(βL, κ1)
    e_s1 = exp(s[1] - log_ref)
    m1 = e_s1 * I_L
    # ∂I/∂β and ∂I/∂M for the half-Gaussian integral
    # I(β,M) = σ√(2π) exp(½β²σ²) Φ(-βσ) where σ=1/√(-M)
    # ∂I/∂β = σ√(2π) [βσ² exp(½β²σ²) Φ(-βσ) + exp(½β²σ²) × (-σ) φ(-βσ)]
    #       = I × βσ² - σ² exp(½β²σ²) × σ√(2π) × φ(-βσ)/√(2π) ... let me simplify
    # Actually: ∂I/∂β = ∫_{-∞}^0 u × exp(βu + ½Mu²) du = E[u] under the Gaussian kernel
    # By completing the square: mean = β/(-M) = βσ², so ∂I/∂β = (βσ²) × I + correction...
    # Simpler: use the identity ∂I/∂β = β/γ × I + 1/γ  where γ=-M
    # Actually: ∫u exp(βu-½γu²)du = (β/γ)I + 1/γ ... let me verify by differentiation.
    # d/dβ ∫exp(βu-½γu²)du = ∫u exp(βu-½γu²)du. Integration by parts or completing square:
    # ∫u exp(βu-½γu²)du from -∞ to 0 = [β/γ ∫exp(...) + 1/γ exp(βu-½γu²)]_{-∞}^{0}
    # = β/γ × I + 1/γ × 1 = (β×I + 1)/γ
    γ_L = -κ1
    # ∂I/∂β = ∫u exp(βu+½Mu²)du = (βI − 1)/γ  (by integration by parts)
    dI_dβ_L = (βL * I_L - 1.0) / γ_L
    # ∂I/∂γ = −½∫u² exp(βu−½γu²)du = −½[(γ+β²)I − β]/γ²
    dI_dγ_L = -0.5 * ((βL^2/γ_L^2 + 1.0/γ_L) * I_L - βL/γ_L^2)
    dI_dM_L = -dI_dγ_L  # since M = -γ

    # dm₁/dθ = ∂(e_s1 × I)/∂θ
    # dm₁/ds₁ = e_s1 × I + e_s1 × dI/dβL × dβL/ds₁ + e_s1 × dI/dM × dM/ds₁
    # But M here is κ₁ (endpoint curvature), not a function of s₁ directly.
    # Wait: m₁ = exp(s₁-lr) × I(βL, κ₁). βL depends on s₁,s₃,κ₁,κ₃,t.
    dm[1,1] = e_s1 * (I_L + dI_dβ_L * bd.dβL_ds1)                           # ds₁
    dm[1,2] = e_s1 * dI_dβ_L * bd.dβL_ds3                                     # ds₃
    dm[1,3] = e_s1 * (dI_dβ_L * bd.dβL_dκ1 + dI_dM_L)                        # dκ₁
    dm[1,4] = e_s1 * dI_dβ_L * bd.dβL_dκ3                                     # dκ₃
    dm[1,5] = e_s1 * dI_dβ_L * bd.dβL_dt1 + m1  # dt₁: boundary exp(S(t₁)-lr)=m1/I_L×I_L...
    # Actually boundary: ∂/∂t₁ ∫_{-∞}^{t₁} = +exp(S(t₁)-lr) + ∫ ∂integrand/∂t₁
    # exp(S(t₁)) = exp(s₁) (since S(t₁)=s₁). So boundary = exp(s₁-lr) = e_s1.
    dm[1,5] = e_s1 * dI_dβ_L * bd.dβL_dt1  # no boundary (u=x−t₁ substitution absorbs it)
    dm[1,6] = e_s1 * dI_dβ_L * bd.dβL_dt2                                     # dt₂
    dm[1,7] = e_s1 * dI_dβ_L * bd.dβL_dt3                                     # dt₃

    # ---- Right tail: m₄ = exp(s₃-lr) × I(-βR, κ₃) ----
    I_R = _half_gaussian_integral(-βR, κ3)
    e_s3 = exp(s[3] - log_ref)
    γ_R = -κ3
    # I_R = I(-βR, κ₃). ∂I/∂(-βR) = ((-βR)×I_R − 1)/γ_R
    dI_dβ_R_neg = (-βR * I_R - 1.0) / γ_R
    # ∂I_R/∂κ₃ = ∂I/∂M = −∂I/∂γ = +½[(γ+β²)I − β]/γ² with β=−βR
    dI_dM_R = 0.5 * ((βR^2/γ_R^2 + 1.0/γ_R) * I_R + βR/γ_R^2)
    # Chain: ∂m₄/∂βR = e_s3 × ∂I_R/∂(-βR) × (-1)
    dm[4,1] = e_s3 * (-dI_dβ_R_neg) * bd.dβR_ds1                              # ds₁
    dm[4,2] = e_s3 * (I_R + (-dI_dβ_R_neg) * bd.dβR_ds3)                      # ds₃
    dm[4,3] = e_s3 * (-dI_dβ_R_neg) * bd.dβR_dκ1                              # dκ₁
    dm[4,4] = e_s3 * ((-dI_dβ_R_neg) * bd.dβR_dκ3 + dI_dM_R)                  # dκ₃
    dm[4,5] = e_s3 * (-dI_dβ_R_neg) * bd.dβR_dt1                              # dt₁
    dm[4,6] = e_s3 * (-dI_dβ_R_neg) * bd.dβR_dt2                              # dt₂
    dm[4,7] = e_s3 * (-dI_dβ_R_neg) * bd.dβR_dt3  # no boundary (u=x−t₃ substitution absorbs it)

    # ---- Interior segments: GL quadrature ----
    @inbounds for seg in 1:2
        a = t[seg]; b = t[seg+1]
        mid = (a+b)*0.5; half = (b-a)*0.5
        for i in 1:16
            x = mid + half * GL16_NODES[i]
            _, ps1, ps3, pκ1, pκ3, pt1, pt2, pt3, _, _ = cspline_eval_partials(x, t, s, βL, βR, κ1, κ3)
            Sv = cspline_eval(x, t, s, βL, βR, κ1, κ3)
            w_exp = GL16_WEIGHTS[i] * exp(Sv - log_ref)
            # ∂m/∂(s₁,s₃,κ₁,κ₃) — no boundary or width change
            dm[seg+1, 1] += w_exp * ps1 * half
            dm[seg+1, 2] += w_exp * ps3 * half
            dm[seg+1, 3] += w_exp * pκ1 * half
            dm[seg+1, 4] += w_exp * pκ3 * half
            # ∂m/∂t_l — includes ∂S/∂t_l AND width/node shift effects
            # The full derivative of ∫_a^b f(x) dx w.r.t. t that changes a or b:
            # d/dt [half × Σ w_i f(mid+half×ξ_i)] where mid=(a+b)/2, half=(b-a)/2
            # For t_l changing a (seg start): ∂mid/∂a=1/2, ∂half/∂a=-1/2
            # For t_l changing b (seg end): ∂mid/∂b=1/2, ∂half/∂b=1/2
            # d/da = -1/2 Σ w_i f + half × Σ w_i f' × (1/2 - ξ_i/2)... complex
            # Simpler: use the Leibniz rule directly.
            # ∂/∂t_l ∫_{t_seg}^{t_{seg+1}} exp(S(x)-lr) dx
            #   = ∫ exp(S-lr) × ∂S/∂t_l dx + [boundary terms from limits]
            # For ∂S/∂t_l: already computed as pt1, pt2, pt3 (partial holding β,κ fixed)
            # BUT: β depends on t through cspline_implied_beta. The partials pt1,pt2,pt3
            # from cspline_eval_partials already include the M₂ dependence on t,
            # but NOT the β dependence. Interior segments don't use β directly
            # (β only appears in tails), so pt_l IS the correct ∂S/∂t_l for interior.
            dm[seg+1, 5] += w_exp * pt1 * half  # ∂S/∂t₁ contribution
            dm[seg+1, 6] += w_exp * pt2 * half
            dm[seg+1, 7] += w_exp * pt3 * half
        end
        # Boundary terms for t derivatives:
        # Segment [t_seg, t_{seg+1}]: ∂/∂t_seg adds -exp(S(t_seg)-lr), ∂/∂t_{seg+1} adds +exp(S(t_{seg+1})-lr)
        e_a = exp(cspline_eval(a, t, s, βL, βR, κ1, κ3) - log_ref)
        e_b = exp(cspline_eval(b, t, s, βL, βR, κ1, κ3) - log_ref)
        dm[seg+1, seg+4] += -e_a   # -exp(S(t_seg)) for lower limit (param t_{seg} = t[seg])
        dm[seg+1, seg+5] += +e_b   # +exp(S(t_{seg+1})) for upper limit
        # Width change: ∂half/∂t_l × Σ w_i f(x_i)... already accounted for?
        # No! The GL quadrature approximates ∫_a^b f dx = half × Σ w_i f(mid+half×ξ_i)
        # When t_l changes, mid and half change, AND the evaluation points x_i shift.
        # The Leibniz integral rule gives: d/dt_l ∫_a^b f(x,t_l) dx
        #   = ∫_a^b ∂f/∂t_l dx + f(b)×∂b/∂t_l - f(a)×∂a/∂t_l
        # The ∂f/∂t_l = exp(S-lr) × ∂S/∂t_l is what we computed above.
        # The boundary terms f(b)×∂b/∂t_l - f(a)×∂a/∂t_l are what we added.
        # BUT: the GL quadrature for ∫∂f/∂t_l dx uses nodes at FIXED positions
        # relative to the segment. This is correct because the partials ∂S/∂t_l
        # already account for how S changes when t_l moves (the x coordinate is
        # independent of t_l in the Leibniz rule).
    end

    dm
end

# Analytical IFT: compute d(s₁,s₃,δ)/d(t₁,t₂,t₃,κ_mean) using analytical mass derivatives.
function cspline_solver_ift(t::Vector{Float64}, s::Vector{Float64},
                             κ_mean::Float64, δ::Float64, buf::SplineSolverBuffers)
    κ1 = κ_mean - δ; κ3 = κ_mean + δ
    βL, βR = cspline_implied_beta(t, s, κ1, κ3)
    log_ref = max(s[1], s[2], s[3])
    # Recompute log_ref including GL nodes (same as in cspline_masses!)
    @inbounds for seg in 1:2
        a = t[seg]; b = t[seg+1]
        mid = (a+b)*0.5; half = (b-a)*0.5
        for i in 1:16
            x = mid + half * GL16_NODES[i]
            v = cspline_eval(x, t, s, βL, βR, κ1, κ3)
            v > log_ref && (log_ref = v)
        end
    end

    # Compute masses and their analytical derivatives
    cspline_masses!(buf.masses, t, s, βL, βR, κ1, κ3, log_ref)
    C = buf.masses[1]+buf.masses[2]+buf.masses[3]+buf.masses[4]
    C < 1e-300 && return (zeros(2,3), zeros(3), zeros(2), 0.0)

    dm = cspline_mass_derivs(t, s, βL, βR, κ1, κ3, log_ref)
    # dm[seg, param]: param order = s₁(1), s₃(2), κ₁(3), κ₃(4), t₁(5), t₂(6), t₃(7)

    # Derivatives of mass fractions F_k = m_k/C - 0.25
    # ∂F_k/∂θ = (∂m_k/∂θ × C - m_k × ∂C/∂θ) / C²
    # where ∂C/∂θ = Σ_j ∂m_j/∂θ
    dF = zeros(3, 7)  # dF[k, param] for k=1,2,3 (segments 1,2,3)
    for p in 1:7
        dC = dm[1,p] + dm[2,p] + dm[3,p] + dm[4,p]
        for k in 1:3
            dF[k, p] = (dm[k,p] * C - buf.masses[k] * dC) / (C*C)
        end
    end

    # Jacobian J = ∂F/∂(s₁, s₃, δ)
    # ∂F/∂s₁ = dF[:,1], ∂F/∂s₃ = dF[:,2]
    # ∂F/∂δ = ∂F/∂κ₁ × (-1) + ∂F/∂κ₃ × (+1) = dF[:,4] - dF[:,3]
    J11=dF[1,1]; J21=dF[2,1]; J31=dF[3,1]  # ∂F/∂s₁
    J12=dF[1,2]; J22=dF[2,2]; J32=dF[3,2]  # ∂F/∂s₃
    J13=dF[1,4]-dF[1,3]; J23=dF[2,4]-dF[2,3]; J33=dF[3,4]-dF[3,3]  # ∂F/∂δ

    det = J11*(J22*J33-J23*J32) - J12*(J21*J33-J23*J31) + J13*(J21*J32-J22*J31)
    abs(det) < 1e-30 && return (zeros(2,3), zeros(3), zeros(2), 0.0)

    # J⁻¹ by cofactor
    iC11=(J22*J33-J23*J32)/det; iC12=-(J21*J33-J23*J31)/det; iC13=(J21*J32-J22*J31)/det
    iC21=-(J12*J33-J13*J32)/det; iC22=(J11*J33-J13*J31)/det; iC23=-(J11*J32-J12*J31)/det
    iC31=(J12*J23-J13*J22)/det; iC32=-(J11*J23-J13*J21)/det; iC33=(J11*J22-J12*J21)/det

    # d(s₁,s₃,δ)/dθ = -J⁻¹ × ∂F/∂θ
    ds_dt = zeros(2, 3); dδ_dt = zeros(3)
    for l in 1:3
        p = l + 4  # param index for t_l
        ds_dt[1, l] = -(iC11*dF[1,p] + iC21*dF[2,p] + iC31*dF[3,p])
        ds_dt[2, l] = -(iC12*dF[1,p] + iC22*dF[2,p] + iC32*dF[3,p])
        dδ_dt[l]    = -(iC13*dF[1,p] + iC23*dF[2,p] + iC33*dF[3,p])
    end

    # ∂F/∂κ_mean = ∂F/∂κ₁ × 1 + ∂F/∂κ₃ × 1 = dF[:,3] + dF[:,4]
    dF_κ = (dF[1,3]+dF[1,4], dF[2,3]+dF[2,4], dF[3,3]+dF[3,4])
    ds_dκ = zeros(2)
    ds_dκ[1] = -(iC11*dF_κ[1] + iC21*dF_κ[2] + iC31*dF_κ[3])
    ds_dκ[2] = -(iC12*dF_κ[1] + iC22*dF_κ[2] + iC32*dF_κ[3])
    dδ_dκ    = -(iC13*dF_κ[1] + iC23*dF_κ[2] + iC33*dF_κ[3])

    (ds_dt, dδ_dt, ds_dκ, dδ_dκ)
end

# Total derivative of S(x) w.r.t. (t₁,t₂,t₃,κ_mean) at point x.
# Combines spline partials, beta derivatives, and solver IFT.
# Returns (dS_dt1, dS_dt2, dS_dt3, dS_dκ).
function cspline_total_score(x::Float64, t::Vector{Float64}, s::Vector{Float64},
                              βL::Float64, βR::Float64, κ1::Float64, κ3::Float64,
                              ds_dt::Matrix{Float64}, dδ_dt::Vector{Float64},
                              ds_dκ::Vector{Float64}, dδ_dκ::Float64)
    # Spline partials
    _, ps1, ps3, pκ1, pκ3, pt1, pt2, pt3, pβL, pβR = cspline_eval_partials(x, t, s, βL, βR, κ1, κ3)

    # Beta derivatives
    bd = cspline_beta_derivs(t, s, κ1, κ3)

    # "Effective" derivatives: fold βL,βR dependence into s₁,s₃,κ₁,κ₃,t
    D_s1 = ps1 + pβL*bd.dβL_ds1 + pβR*bd.dβR_ds1
    D_s3 = ps3 + pβL*bd.dβL_ds3 + pβR*bd.dβR_ds3
    D_κ1 = pκ1 + pβL*bd.dβL_dκ1 + pβR*bd.dβR_dκ1
    D_κ3 = pκ3 + pβL*bd.dβL_dκ3 + pβR*bd.dβR_dκ3

    # Total dS/dt_l = D_t_l + D_s1·ds₁/dt_l + D_s3·ds₃/dt_l + (D_κ3-D_κ1)·dδ/dt_l
    dS = zeros(4)  # (dt1, dt2, dt3, dκ_mean)
    for l in 1:3
        D_tl = (l==1 ? pt1 : l==2 ? pt2 : pt3) +
               pβL*(l==1 ? bd.dβL_dt1 : l==2 ? bd.dβL_dt2 : bd.dβL_dt3) +
               pβR*(l==1 ? bd.dβR_dt1 : l==2 ? bd.dβR_dt2 : bd.dβR_dt3)
        dS[l] = D_tl + D_s1*ds_dt[1,l] + D_s3*ds_dt[2,l] + (D_κ3-D_κ1)*dδ_dt[l]
    end
    # dS/dκ_mean: no direct effect (κ_mean doesn't appear in S directly)
    # but through κ₁=κ_mean-δ, κ₃=κ_mean+δ: dκ₁/dκ_mean=1-dδ/dκ, dκ₃/dκ_mean=1+dδ/dκ
    dS[4] = D_s1*ds_dκ[1] + D_s3*ds_dκ[2] + (D_κ1+D_κ3) + (D_κ3-D_κ1)*dδ_dκ

    dS
end

# ================================================================
#  SEGMENT MASSES (for normalization and quantile constraints)
# ================================================================

using Distributions: Normal, ccdf

# Gaussian half-integral: ∫_{-∞}^{0} exp(β u + ½M u²) du
# With M < 0 (γ = -M > 0): completing the square gives
#   exp(β²/(2γ)) × √(2π/γ) × Φ(−β/√γ)
# where Φ is the standard normal CDF.
# When M = 0: reduces to 1/β (exponential tail).
const _std_normal = Normal()

function _half_gaussian_integral(β::Float64, M::Float64)
    if abs(M) < 1e-12
        return β > 0 ? 1.0/β : Inf
    end
    M >= 0 && return Inf  # not integrable
    γ = -M
    σ = sqrt(1.0/γ)
    # ∫_{-∞}^{0} exp(β u - ½γ u²) du
    # = σ√(2π) × exp(½β²σ²) × Φ(-β σ)
    # Φ(-β σ) = ccdf(Normal(), β σ) for numerical stability
    return σ * sqrt(2π) * exp(0.5 * β^2 * σ^2) * ccdf(_std_normal, β * σ)
end
# ================================================================
#  LOG-SPACE INTEGRATION: ∫ exp(f(x)) dx
#
#  Given f(x) = log p(x) at grid points, fit a cubic spline to f,
#  then integrate exp(cubic) exactly on each segment using the
#  Taylor series recurrence.
#
#  This is much more accurate than Simpson for peaked densities:
#  - Simpson approximates p(x) by piecewise quadratic → O(h⁴)
#  - Log-space: approximates log p(x) by cubic → exact for Gaussian
# ================================================================

"""
    logspace_integrate(log_vals, grid, G)

Compute ∫exp(f(x))dx where f is a natural cubic spline interpolating
log_vals at grid points. Uses exact Taylor series for exp(cubic) on
each segment. Returns the integral value.

Natural cubic spline: f''(grid[1]) = f''(grid[G]) = 0.
"""
function logspace_integrate(log_vals::AbstractVector{Float64},
                            grid::AbstractVector{Float64}, G::Int)
    G < 2 && return 0.0

    # Fit natural cubic spline to log_vals: solve tridiagonal for M (second derivatives)
    # Natural: M[1] = M[G] = 0
    # Interior: h[i-1]M[i-1] + 2(h[i-1]+h[i])M[i] + h[i]M[i+1] = 6(Δ[i]/h[i] - Δ[i-1]/h[i-1])
    # where h[i] = grid[i+1]-grid[i], Δ[i] = log_vals[i+1]-log_vals[i]

    n = G - 2  # number of interior points
    if n == 0
        # Only 2 points: linear interpolation → exp(linear)
        h = grid[2] - grid[1]
        a = log_vals[1]; b = (log_vals[2] - log_vals[1]) / h
        return _exp_cubic_integral(b, 0.0, 0.0, h) * exp(a)
    end

    # Uniform grid: h[i] = h for all i
    h = grid[2] - grid[1]

    # For uniform grid, the tridiagonal system simplifies:
    # h·M[i-1] + 4h·M[i] + h·M[i+1] = 6/h·(f[i+1] - 2f[i] + f[i-1])
    # Divide by h: M[i-1] + 4M[i] + M[i+1] = 6(f[i+1]-2f[i]+f[i-1])/h²
    # With M[0] = M[G-1] = 0 (natural, 0-indexed in math, 1-indexed in code: M[1]=M[G]=0)

    # Clamped cubic spline: specify f' at endpoints via finite differences
    # f'(1) ≈ (-3f₁+4f₂-f₃)/(2h), f'(G) ≈ (f_{G-2}-4f_{G-1}+3f_G)/(2h)
    fp_1 = (-3.0*log_vals[1] + 4.0*log_vals[2] - log_vals[3]) / (2.0*h)
    fp_G = (log_vals[G-2] - 4.0*log_vals[G-1] + 3.0*log_vals[G]) / (2.0*h)

    # Full tridiagonal for G points (clamped BC):
    # Row 1: 2h·M₁ + h·M₂ = 6[(f₂-f₁)/h - fp_1]/h
    # Row i (interior): h·M_{i-1}+4h·M_i+h·M_{i+1} = 6(f_{i+1}-2f_i+f_{i-1})/h
    # Row G: h·M_{G-1}+2h·M_G = 6[fp_G - (f_G-f_{G-1})/h]/h
    M = zeros(G)
    d = zeros(G); rhs = zeros(G)

    # Setup
    d[1] = 2.0; rhs[1] = 6.0*((log_vals[2]-log_vals[1])/h - fp_1) / h
    @inbounds for i in 2:G-1
        d[i] = 4.0
        rhs[i] = 6.0*(log_vals[i+1] - 2.0*log_vals[i] + log_vals[i-1]) / (h*h)
    end
    d[G] = 2.0; rhs[G] = 6.0*(fp_G - (log_vals[G]-log_vals[G-1])/h) / h

    # Thomas algorithm for [d₁ 1; 1 d₂ 1; ...; 1 d_G]
    @inbounds for i in 2:G
        w = 1.0 / d[i-1]
        d[i] -= w
        rhs[i] -= w * rhs[i-1]
    end
    M[G] = rhs[G] / d[G]
    @inbounds for i in G-1:-1:1
        M[i] = (rhs[i] - M[i+1]) / d[i]
    end

    # Integrate exp(cubic) on each segment [grid[i], grid[i+1]]
    # On segment i: f(x) = M[i](grid[i+1]-x)³/(6h) + M[i+1](x-grid[i])³/(6h)
    #              + (f[i]/h - M[i]h/6)(grid[i+1]-x) + (f[i+1]/h - M[i+1]h/6)(x-grid[i])
    # With local var t = x - grid[i], a = h - t:
    # f(t) = M[i](h-t)³/(6h) + M[i+1]t³/(6h) + (f[i]/h-M[i]h/6)(h-t) + (f[i+1]/h-M[i+1]h/6)t
    # f(t) = f[i] + c₁t + c₂t² + c₃t³  where:
    #   c₁ = (f[i+1]-f[i])/h - h(2M[i]+M[i+1])/6
    #   c₂ = M[i]/2
    #   c₃ = (M[i+1]-M[i])/(6h)

    total = 0.0
    @inbounds for i in 1:G-1
        c1 = (log_vals[i+1] - log_vals[i]) / h - h * (2.0*M[i] + M[i+1]) / 6.0
        c2 = M[i] / 2.0
        c3 = (M[i+1] - M[i]) / (6.0 * h)
        total += exp(log_vals[i]) * _exp_cubic_integral(c1, c2, c3, h)
    end
    total
end

# Precompute GL nodes once (kept for backward compatibility)
const _GL16_β = [i / sqrt(4i^2 - 1) for i in 1:15]
const _GL16_J = SymTridiagonal(zeros(16), _GL16_β)
const _GL16_EIG = eigen(_GL16_J)
const GL16_NODES = _GL16_EIG.values
const GL16_WEIGHTS = 2.0 .* _GL16_EIG.vectors[1,:].^2

# ================================================================
#  EXACT INTEGRATION VIA TAYLOR SERIES (replaces GL quadrature)
#
#  ∫₀ᴸ exp(c₁t + c₂t² + c₃t³) dt = Σ aₙ Lⁿ⁺¹/(n+1)
#  where aₙ satisfies the recurrence:
#    n·aₙ = c₁·aₙ₋₁ + 2c₂·aₙ₋₂ + 3c₃·aₙ₋₃,  a₀=1
#
#  This is the power series of exp(cubic), integrated term-by-term.
#  Equivalent to evaluating the incomplete Airy integral exactly.
#  Converges for all finite L (entire function).
# ================================================================

"""
    _exp_cubic_integral(c1, c2, c3, L; maxterms=80, tol=1e-15)

Compute ∫₀ᴸ exp(c₁t + c₂t² + c₃t³) dt exactly via convergent Taylor series.
Returns the integral value. The series converges for all finite L.
"""
function _exp_cubic_integral(c1::Float64, c2::Float64, c3::Float64, L::Float64;
                              maxterms::Int=80, tol::Float64=1e-15)
    # Recurrence: n·aₙ = c₁·aₙ₋₁ + 2c₂·aₙ₋₂ + 3c₃·aₙ₋₃
    a = zeros(maxterms + 1)  # a[n+1] stores aₙ (1-indexed)
    a[1] = 1.0  # a₀ = 1

    result = L  # first term: a₀ × L¹/1
    Ln = L      # Lⁿ⁺¹
    for n in 1:maxterms
        val = 0.0
        n >= 1 && (val += c1 * a[n])      # c₁·aₙ₋₁
        n >= 2 && (val += 2c2 * a[n-1])   # 2c₂·aₙ₋₂
        n >= 3 && (val += 3c3 * a[n-2])   # 3c₃·aₙ₋₃
        a[n+1] = val / n
        Ln *= L
        term = a[n+1] * Ln / (n + 1)
        result += term
        # Require at least 6 terms before checking convergence (avoid early exit when a₁=0)
        n >= 6 && abs(term) < tol * abs(result) && break
    end
    result
end

"""
Compute segment masses using exact Taylor series for interior
and analytical Gaussian for tails. No GL quadrature needed.
"""
function cspline_masses!(masses::Vector{Float64}, t::Vector{Float64},
                         s::Vector{Float64}, β_L::Float64, β_R::Float64,
                         M1::Float64, M3::Float64, log_ref_in::Float64)
    if (β_L <= 0 && M1 >= 0) || (β_R >= 0 && M3 >= 0)
        @inbounds masses[1]=Inf; masses[2]=Inf; masses[3]=Inf; masses[4]=Inf
        return masses
    end

    # log_ref: use max of s values (sufficient for shifted masses)
    log_ref = max(s[1], s[2], s[3], log_ref_in)

    # Left tail: exp(s₁ - log_ref) × ∫_{-∞}^{0} exp(β_L u + ½M₁ u²) du
    @inbounds masses[1] = exp(s[1] - log_ref) * _half_gaussian_integral(β_L, M1)

    # Right tail: exp(s₃ - log_ref) × ∫_{-∞}^{0} exp(-β_R v + ½M₃ v²) dv
    @inbounds masses[4] = exp(s[3] - log_ref) * _half_gaussian_integral(-β_R, M3)

    # Interior segments: exact Taylor series
    h1 = t[2] - t[1]; h2 = t[3] - t[2]; H = h1 + h2
    M2 = (6.0*(s[3]/h2 + s[1]/h1) - M1*h1 - M3*h2) / (2.0*H)

    # Segment [t₁, t₂]: S(t₁+b) = s₁ + c₁b + c₂b² + c₃b³, b ∈ [0, h₁]
    # c₁ = S'(t₁⁺) = βL, c₂ = M₁/2, c₃ = (M₂-M₁)/(6h₁)
    c1_1 = β_L;  c2_1 = M1 / 2.0;  c3_1 = (M2 - M1) / (6.0 * h1)
    @inbounds masses[2] = exp(s[1] - log_ref) * _exp_cubic_integral(c1_1, c2_1, c3_1, h1)

    # Segment [t₂, t₃]: S(t₂+b) = s₂ + c₁b + c₂b² + c₃b³, b ∈ [0, h₂]
    # c₁ = S'(t₂⁺) from right, c₂ = M₂/2, c₃ = (M₃-M₂)/(6h₂)
    # S'(t₂⁺) = (s₃-s₂)/h₂ - h₂(M₂+2M₃)/6 ... no, that's S'(t₃⁻).
    # S'(t₂) from the right segment: using a=t₃-t₂-b, b=x-t₂:
    #   S'(t₂⁺) = -M₂(t₃-t₂)/(2) + s₃/h₂ ... need to compute from spline formula
    # Actually: S'(t₂) = (s₂-s₁)/h₁ + h₁(M₁+2M₂)/6  ... no, this is S'(t₂⁻)
    # From C¹ continuity, S'(t₂⁻) = S'(t₂⁺), so either formula works.
    # From the left segment: S'(t₂) = (s₂-s₁)/h₁ + h₁(M₁+2M₂)/6
    # Wait, S'(t₂) from the [t₁,t₂] segment:
    # S(x) = M₁a³/(6h₁) + M₂b³/(6h₁) + (s₁/h₁-M₁h₁/6)a + (s₂/h₁-M₂h₁/6)b
    # S'(x) = -M₁a²/(2h₁) + M₂b²/(2h₁) - s₁/h₁+M₁h₁/6 + s₂/h₁-M₂h₁/6
    # At x=t₂: a=0, b=h₁: S'(t₂) = M₂h₁/2 - s₁/h₁+M₁h₁/6 + s₂/h₁-M₂h₁/6
    #   = (s₂-s₁)/h₁ + M₁h₁/6 + M₂h₁/3
    slope_t2 = (s[2]-s[1])/h1 + M1*h1/6 + M2*h1/3
    c1_2 = slope_t2;  c2_2 = M2 / 2.0;  c3_2 = (M3 - M2) / (6.0 * h2)
    @inbounds masses[3] = exp(s[2] - log_ref) * _exp_cubic_integral(c1_2, c2_2, c3_2, h2)

    masses
end

# Convenience: linear tails (M1=M3=0)
function cspline_masses!(masses::Vector{Float64}, t::Vector{Float64},
                         s::Vector{Float64}, β_L::Float64, β_R::Float64,
                         log_ref_in::Float64)
    cspline_masses!(masses, t, s, β_L, β_R, 0.0, 0.0, log_ref_in)
end

function cspline_masses!(masses::Vector{Float64}, t::Vector{Float64},
                         s::Vector{Float64}, β_L::Float64, β_R::Float64)
    cspline_masses!(masses, t, s, β_L, β_R, 0.0, 0.0, 0.0)
end

function cspline_masses(t::Vector{Float64}, s::Vector{Float64},
                        β_L::Float64, β_R::Float64)
    masses = zeros(4)
    cspline_masses!(masses, t, s, β_L, β_R, 0.0, 0.0, 0.0)
end

# ================================================================
#  SOLVE FOR s₁, s₂, s₃ GIVEN (t, β_L, β_R, τ)
#
#  4 constraints, 3 unknowns (s₁, s₂, s₃) — overdetermined.
#  But ∫f=1 is one constraint, and F(tₗ)=τₗ gives 3 more.
#  Actually only 3 independent: F(t₁)=τ₁, F(t₂)=τ₂ imply the
#  segment mass ratio M₀/C = τ₁ and (M₀+M₁)/C = τ₂.
#  With C = M₀+M₁+M₂+M₃ and 4 segments, we have:
#    M₀/C = τ₁ = 0.25
#    (M₀+M₁)/C = τ₂ = 0.50
#    (M₀+M₁+M₂)/C = τ₃ = 0.75
#  These give M₁ = M₀, M₂ = M₀, M₃ = M₀ (equal segment masses!
#  because τ gaps are all 0.25). So M₀ = M₁ = M₂ = M₃ = C/4.
#
#  This means we need:
#    exp(s₁)/β_L = M₁(s)  (left tail mass = interior segment 1 mass)
#    M₁(s) = M₂(s)        (two interior segments equal)
#    exp(s₃)/(-β_R) = M₂(s) (right tail mass = interior segment 2 mass)
#
#  3 equations, 3 unknowns (s₁, s₂, s₃). Newton's method.
# ================================================================

"""
Solve for s = [s₁,s₂,s₃] given knots t, tail slopes β_L/β_R, and quantiles τ.
Pin s₂ = 0 to break scale invariance, solve 2×2 system for (s₁, s₃).
"""
function solve_cspline_values!(s::Vector{Float64}, t::Vector{Float64},
                               β_L::Float64, β_R::Float64, τ::Vector{Float64};
                               maxiter::Int=100, tol::Float64=1e-10)
    # Pin s₂ = 0 to break scale invariance (adding constant c to all s_k
    # doesn't change quantile ratios cumM/C). Solve 2×2 system for (s₁, s₃).
    s[1] = 0.0; s[2] = 0.0; s[3] = 0.0

    # Pre-allocate working arrays
    masses = zeros(4); mp = zeros(4); mm = zeros(4); m_new = zeros(4)
    R = zeros(2); R_new = zeros(2)
    J = zeros(2, 2); Δ = zeros(2)
    sp = zeros(3); sm = zeros(3); s_new = zeros(3)

    h = 1e-7

    for iter in 1:maxiter
        log_ref = max(s[1], s[2], s[3])
        cspline_masses!(masses, t, s, β_L, β_R, log_ref)
        any(isinf, masses) && break
        C = masses[1] + masses[2] + masses[3] + masses[4]
        C < 1e-300 && break
        cumM1 = masses[1]; cumM2 = cumM1+masses[2]

        # 2 independent residuals (s₂=0 fixed, so only 2 unknowns s₁, s₃)
        @inbounds R[1] = cumM1/C - τ[1]
        @inbounds R[2] = cumM2/C - τ[2]
        Rnorm = sqrt(R[1]^2 + R[2]^2)
        Rnorm < tol && break

        # 2×2 Jacobian: derivatives w.r.t. s₁ and s₃ (s₂=0 fixed)
        @inbounds for (jcol, jvar) in enumerate((1, 3))
            sp .= s; sp[jvar] += h
            sm .= s; sm[jvar] -= h
            log_ref_p = max(sp[1], sp[2], sp[3])
            log_ref_m = max(sm[1], sm[2], sm[3])
            cspline_masses!(mp, t, sp, β_L, β_R, log_ref_p)
            cspline_masses!(mm, t, sm, β_L, β_R, log_ref_m)
            Cp = mp[1]+mp[2]+mp[3]+mp[4]
            Cm = mm[1]+mm[2]+mm[3]+mm[4]
            cmp1=mp[1]; cmp2=cmp1+mp[2]
            cmm1=mm[1]; cmm2=cmm1+mm[2]
            J[1,jcol] = (cmp1/Cp - cmm1/Cm) / (2h)
            J[2,jcol] = (cmp2/Cp - cmm2/Cm) / (2h)
        end

        # 2×2 Cramer's rule
        detJ = J[1,1]*J[2,2] - J[1,2]*J[2,1]
        abs(detJ) < 1e-30 && break
        Δ[1] = (-R[1]*J[2,2] + R[2]*J[1,2]) / detJ
        Δ[2] = (-R[2]*J[1,1] + R[1]*J[2,1]) / detJ

        α = 1.0
        for _ in 1:20
            s_new[1] = s[1] + α*Δ[1]; s_new[2] = 0.0; s_new[3] = s[3] + α*Δ[2]
            log_ref_new = max(s_new[1], s_new[2], s_new[3])
            cspline_masses!(m_new, t, s_new, β_L, β_R, log_ref_new)
            if !any(isinf, m_new)
                C_new = m_new[1]+m_new[2]+m_new[3]+m_new[4]
                if C_new > 1e-300
                    cm1=m_new[1]; cm2=cm1+m_new[2]
                    R_new[1] = cm1/C_new - τ[1]
                    R_new[2] = cm2/C_new - τ[2]
                    Rnew_norm = sqrt(R_new[1]^2 + R_new[2]^2)
                    if Rnew_norm < Rnorm
                        s .= s_new
                        break
                    end
                end
            end
            α *= 0.5
        end
    end
    s
end

# Compute ∂(s₁,s₃)/∂(t₁,t₂,t₃,β_L,β_R) via implicit function theorem on the 2×2
# Newton system R(s₁,s₃;t,β)=0 with s₂=0 pinned.
# Returns ds_dt (2×3) and ds_dβ (2×2) where rows=(s₁,s₃), cols=(t₁,t₂,t₃) and (β_L,β_R).
function solve_cspline_sensitivities(t::Vector{Float64}, s::Vector{Float64},
                                      β_L::Float64, β_R::Float64, τ::Vector{Float64})
    h = 1e-7
    mp = zeros(4); mm = zeros(4)

    # 2×2 Jacobian ∂R/∂(s₁,s₃)
    J_s = zeros(2, 2)
    sp = copy(s); sm = copy(s)
    for (jcol, jvar) in enumerate((1, 3))
        sp .= s; sp[jvar] += h; sm .= s; sm[jvar] -= h
        lr_p = max(sp[1],sp[2],sp[3]); lr_m = max(sm[1],sm[2],sm[3])
        cspline_masses!(mp, t, sp, β_L, β_R, lr_p)
        cspline_masses!(mm, t, sm, β_L, β_R, lr_m)
        Cp = sum(mp); Cm = sum(mm)
        J_s[1,jcol] = (mp[1]/Cp - mm[1]/Cm) / (2h)
        J_s[2,jcol] = ((mp[1]+mp[2])/Cp - (mm[1]+mm[2])/Cm) / (2h)
    end

    det_Js = J_s[1,1]*J_s[2,2] - J_s[1,2]*J_s[2,1]
    if abs(det_Js) < 1e-30
        return zeros(2,3), zeros(2,2)
    end
    inv_Js = [J_s[2,2] -J_s[1,2]; -J_s[2,1] J_s[1,1]] ./ det_Js

    # ∂R/∂t: 2×3
    J_t = zeros(2, 3)
    tp_v = copy(t); tm_v = copy(t)
    lr = max(s[1],s[2],s[3])
    for j in 1:3
        tp_v .= t; tp_v[j] += h; tm_v .= t; tm_v[j] -= h
        cspline_masses!(mp, tp_v, s, β_L, β_R, lr)
        cspline_masses!(mm, tm_v, s, β_L, β_R, lr)
        Cp = sum(mp); Cm = sum(mm)
        J_t[1,j] = (mp[1]/Cp - mm[1]/Cm) / (2h)
        J_t[2,j] = ((mp[1]+mp[2])/Cp - (mm[1]+mm[2])/Cm) / (2h)
    end

    # ∂R/∂β: 2×2
    J_β = zeros(2, 2)
    cspline_masses!(mp, t, s, β_L+h, β_R, lr)
    cspline_masses!(mm, t, s, β_L-h, β_R, lr)
    Cp = sum(mp); Cm = sum(mm)
    J_β[1,1] = (mp[1]/Cp - mm[1]/Cm) / (2h)
    J_β[2,1] = ((mp[1]+mp[2])/Cp - (mm[1]+mm[2])/Cm) / (2h)
    cspline_masses!(mp, t, s, β_L, β_R+h, lr)
    cspline_masses!(mm, t, s, β_L, β_R-h, lr)
    Cp = sum(mp); Cm = sum(mm)
    J_β[1,2] = (mp[1]/Cp - mm[1]/Cm) / (2h)
    J_β[2,2] = ((mp[1]+mp[2])/Cp - (mm[1]+mm[2])/Cm) / (2h)

    ds_dt = -inv_Js * J_t   # 2×3
    ds_dβ = -inv_Js * J_β   # 2×2
    ds_dt, ds_dβ
end

# Convenience wrapper
function solve_cspline_values(t::Vector{Float64}, β_L::Float64, β_R::Float64,
                              τ::Vector{Float64}; maxiter::Int=100, tol::Float64=1e-10)
    s = zeros(3)
    solve_cspline_values!(s, t, β_L, β_R, τ; maxiter=maxiter, tol=tol)
end

# ================================================================
#  CONDITIONAL LOG-DENSITY
# ================================================================

"""
Evaluate normalized log f(x | η_{t-1}) using cubic spline.
Solves for s values at each call (could be cached).
"""
function cspline_logdens(x::Float64, η_lag::Float64, a_Q::Matrix{Float64},
                         β_L::Float64, β_R::Float64,
                         K::Int, σy::Float64, τ::Vector{Float64})
    # Compute knot locations
    z = η_lag / σy
    hv = zeros(K+1); hv[1]=1.0; K>=1 && (hv[2]=z)
    for k in 2:K; hv[k+1] = z*hv[k] - (k-1)*hv[k-1]; end
    t = [dot(view(a_Q,:,l), hv) for l in 1:3]

    # Check ordering
    (t[2] <= t[1] || t[3] <= t[2]) && return -1e10

    # Solve for s values
    s = solve_cspline_values(t, β_L, β_R, τ)

    # Evaluate and normalize
    masses = cspline_masses(t, s, β_L, β_R)
    C = sum(masses)
    C < 1e-300 && return -1e10

    cspline_eval(x, t, s, β_L, β_R) - log(C)
end

# ================================================================
#  TRANSITION MATRIX FOR FORWARD FILTER
# ================================================================

# C² version: β determined by spline slopes, quadratic tails.
# κ_mean is the model parameter; solver finds κ₁,κ₃ via δ=(κ₃−κ₁)/2.
function cspline_transition_matrix!(T_mat::Matrix{Float64},
                                    grid::Vector{Float64}, G::Int,
                                    a_Q::Matrix{Float64}, κ_mean_Q::Float64,
                                    K::Int, σy::Float64, τ::Vector{Float64},
                                    hv::Vector{Float64}, t::Vector{Float64},
                                    s::Vector{Float64}, masses::Vector{Float64},
                                    c1buf::C1SolverBuffers)
    βL_ref = Ref(0.0); βR_ref = Ref(0.0)
    κ1_ref = Ref(0.0); κ3_ref = Ref(0.0)

    @inbounds for g1 in 1:G
        z = grid[g1] / σy
        hv[1]=1.0; K>=1 && (hv[2]=z)
        for k in 2:K; hv[k+1] = z*hv[k] - (k-1)*hv[k-1]; end
        for l in 1:3; t[l] = dot(view(a_Q,:,l), hv); end

        if t[2] <= t[1] || t[3] <= t[2]
            for g2 in 1:G; T_mat[g1,g2] = 1e-300; end
            continue
        end

        solve_cspline_c2!(s, βL_ref, βR_ref, κ1_ref, κ3_ref, t, τ, κ_mean_Q, c1buf)
        β_L = βL_ref[]; β_R = βR_ref[]
        κ1 = κ1_ref[]; κ3 = κ3_ref[]

        log_ref = s[1]
        @inbounds for g2 in 1:G
            v = cspline_eval(grid[g2], t, s, β_L, β_R, κ1, κ3)
            v > log_ref && (log_ref = v)
        end

        cspline_masses!(masses, t, s, β_L, β_R, κ1, κ3, log_ref)
        C_shifted = masses[1]+masses[2]+masses[3]+masses[4]
        if C_shifted < 1e-300
            for g2 in 1:G; T_mat[g1,g2] = 1e-300; end
            continue
        end

        @inbounds for g2 in 1:G
            T_mat[g1,g2] = exp(cspline_eval(grid[g2], t, s, β_L, β_R, κ1, κ3) - log_ref) / C_shifted
        end
    end
end

# ================================================================
#  FORWARD FILTER LIKELIHOOD
# ================================================================

# Neg avg log-likelihood via forward filter with Simpson's rule.
# C¹ cubic spline: β determined by spline slopes at knots.
# Parameters: v = [vec(a_Q)(9), a_init(3), a_eps(2)] = 14 total for K=2.
# Pre-allocated workspace for repeated likelihood evaluations.
# Grid is adaptive: base grid [grid_min, grid_max] with G_base points,
# extended by up to G_ext points on each side for tail coverage.
struct CSplineWorkspace
    # Full grid (base + extensions): max size G_base + 2*G_ext
    grid::Vector{Float64}
    sw::Vector{Float64}
    T_mat::Matrix{Float64}
    f_init::Vector{Float64}
    p::Vector{Float64}
    p_new::Vector{Float64}
    pw::Vector{Float64}
    a_init_s::Vector{Float64}
    a_eps_s::Vector{Float64}
    s_buf::Vector{Float64}
    masses_buf::Vector{Float64}
    hv_buf::Vector{Float64}
    t_buf::Vector{Float64}
    c1buf::C1SolverBuffers
    vtmp::Vector{Float64}
    # Adaptive grid parameters
    G_base::Int          # number of base grid points
    G_ext::Int           # max extension points per side
    grid_min::Float64    # base grid min
    grid_max::Float64    # base grid max
    G_actual::Base.RefValue{Int}  # actual grid size (≤ G_base + 2*G_ext)
end

function CSplineWorkspace(G::Int, K::Int=2; grid_min=-8.0, grid_max=8.0, G_ext::Int=20)
    G = isodd(G) ? G : G+1
    G_max = G + 2*G_ext
    grid = zeros(G_max)
    sw = zeros(G_max)
    # Initialize with base grid
    base = collect(range(grid_min, grid_max, length=G))
    grid[1:G] .= base
    h = (grid_max - grid_min) / (G-1)
    sw[1]=1.0; sw[G]=1.0
    @inbounds for i in 2:G-1; sw[i] = iseven(i) ? 4.0 : 2.0; end
    @views sw[1:G] .*= h/3
    np = (K+1)*3 + 1 + 3 + 1 + 2 + 1  # 9+1+3+1+2+1 = 17 for K=2
    CSplineWorkspace(grid, sw, zeros(G_max, G_max), zeros(G_max), zeros(G_max),
                     zeros(G_max), zeros(G_max),
                     zeros(3), zeros(3), zeros(3), zeros(4),
                     zeros(K+1), zeros(3), C1SolverBuffers(), zeros(np),
                     G, G_ext, grid_min, grid_max, Ref(G))
end

# Set up adaptive grid: extend base grid to cover the full support of
# all transition densities. Tail grid points use the same spacing as the base grid.
function setup_adaptive_grid!(ws::CSplineWorkspace, a_Q::Matrix{Float64},
                               K::Int, σy::Float64)
    G_base = ws.G_base
    h_grid = (ws.grid_max - ws.grid_min) / (G_base - 1)
    base_grid = range(ws.grid_min, ws.grid_max, length=G_base)
    hv = ws.hv_buf; t = ws.t_buf

    # Find extent needed: scan base grid points for min(t₁) and max(t₃)
    t_min = Inf; t_max = -Inf
    @inbounds for g1 in 1:G_base
        z = base_grid[g1] / σy
        hv[1]=1.0; K>=1 && (hv[2]=z)
        for k in 2:K; hv[k+1] = z*hv[k] - (k-1)*hv[k-1]; end
        for l in 1:3; t[l] = dot(view(a_Q,:,l), hv); end
        t[1] < t_min && (t_min = t[1])
        t[3] > t_max && (t_max = t[3])
    end

    # Add margin for tail (a few multiples of 1/β ≈ 0.5)
    margin = 3.0
    needed_min = t_min - margin
    needed_max = t_max + margin

    # Compute number of extension points needed on each side
    n_left = max(0, min(ws.G_ext, ceil(Int, (ws.grid_min - needed_min) / h_grid)))
    n_right = max(0, min(ws.G_ext, ceil(Int, (needed_max - ws.grid_max) / h_grid)))

    G_total = G_base + n_left + n_right

    # Build grid: left extension + base + right extension
    @inbounds for i in 1:n_left
        ws.grid[i] = ws.grid_min - (n_left - i + 1) * h_grid
    end
    @inbounds for i in 1:G_base
        ws.grid[n_left + i] = base_grid[i]
    end
    @inbounds for i in 1:n_right
        ws.grid[n_left + G_base + i] = ws.grid_max + i * h_grid
    end

    # Simpson weights for the full grid
    @inbounds for i in 1:G_total
        ws.sw[i] = (i == 1 || i == G_total) ? 1.0 : (iseven(i) ? 4.0 : 2.0)
    end
    @views ws.sw[1:G_total] .*= h_grid/3
    # Zero out unused entries
    @inbounds for i in G_total+1:length(ws.sw)
        ws.sw[i] = 0.0
    end

    ws.G_actual[] = G_total
    G_total
end

function cspline_neg_loglik(a_Q::Matrix{Float64}, M_Q::Float64,
                            a_init::Vector{Float64}, M_init::Float64,
                            a_eps1::Float64, a_eps3::Float64, M_eps::Float64,
                            y::Matrix{Float64}, K::Int, σy::Float64, τ::Vector{Float64},
                            ws::CSplineWorkspace)
    N, T = size(y)

    G = ws.G_base  # fixed grid

    # Build transition matrix (C²: κ_mean_Q is model param, solver finds κ₁,κ₃)
    cspline_transition_matrix!(ws.T_mat, ws.grid, G, a_Q, M_Q, K, σy, τ,
                               ws.hv_buf, ws.t_buf, ws.s_buf, ws.masses_buf, ws.c1buf)
    ws.T_mat[1,1] < 0 && return Inf

    # Init density (C² with κ_mean = M_init)
    ws.a_init_s .= a_init
    (ws.a_init_s[2] <= ws.a_init_s[1] || ws.a_init_s[3] <= ws.a_init_s[2]) && return Inf
    βLi_ref = Ref(0.0); βRi_ref = Ref(0.0)
    κ1i_ref = Ref(0.0); κ3i_ref = Ref(0.0)
    solve_cspline_c2!(ws.s_buf, βLi_ref, βRi_ref, κ1i_ref, κ3i_ref, ws.a_init_s, τ, M_init, ws.c1buf)
    β_L_init = βLi_ref[]; β_R_init = βRi_ref[]
    κ1_init = κ1i_ref[]; κ3_init = κ3i_ref[]
    log_ref_init = ws.s_buf[1]
    @inbounds for g in 1:G
        v = cspline_eval(ws.grid[g], ws.a_init_s, ws.s_buf, β_L_init, β_R_init, κ1_init, κ3_init)
        v > log_ref_init && (log_ref_init = v)
    end
    cspline_masses!(ws.masses_buf, ws.a_init_s, ws.s_buf, β_L_init, β_R_init, κ1_init, κ3_init, log_ref_init)
    C_init_shifted = ws.masses_buf[1]+ws.masses_buf[2]+ws.masses_buf[3]+ws.masses_buf[4]
    C_init_shifted < 1e-300 && return Inf
    @inbounds for g in 1:G
        ws.f_init[g] = exp(cspline_eval(ws.grid[g], ws.a_init_s, ws.s_buf, β_L_init, β_R_init, κ1_init, κ3_init) - log_ref_init) / C_init_shifted
    end

    # Eps density (C² with κ_mean = M_eps)
    ws.a_eps_s[1] = a_eps1; ws.a_eps_s[2] = 0.0; ws.a_eps_s[3] = a_eps3
    (ws.a_eps_s[2] <= ws.a_eps_s[1] || ws.a_eps_s[3] <= ws.a_eps_s[2]) && return Inf
    βLe_ref = Ref(0.0); βRe_ref = Ref(0.0)
    κ1e_ref = Ref(0.0); κ3e_ref = Ref(0.0)
    solve_cspline_c2!(ws.s_buf, βLe_ref, βRe_ref, κ1e_ref, κ3e_ref, ws.a_eps_s, τ, M_eps, ws.c1buf)
    β_L_eps = βLe_ref[]; β_R_eps = βRe_ref[]
    κ1_eps = κ1e_ref[]; κ3_eps = κ3e_ref[]
    log_ref_eps = ws.s_buf[1]
    @inbounds for g in 1:G
        v = cspline_eval(ws.grid[g], ws.a_eps_s, ws.s_buf, β_L_eps, β_R_eps, κ1_eps, κ3_eps)
        v > log_ref_eps && (log_ref_eps = v)
    end
    cspline_masses!(ws.masses_buf, ws.a_eps_s, ws.s_buf, β_L_eps, β_R_eps, κ1_eps, κ3_eps, log_ref_eps)
    C_eps_shifted = ws.masses_buf[1]+ws.masses_buf[2]+ws.masses_buf[3]+ws.masses_buf[4]
    C_eps_shifted < 1e-300 && return Inf

    total_ll = 0.0

    # Views for the active portion of the grid
    grid_v = view(ws.grid, 1:G)

    # Log-density buffer for logspace normalization
    log_p = zeros(G)
    pw_buf = zeros(G)
    p_new_v = view(ws.p_new, 1:G)
    T_v = view(ws.T_mat, 1:G, 1:G)

    @inbounds for i in 1:N
        # ---- t=1: log p(g) = log f_init(g) + log f_eps(y₁-g) ----
        for g in 1:G
            log_p[g] = log(max(ws.f_init[g], 1e-300)) +
                        cspline_eval(y[i,1]-ws.grid[g], ws.a_eps_s, ws.s_buf,
                                     β_L_eps, β_R_eps, κ1_eps, κ3_eps) - log_ref_eps - log(C_eps_shifted)
        end
        # Normalize using logspace integration (exact exp(cubic) per segment)
        L1 = logspace_integrate(log_p, grid_v, G)
        L1 < 1e-300 && return Inf
        total_ll += log(L1)
        # Normalized p for prediction step
        @inbounds for g in 1:G; ws.p[g] = exp(log_p[g]) / L1; end

        for t_step in 2:T
            # ---- Prediction: p_pred = T' × (p ⊙ sw) via matrix-vector ----
            # (logspace not beneficial here: log T has complex η-dependence)
            @inbounds for g in 1:G; pw_buf[g] = ws.p[g] * ws.sw[g]; end
            mul!(p_new_v, transpose(T_v), view(pw_buf, 1:G))

            # ---- Observation update in log-space ----
            for g in 1:G
                log_p[g] = log(max(ws.p_new[g], 1e-300)) +
                            cspline_eval(y[i,t_step]-ws.grid[g], ws.a_eps_s, ws.s_buf,
                                         β_L_eps, β_R_eps, κ1_eps, κ3_eps) - log_ref_eps - log(C_eps_shifted)
            end
            Lt = logspace_integrate(log_p, grid_v, G)
            Lt < 1e-300 && return Inf
            total_ll += log(Lt)
            @inbounds for g in 1:G; ws.p[g] = exp(log_p[g]) / Lt; end
        end
    end
    -total_ll / N
end

# ================================================================
#  GPU-ACCELERATED FORWARD FILTER (optional, requires CUDA.jl)
#
#  Best practices from https://cuda.juliagpu.org/stable/tutorials/performance/
#  - Batch N observations into single GEMM: T'×P where P is [G×N]
#  - Use CuArray broadcasting for element-wise f_eps evaluation
#  - Minimize CPU↔GPU transfers: keep P on GPU throughout filter
#  - Use Float64 (needed for likelihood precision)
#  - Avoid scalar indexing on GPU arrays
# ================================================================

"""
GPU-batched forward filter. Requires CUDA.jl to be loaded.
Falls back to CPU if CUDA is not available.

Key GPU operation: instead of N sequential G×G mat-vec products,
do a single G×G × G×N matrix-matrix multiply per time step.
"""
function cspline_neg_loglik_gpu(a_Q::Matrix{Float64}, M_Q::Float64,
                                a_init::Vector{Float64}, M_init::Float64,
                                a_eps1::Float64, a_eps3::Float64, M_eps::Float64,
                                y::Matrix{Float64}, K::Int, σy::Float64, τ::Vector{Float64},
                                ws::CSplineWorkspace)
    N, T_obs = size(y)
    G = ws.G_base

    # Build transition matrix on CPU (G Newton solves — sequential)
    cspline_transition_matrix!(ws.T_mat, ws.grid, G, a_Q, M_Q, K, σy, τ,
                               ws.hv_buf, ws.t_buf, ws.s_buf, ws.masses_buf, ws.c1buf)
    ws.T_mat[1,1] < 0 && return Inf

    # Init density on CPU
    ws.a_init_s .= a_init
    (ws.a_init_s[2] <= ws.a_init_s[1] || ws.a_init_s[3] <= ws.a_init_s[2]) && return Inf
    βLi = Ref(0.0); βRi = Ref(0.0); κ1i = Ref(0.0); κ3i = Ref(0.0)
    solve_cspline_c2!(ws.s_buf, βLi, βRi, κ1i, κ3i, ws.a_init_s, τ, M_init, ws.c1buf)
    log_ref_init = ws.s_buf[1]
    @inbounds for g in 1:G
        v = cspline_eval(ws.grid[g], ws.a_init_s, ws.s_buf, βLi[], βRi[], κ1i[], κ3i[])
        v > log_ref_init && (log_ref_init = v)
    end
    cspline_masses!(ws.masses_buf, ws.a_init_s, ws.s_buf, βLi[], βRi[], κ1i[], κ3i[], log_ref_init)
    C_init = ws.masses_buf[1]+ws.masses_buf[2]+ws.masses_buf[3]+ws.masses_buf[4]
    C_init < 1e-300 && return Inf
    f_init_cpu = [exp(cspline_eval(ws.grid[g], ws.a_init_s, ws.s_buf, βLi[], βRi[], κ1i[], κ3i[]) - log_ref_init) / C_init for g in 1:G]

    # Eps density on CPU — precompute log f_eps on a fine grid for interpolation
    a_eps_s = [a_eps1, 0.0, a_eps3]
    (a_eps_s[2] <= a_eps_s[1] || a_eps_s[3] <= a_eps_s[2]) && return Inf
    s_eps = zeros(3); βLe = Ref(0.0); βRe = Ref(0.0); κ1e = Ref(0.0); κ3e = Ref(0.0)
    solve_cspline_c2!(s_eps, βLe, βRe, κ1e, κ3e, a_eps_s, τ, M_eps, ws.c1buf)
    log_ref_eps = s_eps[1]
    @inbounds for g in 1:G
        v = cspline_eval(ws.grid[g], a_eps_s, s_eps, βLe[], βRe[], κ1e[], κ3e[])
        v > log_ref_eps && (log_ref_eps = v)
    end
    cspline_masses!(ws.masses_buf, a_eps_s, s_eps, βLe[], βRe[], κ1e[], κ3e[], log_ref_eps)
    C_eps = ws.masses_buf[1]+ws.masses_buf[2]+ws.masses_buf[3]+ws.masses_buf[4]
    C_eps < 1e-300 && return Inf

    # Precompute f_eps(y[i,t]-grid[g]) for all (i,t,g) on CPU
    # Store as eps_dens[g, i, t] to batch across observations
    eps_dens = zeros(G, N, T_obs)
    @inbounds for t_step in 1:T_obs, i in 1:N, g in 1:G
        eps_x = y[i, t_step] - ws.grid[g]
        eps_dens[g, i, t_step] = exp(cspline_eval(eps_x, a_eps_s, s_eps, βLe[], βRe[], κ1e[], κ3e[]) - log_ref_eps) / C_eps
    end

    # Check if CUDA is loaded and functional
    gpu_available = isdefined(Main, :CUDA) && isdefined(Main, :CuArray)

    if gpu_available
        CuArray_fn = Main.CuArray
        # Transfer to GPU (single transfer, minimize CPU↔GPU copies)
        T_d = CuArray_fn(view(ws.T_mat, 1:G, 1:G))  # G×G
        sw_d = CuArray_fn(ws.sw[1:G])                  # G
        f_init_d = CuArray_fn(f_init_cpu)               # G
        eps_d = CuArray_fn(eps_dens)                     # G×N×T

        # t=1: P[g,i] = f_init[g] × f_eps[g,i,1]
        P_d = f_init_d .* view(eps_d, :, :, 1)  # G×N, broadcasted on GPU

        # L[i] = Σ_g P[g,i] × sw[g]  → column-wise weighted sum
        L_d = sw_d' * P_d   # 1×N dot products via GEMV
        L_cpu = Array(L_d)   # bring back for log
        any(L_cpu .< 1e-300) && return Inf
        total_ll = sum(log.(L_cpu))
        P_d ./= L_d          # normalize each column

        # t≥2: prediction step via batched GEMM
        for t_step in 2:T_obs
            # PW[g,i] = P[g,i] × sw[g]
            PW_d = P_d .* sw_d  # G×N, GPU broadcast
            # P_pred = T' × PW — single GEMM (cuBLAS), the main GPU win
            P_pred_d = transpose(T_d) * PW_d  # G×N
            # Multiply by f_eps
            P_d = P_pred_d .* view(eps_d, :, :, t_step)
            # Normalize
            L_d = sw_d' * P_d
            L_cpu = Array(L_d)
            any(L_cpu .< 1e-300) && return Inf
            total_ll += sum(log.(L_cpu))
            P_d ./= L_d
        end

        return -total_ll / N
    end

    # CPU fallback: batched matrix multiply (still faster than per-observation)
    T_cpu = view(ws.T_mat, 1:G, 1:G)
    sw_cpu = ws.sw[1:G]

    P = zeros(G, N)
    @inbounds for i in 1:N, g in 1:G
        P[g, i] = f_init_cpu[g] * eps_dens[g, i, 1]
    end
    L_vec = transpose(sw_cpu) * P  # 1×N
    any(L_vec .< 1e-300) && return Inf
    total_ll = sum(log.(L_vec))
    P ./= L_vec

    PW = zeros(G, N)
    P_pred = zeros(G, N)
    for t_step in 2:T_obs
        @inbounds for i in 1:N, g in 1:G; PW[g, i] = P[g, i] * sw_cpu[g]; end
        mul!(P_pred, transpose(T_cpu), PW)  # single GEMM
        @inbounds for i in 1:N, g in 1:G
            P[g, i] = P_pred[g, i] * eps_dens[g, i, t_step]
        end
        L_vec = transpose(sw_cpu) * P
        any(L_vec .< 1e-300) && return Inf
        total_ll += sum(log.(L_vec))
        P ./= L_vec
    end

    -total_ll / N
end

# ================================================================
#  PARAMETER PACKING
# ================================================================

# C² model: β determined by spline, quantile ordering guaranteed by construction.
#
# a_Q reparameterization for K=2:
#   Median quantile: a_Q[:,2] = (m₀, m₁, m₂) — free
#   Left gap d_L = a_Q[:,2] - a_Q[:,1]: d_L(z) = d₀ + d₁z + d₂(z²-1) > 0 ∀z
#     Enforced by: d₂ = exp(δ₁) > 0
#                  d₀ = d₂ + exp(δ₂)  (so d₀ > d₂)
#                  d₁ = 2√(d₂(d₀-d₂)) tanh(δ₃)  (discriminant < 0)
#   Right gap d_R = a_Q[:,3] - a_Q[:,2]: same structure with (δ₄, δ₅, δ₆)
#
# Order: [median(3), δ_L(3), δ_R(3), log(-M_Q),
#         a_init(3), log(-M_init),
#         a_eps(2), log(-M_eps)] = 17 total for K=2
# M₁ = M₃ = M for each density (symmetric tails; profiled to satisfy F(t₂)=τ₂)

# Helper: convert (δ₁,δ₂,δ₃) → (d₀,d₁,d₂) with d₂z²+d₁z+(d₀-d₂) > 0 ∀z
# d₂ = exp(δ₁) > 0 strictly (excludes d₂=0, which is measure zero)
function gap_from_delta(δ₁::Float64, δ₂::Float64, δ₃::Float64)
    d2 = exp(δ₁)               # > 0
    d0 = d2 + exp(δ₂)          # > d₂
    d1 = 2.0 * sqrt(d2 * exp(δ₂)) * tanh(δ₃)  # |d₁| < 2√(d₂(d₀-d₂))
    (d0, d1, d2)
end

# Helper: convert (d₀,d₁,d₂) → (δ₁,δ₂,δ₃)
function delta_from_gap(d0::Float64, d1::Float64, d2::Float64)
    d2 = max(d2, 1e-10)        # project to interior if on boundary
    gap = max(d0 - d2, 1e-10)
    δ₁ = log(d2)
    δ₂ = log(gap)
    bound = 2.0 * sqrt(d2 * gap)
    δ₃ = atanh(clamp(d1 / bound, -0.9999, 0.9999))
    (δ₁, δ₂, δ₃)
end

function pack_cspline(a_Q::Matrix{Float64}, M_Q::Float64,
                      a_init::Vector{Float64}, M_init::Float64,
                      a_eps1::Float64, a_eps3::Float64, M_eps::Float64)
    median_q = a_Q[:, 2]
    dL0 = a_Q[1,2] - a_Q[1,1]; dL1 = a_Q[2,2] - a_Q[2,1]; dL2 = a_Q[3,2] - a_Q[3,1]
    δL1, δL2, δL3 = delta_from_gap(dL0, dL1, dL2)
    dR0 = a_Q[1,3] - a_Q[1,2]; dR1 = a_Q[2,3] - a_Q[2,2]; dR2 = a_Q[3,3] - a_Q[3,2]
    δR1, δR2, δR3 = delta_from_gap(dR0, dR1, dR2)
    init_median = a_init[2]
    init_log_gap_L = log(a_init[2] - a_init[1])
    init_log_gap_R = log(a_init[3] - a_init[2])
    vcat(median_q, δL1, δL2, δL3, δR1, δR2, δR3,
         log(-M_Q),
         init_median, init_log_gap_L, init_log_gap_R,
         log(-M_init),
         log(-a_eps1), log(a_eps3),
         log(-M_eps))  # 3+3+3+1+3+1+2+1 = 17
end

function unpack_cspline(v::Vector{Float64}, K::Int)
    nk = K + 1  # 3 for K=2
    median_q = v[1:nk]
    dL0, dL1, dL2 = gap_from_delta(v[nk+1], v[nk+2], v[nk+3])
    dR0, dR1, dR2 = gap_from_delta(v[nk+4], v[nk+5], v[nk+6])
    a_Q = zeros(nk, 3)
    a_Q[:, 2] .= median_q
    a_Q[1,1] = median_q[1] - dL0; a_Q[2,1] = median_q[2] - dL1; a_Q[3,1] = median_q[3] - dL2
    a_Q[1,3] = median_q[1] + dR0; a_Q[2,3] = median_q[2] + dR1; a_Q[3,3] = median_q[3] + dR2
    M_Q = -exp(v[nk+7])  # M₁=M₃=M for symmetric tails
    p = nk + 7
    init_median = v[p+1]
    gap_L = exp(v[p+2])
    gap_R = exp(v[p+3])
    a_init = [init_median - gap_L, init_median, init_median + gap_R]
    M_init = -exp(v[p+4])
    a_eps1 = -exp(v[p+5])
    a_eps3 = exp(v[p+6])
    M_eps = -exp(v[p+7])
    (a_Q, M_Q, a_init, M_init, a_eps1, a_eps3, M_eps)
end

# ================================================================
#  DATA GENERATION
# ================================================================

"""Draw from cubic spline density by inverse CDF on fine grid."""
function cspline_draw(rng::AbstractRNG, t::Vector{Float64}, s::Vector{Float64},
                      β_L::Float64, β_R::Float64, M1::Float64, M3::Float64, C::Float64;
                      grid_min=-8.0, grid_max=8.0, n_grid=500)
    grid = collect(range(grid_min, grid_max, length=n_grid))
    dg = (grid_max - grid_min) / (n_grid - 1)
    cdf = zeros(n_grid)
    @inbounds for g in 1:n_grid
        cdf[g] = exp(cspline_eval(grid[g], t, s, β_L, β_R, M1, M3)) / C
    end
    cdf .*= dg
    cumsum!(cdf, cdf)
    cdf ./= cdf[end]
    # Inverse CDF
    u = rand(rng)
    idx = searchsortedfirst(cdf, u)
    idx = clamp(idx, 1, n_grid)
    grid[idx]
end

# C² version: β determined by spline, quadratic tails with κ_mean
function generate_data_cspline(N::Int, a_Q::Matrix{Float64}, M_Q::Float64,
                                a_init::Vector{Float64}, M_init::Float64,
                                a_eps1::Float64, a_eps3::Float64, M_eps::Float64,
                                K::Int, σy::Float64, τ::Vector{Float64};
                                seed::Int=42)
    rng = MersenneTwister(seed)
    T = 3
    eta = zeros(N, T); y = zeros(N, T)
    hv = zeros(K+1); t = zeros(3); s = zeros(3); masses = zeros(4)
    βL_ref = Ref(0.0); βR_ref = Ref(0.0)
    κ1_ref = Ref(0.0); κ3_ref = Ref(0.0)

    # Init density
    a_init_s = a_init
    s_init = zeros(3)
    solve_cspline_c2!(s_init, βL_ref, βR_ref, κ1_ref, κ3_ref, a_init_s, τ, M_init)
    β_L_init = βL_ref[]; β_R_init = βR_ref[]
    κ1_init = κ1_ref[]; κ3_init = κ3_ref[]
    m_init = zeros(4)
    cspline_masses!(m_init, a_init_s, s_init, β_L_init, β_R_init, κ1_init, κ3_init, 0.0)
    C_init = sum(m_init)

    # Eps density
    a_eps_s = [a_eps1, 0.0, a_eps3]
    s_eps = zeros(3)
    solve_cspline_c2!(s_eps, βL_ref, βR_ref, κ1_ref, κ3_ref, a_eps_s, τ, M_eps)
    β_L_eps = βL_ref[]; β_R_eps = βR_ref[]
    κ1_eps = κ1_ref[]; κ3_eps = κ3_ref[]
    m_eps = zeros(4)
    cspline_masses!(m_eps, a_eps_s, s_eps, β_L_eps, β_R_eps, κ1_eps, κ3_eps, 0.0)
    C_eps = sum(m_eps)

    for i in 1:N
        eta[i,1] = cspline_draw(rng, a_init_s, s_init, β_L_init, β_R_init, κ1_init, κ3_init, C_init)
    end
    for t_step in 2:T, i in 1:N
        z = eta[i,t_step-1] / σy
        hv[1]=1.0; K>=1 && (hv[2]=z)
        for k in 2:K; hv[k+1] = z*hv[k] - (k-1)*hv[k-1]; end
        for l in 1:3; t[l] = dot(view(a_Q,:,l), hv); end
        if t[2] <= t[1] || t[3] <= t[2]; continue; end
        solve_cspline_c2!(s, βL_ref, βR_ref, κ1_ref, κ3_ref, t, τ, M_Q)
        cspline_masses!(masses, t, s, βL_ref[], βR_ref[], κ1_ref[], κ3_ref[], 0.0)
        C = masses[1]+masses[2]+masses[3]+masses[4]
        eta[i,t_step] = cspline_draw(rng, t, s, βL_ref[], βR_ref[], κ1_ref[], κ3_ref[], C)
    end
    for t_step in 1:T, i in 1:N
        y[i,t_step] = eta[i,t_step] + cspline_draw(rng, a_eps_s, s_eps, β_L_eps, β_R_eps, κ1_eps, κ3_eps, C_eps)
    end
    y, eta
end

# ================================================================
#  ANALYTICAL GRADIENT OF NEG-LOG-LIKELIHOOD
#
#  Computes nll and gradient in a single forward pass.
#  Score identity: ∂log f(x)/∂θ = ∂S(x)/∂θ − E[∂S/∂θ]
#  Tangent propagation through the forward filter.
# ================================================================

function cspline_neg_loglik_and_grad!(grad_v_out::Vector{Float64},
        v::Vector{Float64}, y::Matrix{Float64}, K::Int, σy::Float64,
        τ::Vector{Float64}, ws::CSplineWorkspace)
    np = length(v)
    nk = K + 1  # = 3 for K=2
    a_Q, M_Q, a_init, M_init, a_eps1, a_eps3, M_eps = unpack_cspline(v, K)
    N, T_obs = size(y)
    G = ws.G_base
    buf = ws.c1buf
    βLr = Ref(0.0); βRr = Ref(0.0); κ1r = Ref(0.0); κ3r = Ref(0.0)

    # ============================================================
    # PHASE 1: Build transition matrix + score matrices W[g₁,g₂,p]
    # W[g₁,g₂,p] = T(g₁,g₂) × [dS_p(g₂;t(g₁)) − Ē_p(g₁)]
    # p=1,2,3 for dt₁,dt₂,dt₃;  p=4 for dκ_mean
    # ============================================================
    W_score = zeros(G, G, 4)
    t_loc = zeros(3); s_loc = zeros(3)

    @inbounds for g1 in 1:G
        z = ws.grid[g1] / σy
        hv_loc = zeros(nk); hv_loc[1]=1.0; K>=1 && (hv_loc[2]=z)
        for k in 2:K; hv_loc[k+1] = z*hv_loc[k] - (k-1)*hv_loc[k-1]; end
        for l in 1:3; t_loc[l] = dot(view(a_Q,:,l), hv_loc); end

        if t_loc[2] <= t_loc[1] || t_loc[3] <= t_loc[2]
            for g2 in 1:G; ws.T_mat[g1,g2] = 1e-300; end
            continue
        end

        solve_cspline_c2!(s_loc, βLr, βRr, κ1r, κ3r, t_loc, τ, M_Q, buf)
        bL=βLr[]; bR=βRr[]; k1=κ1r[]; k3=κ3r[]
        δ_v = (k3-k1)/2
        ds_dt, dδ_dt, ds_dκ, dδ_dκ = cspline_solver_ift(t_loc, s_loc, M_Q, δ_v, buf)

        lr = s_loc[1]
        for g2 in 1:G
            vv = cspline_eval(ws.grid[g2], t_loc, s_loc, bL, bR, k1, k3)
            vv > lr && (lr = vv)
        end
        cspline_masses!(ws.masses_buf, t_loc, s_loc, bL, bR, k1, k3, lr)
        C_sh = ws.masses_buf[1]+ws.masses_buf[2]+ws.masses_buf[3]+ws.masses_buf[4]
        if C_sh < 1e-300
            for g2 in 1:G; ws.T_mat[g1,g2] = 1e-300; end
            continue
        end

        Ē = zeros(4)
        for g2 in 1:G
            Tv = exp(cspline_eval(ws.grid[g2], t_loc, s_loc, bL, bR, k1, k3) - lr) / C_sh
            ws.T_mat[g1,g2] = Tv
            dS = cspline_total_score(ws.grid[g2], t_loc, s_loc, bL, bR, k1, k3,
                                      ds_dt, dδ_dt, ds_dκ, dδ_dκ)
            for p in 1:4; W_score[g1,g2,p] = dS[p]; Ē[p] += Tv*dS[p]*ws.sw[g2]; end
        end
        for g2 in 1:G
            Tv = ws.T_mat[g1,g2]
            for p in 1:4; W_score[g1,g2,p] = Tv*(W_score[g1,g2,p]-Ē[p]); end
        end
    end

    # ============================================================
    # PHASE 2: Init density + log-density scores dlogf_init[g,p]
    # ============================================================
    ws.a_init_s .= a_init
    (ws.a_init_s[2]<=ws.a_init_s[1] || ws.a_init_s[3]<=ws.a_init_s[2]) && (fill!(grad_v,0.0); return 1e10)
    solve_cspline_c2!(ws.s_buf, βLr, βRr, κ1r, κ3r, ws.a_init_s, τ, M_init, buf)
    bLi=βLr[]; bRi=βRr[]; k1i=κ1r[]; k3i=κ3r[]
    δ_i=(k3i-k1i)/2
    ds_dt_i,dδ_dt_i,ds_dκ_i,dδ_dκ_i = cspline_solver_ift(ws.a_init_s,ws.s_buf,M_init,δ_i,buf)

    lr_i = ws.s_buf[1]
    @inbounds for g in 1:G
        vv=cspline_eval(ws.grid[g],ws.a_init_s,ws.s_buf,bLi,bRi,k1i,k3i); vv>lr_i&&(lr_i=vv)
    end
    cspline_masses!(ws.masses_buf,ws.a_init_s,ws.s_buf,bLi,bRi,k1i,k3i,lr_i)
    C_i=ws.masses_buf[1]+ws.masses_buf[2]+ws.masses_buf[3]+ws.masses_buf[4]
    C_i<1e-300 && (fill!(grad_v,0.0); return 1e10)

    dlogf_init = zeros(G,4); Ē_i = zeros(4)
    @inbounds for g in 1:G
        ws.f_init[g]=exp(cspline_eval(ws.grid[g],ws.a_init_s,ws.s_buf,bLi,bRi,k1i,k3i)-lr_i)/C_i
        dS=cspline_total_score(ws.grid[g],ws.a_init_s,ws.s_buf,bLi,bRi,k1i,k3i,ds_dt_i,dδ_dt_i,ds_dκ_i,dδ_dκ_i)
        for p in 1:4; dlogf_init[g,p]=dS[p]; Ē_i[p]+=ws.f_init[g]*dS[p]*ws.sw[g]; end
    end
    for g in 1:G, p in 1:4; dlogf_init[g,p]-=Ē_i[p]; end

    # ============================================================
    # PHASE 3: Eps density + expected score Ē_eps
    # Score evaluated on-the-fly at y-grid[g]; Ē_eps precomputed.
    # ============================================================
    ws.a_eps_s[1]=a_eps1; ws.a_eps_s[2]=0.0; ws.a_eps_s[3]=a_eps3
    (ws.a_eps_s[2]<=ws.a_eps_s[1] || ws.a_eps_s[3]<=ws.a_eps_s[2]) && (fill!(grad_v,0.0); return 1e10)
    s_eps_save = zeros(3)
    solve_cspline_c2!(s_eps_save, βLr, βRr, κ1r, κ3r, ws.a_eps_s, τ, M_eps, buf)
    bLe=βLr[]; bRe=βRr[]; k1e=κ1r[]; k3e=κ3r[]
    δ_e=(k3e-k1e)/2
    ds_dt_e,dδ_dt_e,ds_dκ_e,dδ_dκ_e = cspline_solver_ift(ws.a_eps_s,s_eps_save,M_eps,δ_e,buf)

    lr_e = s_eps_save[1]
    @inbounds for g in 1:G
        vv=cspline_eval(ws.grid[g],ws.a_eps_s,s_eps_save,bLe,bRe,k1e,k3e); vv>lr_e&&(lr_e=vv)
    end
    cspline_masses!(ws.masses_buf,ws.a_eps_s,s_eps_save,bLe,bRe,k1e,k3e,lr_e)
    C_e=ws.masses_buf[1]+ws.masses_buf[2]+ws.masses_buf[3]+ws.masses_buf[4]
    C_e<1e-300 && (fill!(grad_v,0.0); return 1e10)

    # Precompute Ē_eps = ∫f_eps(u) dS(u)/dθ du (on a grid covering eps support)
    Ē_e = zeros(4)  # dt1,dt2,dt3,dκ for eps
    eps_grid_n = 401
    eps_lo = ws.a_eps_s[1] - 5.0/sqrt(-M_eps)  # ~5σ below t₁
    eps_hi = ws.a_eps_s[3] + 5.0/sqrt(-M_eps)
    eps_dg = (eps_hi-eps_lo)/(eps_grid_n-1)
    for ig in 1:eps_grid_n
        u = eps_lo + (ig-1)*eps_dg
        fu = exp(cspline_eval(u,ws.a_eps_s,s_eps_save,bLe,bRe,k1e,k3e)-lr_e)/C_e
        dSu = cspline_total_score(u,ws.a_eps_s,s_eps_save,bLe,bRe,k1e,k3e,ds_dt_e,dδ_dt_e,ds_dκ_e,dδ_dκ_e)
        w = (ig==1||ig==eps_grid_n) ? 1.0 : (iseven(ig) ? 4.0 : 2.0)
        for p in 1:4; Ē_e[p] += fu*dSu[p]*w*eps_dg/3; end
    end

    # ============================================================
    # PHASE 4: Forward filter + tangent gradient accumulation
    # ============================================================
    # Flat gradient accumulator (unpacked params):
    # 1..nk*3: a_Q[k,l] col-major, nk*3+1: M_Q,
    # nk*3+2..nk*3+4: a_init[1:3], nk*3+5: M_init,
    # nk*3+6: a_eps1, nk*3+7: a_eps3, nk*3+8: M_eps

    # Hermite values at grid points
    hv_grid = zeros(G, nk)
    @inbounds for g in 1:G
        z = ws.grid[g]/σy; hv_grid[g,1]=1.0
        nk>=2 && (hv_grid[g,2]=z)
        for k in 2:K; hv_grid[g,k+1]=z*hv_grid[g,k]-(k-1)*hv_grid[g,k-1]; end
    end

    # Per-individual tangent vectors:
    # n_Q = nk*3 + 1 transition params (a_Q[k,l] and M_Q)
    n_Q = nk*3+1; n_I = 4; n_E = 3
    α = zeros(G, n_Q+n_I+n_E)
    α_new = zeros(G, n_Q+n_I+n_E)
    dp = zeros(G)  # temp for ∂p/∂θ
    n_tot = n_Q+n_I+n_E
    grad_unp = zeros(n_tot)  # flat gradient accumulator

    total_ll = 0.0
    p_v = view(ws.p, 1:G)
    p_new_v = view(ws.p_new, 1:G)
    sw_v = view(ws.sw, 1:G)
    T_v = view(ws.T_mat, 1:G, 1:G)

    @inbounds for i in 1:N
        # ---- t=1: p(g) = f_init(g) × f_eps(y₁-g) ----
        fill!(α, 0.0)
        for g in 1:G
            eps_x = y[i,1]-ws.grid[g]
            f_e = exp(cspline_eval(eps_x,ws.a_eps_s,s_eps_save,bLe,bRe,k1e,k3e)-lr_e)/C_e
            ws.p[g] = ws.f_init[g]*f_e

            # Init tangents: ∂p/∂θ_init = p × dlogf_init[g,p]
            for j in 1:n_I; α[g, n_Q+j] = ws.p[g]*dlogf_init[g,j]; end

            # Eps tangents: ∂p/∂θ_eps = p × dlogf_eps(eps_x)
            dS_e = cspline_total_score(eps_x,ws.a_eps_s,s_eps_save,bLe,bRe,k1e,k3e,
                                        ds_dt_e,dδ_dt_e,ds_dκ_e,dδ_dκ_e)
            # eps params: (dt₁→ae1, dt₃→ae3, dκ→Me)
            α[g, n_Q+n_I+1] = ws.p[g]*(dS_e[1]-Ē_e[1])  # dae1 → dt₁
            α[g, n_Q+n_I+2] = ws.p[g]*(dS_e[3]-Ē_e[3])  # dae3 → dt₃
            α[g, n_Q+n_I+3] = ws.p[g]*(dS_e[4]-Ē_e[4])  # dMe → dκ
            # Transition tangents = 0 (no T at t=1)
        end

        L1 = dot(p_v, sw_v)
        L1 < 1e-300 && (fill!(grad_v_out,0.0); return 1e10)
        for j in 1:n_tot
            dL = 0.0
            for g in 1:G; dL += α[g,j]*ws.sw[g]; end
            for g in 1:G; α[g,j] = (α[g,j] - ws.p[g]/L1*dL)/L1; end
            grad_unp[j] -= dL/(L1*N)
        end
        total_ll += log(L1); p_v ./= L1

        # ---- t≥2 ----
        for t_step in 2:T_obs
            @inbounds for g in 1:G; ws.pw[g] = ws.p[g]*ws.sw[g]; end
            mul!(p_new_v, transpose(T_v), view(ws.pw,1:G))

            for g2 in 1:G
                eps_x = y[i,t_step]-ws.grid[g2]
                f_e = exp(cspline_eval(eps_x,ws.a_eps_s,s_eps_save,bLe,bRe,k1e,k3e)-lr_e)/C_e
                ws.p_new[g2] *= f_e
            end

            # Tangent for each parameter
            fill!(α_new, 0.0)
            for j in 1:n_tot
                # D_j(g') = Σ_g T(g,g') α[g,j] sw[g]  (propagation from previous step)
                for g2 in 1:G
                    d = 0.0
                    for g1 in 1:G; d += ws.T_mat[g1,g2]*α[g1,j]*ws.sw[g1]; end
                    dp[g2] = d
                end

                # B_j(g') = Σ_g W_score_j(g,g') pw(g) (transition score term, only for Q params)
                if j <= n_Q
                    # Map j to (knot_l, herm_k) or κ
                    if j <= nk*3  # a_Q params
                        l_idx = ((j-1) ÷ nk) + 1  # knot index 1,2,3
                        k_idx = ((j-1) % nk) + 1   # Hermite index 1,...,nk
                        for g2 in 1:G
                            b = 0.0
                            for g1 in 1:G; b += W_score[g1,g2,l_idx]*hv_grid[g1,k_idx]*ws.pw[g1]; end
                            dp[g2] += b
                        end
                    else  # M_Q (κ_mean)
                        for g2 in 1:G
                            b = 0.0
                            for g1 in 1:G; b += W_score[g1,g2,4]*ws.pw[g1]; end
                            dp[g2] += b
                        end
                    end
                end

                # ∂p_new/∂θ = dp × f_eps + p_pred × ∂f_eps/∂θ
                # For eps params, add the eps score contribution
                for g2 in 1:G
                    eps_x = y[i,t_step]-ws.grid[g2]
                    f_e = exp(cspline_eval(eps_x,ws.a_eps_s,s_eps_save,bLe,bRe,k1e,k3e)-lr_e)/C_e
                    α_new[g2,j] = dp[g2]*f_e
                end
                if j > n_Q+n_I  # eps param
                    ep = j - n_Q - n_I  # 1=ae1, 2=ae3, 3=Me
                    score_idx = ep==1 ? 1 : ep==2 ? 3 : 4  # dt₁, dt₃, dκ
                    for g2 in 1:G
                        eps_x = y[i,t_step]-ws.grid[g2]
                        dS_e = cspline_total_score(eps_x,ws.a_eps_s,s_eps_save,bLe,bRe,k1e,k3e,
                                                    ds_dt_e,dδ_dt_e,ds_dκ_e,dδ_dκ_e)
                        # p_pred(g₂) = p_new(g₂)/f_eps
                        p_pred = ws.p_new[g2]  # already has f_e multiplied
                        α_new[g2,j] += p_pred*(dS_e[score_idx]-Ē_e[score_idx])
                    end
                end
            end

            Lt = dot(p_new_v, sw_v)
            Lt < 1e-300 && (fill!(grad_v_out,0.0); return 1e10)

            for j in 1:n_tot
                dL = 0.0
                for g in 1:G; dL += α_new[g,j]*ws.sw[g]; end
                for g in 1:G; α_new[g,j] = (α_new[g,j] - ws.p_new[g]/Lt*dL)/Lt; end
                grad_unp[j] -= dL/(Lt*N)
            end
            total_ll += log(Lt); p_new_v ./= Lt
            @inbounds for g in 1:G; ws.p[g]=ws.p_new[g]; end
            α .= α_new
        end
    end

    nll = -total_ll/N

    # ============================================================
    # PHASE 5: Chain rule — unpacked → packed gradient (ANALYTICAL)
    # grad_unp[j] = ∂nll/∂θ_j for unpacked params θ.
    # Need ∂nll/∂v = Σ_j (∂nll/∂θ_j)(∂θ_j/∂v_i).
    #
    # Packed v layout (K=2, nk=3):
    #   v[1:3]: median_q = a_Q[:,2]
    #   v[4:6]: δ_L = (δ₁,δ₂,δ₃) for left gap
    #   v[7:9]: δ_R = (δ₁,δ₂,δ₃) for right gap
    #   v[10]: log(-M_Q)
    #   v[11]: init_median = a_init[2]
    #   v[12]: log(a_init[2]-a_init[1])
    #   v[13]: log(a_init[3]-a_init[2])
    #   v[14]: log(-M_init)
    #   v[15]: log(-a_eps1)
    #   v[16]: log(a_eps3)
    #   v[17]: log(-M_eps)
    # ============================================================
    fill!(grad_v_out, 0.0)

    # -- Median q: v[k] = a_Q[k,2], so ∂a_Q[k,2]/∂v[k] = 1
    #    Also a_Q[k,1] = v[k] - d_L[k], a_Q[k,3] = v[k] + d_R[k]
    #    ∂a_Q[k,1]/∂v[k] = 1, ∂a_Q[k,3]/∂v[k] = 1 (median shifts all three)
    for k in 1:nk
        idx1 = k; idx2 = nk+k; idx3 = 2*nk+k  # grad_unp indices for a_Q[k,1], a_Q[k,2], a_Q[k,3]
        grad_v_out[k] = grad_unp[idx1] + grad_unp[idx2] + grad_unp[idx3]
    end

    # -- Left gap: v[nk+1:nk+3] = (δ₁,δ₂,δ₃) → d_L = (d₀,d₁,d₂) → a_Q[:,1] = median - d_L
    # d₂ = exp(δ₁), d₀ = d₂ + exp(δ₂), d₁ = 2√(d₂·exp(δ₂))·tanh(δ₃)
    # a_Q[1,1] = median[1] - d₀, a_Q[2,1] = median[2] - d₁, a_Q[3,1] = median[3] - d₂
    δ₁L = v[nk+1]; δ₂L = v[nk+2]; δ₃L = v[nk+3]
    d2L = exp(δ₁L); eδ2L = exp(δ₂L); d0L = d2L + eδ2L
    sqL = sqrt(d2L * eδ2L); tanhL = tanh(δ₃L); d1L = 2.0*sqL*tanhL

    # ∂d₀/∂δ₁ = d₂, ∂d₀/∂δ₂ = eδ₂
    # ∂d₁/∂δ₁ = 2·(eδ₂/(2√(d₂eδ₂)))·d₂·tanh = sqL·tanh·(eδ₂/sqL)... let me compute directly
    # d₁ = 2√(d₂eδ₂)·tanh(δ₃). ∂d₂/∂δ₁ = d₂, ∂eδ₂/∂δ₁ = 0
    # ∂d₁/∂δ₁ = 2·tanh·∂√(d₂eδ₂)/∂δ₁ = 2·tanh·eδ₂/(2√(d₂eδ₂))·d₂ = tanh·d₂·eδ₂/sqL = tanh·sqL
    # Wait: ∂√(d₂eδ₂)/∂δ₁ = (eδ₂·d₂)/(2√(d₂eδ₂)) = sqL/2 ... no.
    # √(d₂eδ₂) = sqL. d₂ = exp(δ₁). ∂d₂/∂δ₁ = d₂.
    # ∂sqL/∂δ₁ = ∂√(d₂eδ₂)/∂δ₁ = eδ₂·d₂/(2sqL) = d₂·eδ₂/(2sqL)
    # Hmm: sqL² = d₂·eδ₂. ∂(sqL²)/∂δ₁ = eδ₂·d₂. So 2sqL·∂sqL/∂δ₁ = eδ₂·d₂ → ∂sqL/∂δ₁ = eδ₂·d₂/(2sqL)
    # Then ∂d₁/∂δ₁ = 2·tanh·eδ₂·d₂/(2sqL) = tanh·eδ₂·d₂/sqL = tanh·sqL (since sqL = √(d₂eδ₂))
    dd0_dδ1L = d2L;       dd0_dδ2L = eδ2L;      dd0_dδ3L = 0.0
    dd1_dδ1L = tanhL*sqL;  dd1_dδ2L = tanhL*sqL;  dd1_dδ3L = 2.0*sqL*(1.0-tanhL^2)
    dd2_dδ1L = d2L;       dd2_dδ2L = 0.0;        dd2_dδ3L = 0.0

    # a_Q[k,1] = median[k] - d_L[k]: ∂a_Q[1,1]/∂δᵢ = -∂d₀/∂δᵢ, ∂a_Q[2,1]/∂δᵢ = -∂d₁/∂δᵢ, etc.
    for i in 1:3
        dd = i==1 ? (dd0_dδ1L, dd1_dδ1L, dd2_dδ1L) :
             i==2 ? (dd0_dδ2L, dd1_dδ2L, dd2_dδ2L) :
                    (dd0_dδ3L, dd1_dδ3L, dd2_dδ3L)
        for k in 1:nk
            grad_v_out[nk+i] -= grad_unp[k] * dd[k]  # -∂d_L[k]/∂δ_i × ∂nll/∂a_Q[k,1]
        end
    end

    # -- Right gap: v[2nk+1:2nk+3] = (δ₄,δ₅,δ₆) → d_R → a_Q[:,3] = median + d_R
    δ₁R = v[2*nk+1]; δ₂R = v[2*nk+2]; δ₃R = v[2*nk+3]
    d2R = exp(δ₁R); eδ2R = exp(δ₂R); d0R = d2R + eδ2R
    sqR = sqrt(d2R * eδ2R); tanhR = tanh(δ₃R)

    dd0_dδ1R = d2R;       dd0_dδ2R = eδ2R;      dd0_dδ3R = 0.0
    dd1_dδ1R = tanhR*sqR;  dd1_dδ2R = tanhR*sqR;  dd1_dδ3R = 2.0*sqR*(1.0-tanhR^2)
    dd2_dδ1R = d2R;       dd2_dδ2R = 0.0;        dd2_dδ3R = 0.0

    for i in 1:3
        dd = i==1 ? (dd0_dδ1R, dd1_dδ1R, dd2_dδ1R) :
             i==2 ? (dd0_dδ2R, dd1_dδ2R, dd2_dδ2R) :
                    (dd0_dδ3R, dd1_dδ3R, dd2_dδ3R)
        for k in 1:nk
            grad_v_out[2*nk+i] += grad_unp[2*nk+k] * dd[k]  # +∂d_R[k]/∂δ_i × ∂nll/∂a_Q[k,3]
        end
    end

    # -- M_Q: v[3nk+1] = log(-M_Q) → M_Q = -exp(v), ∂M_Q/∂v = M_Q
    grad_v_out[3*nk+1] = grad_unp[nk*3+1] * M_Q

    # -- Init: v[3nk+2] = median, v[3nk+3] = log(gap_L), v[3nk+4] = log(gap_R)
    p = 3*nk + 1
    gap_L_init = a_init[2] - a_init[1]
    gap_R_init = a_init[3] - a_init[2]
    # a_init = [median-gap_L, median, median+gap_R]
    # ∂a_init[1]/∂v_median = 1, ∂a_init[2]/∂v_median = 1, ∂a_init[3]/∂v_median = 1
    grad_v_out[p+1] = grad_unp[nk*3+2] + grad_unp[nk*3+3] + grad_unp[nk*3+4]
    # ∂a_init[1]/∂v_logL = -gap_L, ∂a_init[2]/∂v_logL = 0, ∂a_init[3]/∂v_logL = 0
    grad_v_out[p+2] = -grad_unp[nk*3+2] * gap_L_init
    # ∂a_init[3]/∂v_logR = +gap_R
    grad_v_out[p+3] = grad_unp[nk*3+4] * gap_R_init

    # -- M_init: v[p+4] = log(-M_init) → ∂M_init/∂v = M_init
    grad_v_out[p+4] = grad_unp[nk*3+5] * M_init

    # -- a_eps1: v[p+5] = log(-a_eps1) → a_eps1 = -exp(v), ∂a_eps1/∂v = a_eps1
    grad_v_out[p+5] = grad_unp[nk*3+6] * a_eps1

    # -- a_eps3: v[p+6] = log(a_eps3) → ∂a_eps3/∂v = a_eps3
    grad_v_out[p+6] = grad_unp[nk*3+7] * a_eps3

    # -- M_eps: v[p+7] = log(-M_eps) → ∂M_eps/∂v = M_eps
    grad_v_out[p+7] = grad_unp[nk*3+8] * M_eps

    nll
end

# ================================================================
#  MLE ESTIMATION (LBFGS)
# ================================================================

function estimate_cspline_ml(y::Matrix{Float64}, K::Int, σy::Float64,
                              v0::Vector{Float64}, τ::Vector{Float64};
                              G::Int=201, maxiter::Int=50, verbose::Bool=true,
                              use_analytical_grad::Bool=true)
    ws = CSplineWorkspace(G, K)
    np = length(v0)

    function obj(v)
        a_Q, M_Q, a_init, M_init, a_eps1, a_eps3, M_eps = unpack_cspline(v, K)
        val = cspline_neg_loglik(a_Q, M_Q, a_init, M_init,
                                  a_eps1, a_eps3, M_eps, y, K, σy, τ, ws)
        isinf(val) ? 1e10 : val
    end

    function grad!(g, v)
        if use_analytical_grad
            val = cspline_neg_loglik_and_grad!(g, v, y, K, σy, τ, ws)
            isinf(val) && fill!(g, 0.0)
        else
            grad_h = 1e-3
            ws.vtmp .= v
            @inbounds for j in 1:np
                ws.vtmp[j] = v[j] + grad_h
                fp = obj(ws.vtmp)
                ws.vtmp[j] = v[j] - grad_h
                fm = obj(ws.vtmp)
                ws.vtmp[j] = v[j]
                g[j] = (fp - fm) / (2 * grad_h)
            end
        end
    end

    verbose && @printf("  CSpline ML initial obj: %.6f\n", obj(v0)); flush(stdout)

    res = optimize(obj, grad!, v0,
                   LBFGS(; linesearch=LineSearches.BackTracking()),
                   Optim.Options(iterations=maxiter, g_tol=1e-3,
                                 show_trace=verbose, show_every=10))
    v_opt = Optim.minimizer(res)
    @printf("  CSpline ML final obj: %.6f (iters=%d)\n",
            Optim.minimum(res), Optim.iterations(res)); flush(stdout)
    v_opt, Optim.minimum(res)
end

# ================================================================
#  TRUE PARAMETERS
# ================================================================

function make_true_cspline(; rho=0.8, sigma_v=0.5, sigma_eps=0.3, sigma_eta1=1.0, K=2,
                             d2_Q=0.005)
    τ = [0.25, 0.50, 0.75]
    par = make_true_params_linear(tau=τ, sigma_y=1.0, K=K,
                                   rho=rho, sigma_v=sigma_v, sigma_eps=sigma_eps,
                                   sigma_eta1=sigma_eta1)
    # Add quadratic heterogeneity to transition quantile gaps
    a_Q = copy(par.a_Q)
    a_Q[K+1, 1] -= d2_Q
    a_Q[K+1, 3] += d2_Q
    # Tail curvatures: M = -1/σ² from Gaussian approximation (symmetric: M₁=M₃=M)
    M_Q = -1.0/sigma_v^2
    M_init = -1.0/sigma_eta1^2
    M_eps = -1.0/sigma_eps^2
    (a_Q=a_Q, M_Q=M_Q,
     a_init=par.a_init, M_init=M_init,
     a_eps1=par.a_eps[1], a_eps3=par.a_eps[3], M_eps=M_eps)
end

using Optim, LineSearches

# ================================================================
#  FORWARD FILTER BACKWARD SAMPLER (FFBS)
#
#  Draw η₁,...,η_T from p(η|y) using the grid-based forward filter.
#  Forward pass: compute filtering distributions p(η_t | y_1,...,y_t)
#  Backward pass: sample η_T ~ p(η_T|y), then
#    η_{t-1} ~ p(η_{t-1}|η_t, y_1,...,y_{t-1}) ∝ T(η_t|η_{t-1}) p(η_{t-1}|y_1,...,y_{t-1})
# ================================================================

function cspline_ffbs!(eta_draw::Matrix{Float64},
                       a_Q::Matrix{Float64}, M_Q::Float64,
                       a_init::Vector{Float64}, M_init::Float64,
                       a_eps1::Float64, a_eps3::Float64, M_eps::Float64,
                       y::Matrix{Float64}, K::Int, σy::Float64, τ::Vector{Float64},
                       rng::AbstractRNG; G::Int=201)
    N, T_obs = size(y)
    G = isodd(G) ? G : G+1
    grid = collect(range(-8.0, 8.0, length=G))
    h_grid = (grid[end] - grid[1]) / (G-1)
    sw = zeros(G); sw[1]=1.0; sw[G]=1.0
    @inbounds for i in 2:G-1; sw[i] = iseven(i) ? 4.0 : 2.0; end
    sw .*= h_grid/3

    # Build transition matrix (C²)
    T_mat = zeros(G, G)
    hv_buf = zeros(K+1); t_buf = zeros(3)
    s_buf = zeros(3); masses_buf = zeros(4)
    c1buf = C1SolverBuffers()
    cspline_transition_matrix!(T_mat, grid, G, a_Q, M_Q, K, σy, τ, hv_buf, t_buf, s_buf, masses_buf, c1buf)

    # Init density (C²)
    βL_ref = Ref(0.0); βR_ref = Ref(0.0)
    κ1_ref = Ref(0.0); κ3_ref = Ref(0.0)
    a_init_s = copy(a_init)
    solve_cspline_c2!(s_buf, βL_ref, βR_ref, κ1_ref, κ3_ref, a_init_s, τ, M_init)
    β_L_init = βL_ref[]; β_R_init = βR_ref[]
    κ1_init = κ1_ref[]; κ3_init = κ3_ref[]
    log_ref = max(s_buf[1], s_buf[2], s_buf[3])
    cspline_masses!(masses_buf, a_init_s, s_buf, β_L_init, β_R_init, κ1_init, κ3_init, log_ref)
    C_init = sum(masses_buf)
    f_init = [exp(cspline_eval(grid[g], a_init_s, s_buf, β_L_init, β_R_init, κ1_init, κ3_init) - log_ref) / C_init for g in 1:G]

    # Eps density (C²)
    a_eps_s = [a_eps1, 0.0, a_eps3]
    solve_cspline_c2!(s_buf, βL_ref, βR_ref, κ1_ref, κ3_ref, a_eps_s, τ, M_eps)
    β_L_eps = βL_ref[]; β_R_eps = βR_ref[]
    κ1_eps = κ1_ref[]; κ3_eps = κ3_ref[]
    log_ref_eps = max(s_buf[1], s_buf[2], s_buf[3])
    cspline_masses!(masses_buf, a_eps_s, s_buf, β_L_eps, β_R_eps, κ1_eps, κ3_eps, log_ref_eps)
    C_eps = sum(masses_buf)

    # Store all filtering distributions: filter_p[g, i, t]
    filter_p = zeros(G, N, T_obs)
    p = zeros(G); p_new = zeros(G); pw = zeros(G)

    # Forward pass
    @inbounds for i in 1:N
        for g in 1:G
            f_e = exp(cspline_eval(y[i,1]-grid[g], a_eps_s, s_buf, β_L_eps, β_R_eps, κ1_eps, κ3_eps) - log_ref_eps) / C_eps
            p[g] = f_init[g] * f_e
        end
        L1 = dot(p, sw); p ./= L1
        filter_p[:, i, 1] .= p

        for t_step in 2:T_obs
            pw .= p .* sw
            mul!(p_new, transpose(T_mat), pw)
            for g in 1:G
                f_e = exp(cspline_eval(y[i,t_step]-grid[g], a_eps_s, s_buf, β_L_eps, β_R_eps, κ1_eps, κ3_eps) - log_ref_eps) / C_eps
                p_new[g] *= f_e
            end
            Lt = dot(p_new, sw); p_new ./= Lt
            filter_p[:, i, t_step] .= p_new
            p .= p_new
        end
    end

    # Backward sampling
    cdf = zeros(G)
    @inbounds for i in 1:N
        # Sample η_T from final filtering distribution
        for g in 1:G; cdf[g] = filter_p[g, i, T_obs] * sw[g]; end
        cumsum!(cdf, cdf); cdf ./= cdf[end]
        u = rand(rng)
        idx = searchsortedfirst(cdf, u)
        idx = clamp(idx, 1, G)
        eta_draw[i, T_obs] = grid[idx]

        # Sample backwards: η_{t-1} | η_t
        for t_step in (T_obs-1):-1:1
            η_next = eta_draw[i, t_step+1]
            # p(η_{t-1} | η_t, y_1,...,y_{t-1}) ∝ T(η_t | η_{t-1}) × filter_p(η_{t-1})
            # T(η_t | η_{t-1}) = T_mat[g_lag, g_next] where g_next is the grid index for η_t
            # Find nearest grid index for η_next
            g_next = clamp(round(Int, (η_next - grid[1]) / h_grid) + 1, 1, G)
            for g in 1:G
                p[g] = T_mat[g, g_next] * filter_p[g, i, t_step] * sw[g]
            end
            cumsum!(cdf, p); cdf ./= cdf[end]
            u = rand(rng)
            idx = searchsortedfirst(cdf, u)
            idx = clamp(idx, 1, G)
            eta_draw[i, t_step] = grid[idx]
        end
    end
    eta_draw
end

# ================================================================
#  QR M-STEP FOR CUBIC SPLINE MODEL
#
#  Given η draws from FFBS, estimate quantile knots by QR.
#  Returns: (a_Q, a_init, a_eps1, a_eps3)
#  Note: β_L, β_R are NOT estimated by QR (tail shape parameters).
# ================================================================

function cspline_qr_mstep(eta_draw::Matrix{Float64}, y::Matrix{Float64},
                           K::Int, σy::Float64, τ::Vector{Float64})
    N, T_obs = size(y)
    L = length(τ)

    # Transition: QR of η_t on H(η_{t-1}/σy) for t=2,...,T
    n_trans = N * (T_obs - 1)
    eta_t = zeros(n_trans)
    H_mat = zeros(n_trans, K+1)
    hv = zeros(K+1)
    idx = 0
    @inbounds for t_step in 2:T_obs, i in 1:N
        idx += 1
        eta_t[idx] = eta_draw[i, t_step]
        z = eta_draw[i, t_step-1] / σy
        hv[1]=1.0; K>=1 && (hv[2]=z)
        for k in 2:K; hv[k+1] = z*hv[k] - (k-1)*hv[k-1]; end
        H_mat[idx, :] .= hv
    end

    a_Q = zeros(K+1, L)
    for l in 1:L
        tau_l = τ[l]
        obj(a) = begin
            r = eta_t .- H_mat * a
            mean(r .* (tau_l .- (r .< 0)))
        end
        a0 = zeros(K+1)
        res = optimize(obj, a0, LBFGS(),
                       Optim.Options(iterations=200, g_tol=1e-8, show_trace=false))
        a_Q[:, l] .= Optim.minimizer(res)
    end

    # Transition tail rates from QR residuals (MLE of exponential tail)
    r1 = eta_t .- H_mat * a_Q[:, 1]   # residuals below q₁
    rL = eta_t .- H_mat * a_Q[:, L]   # residuals above q_L
    mask_lo = r1 .<= 0; mask_hi = rL .>= 0
    s_lo = sum(r1[mask_lo]); s_hi = sum(rL[mask_hi])
    β_L_Q = s_lo < -1e-10 ? -count(mask_lo) / s_lo : 2.0   # positive rate
    β_R_Q = s_hi >  1e-10 ?  count(mask_hi) / s_hi : 2.0  # positive rate

    # Initial η₁: sample quantiles + tail rates
    eta1 = eta_draw[:, 1]
    a_init = [quantile(eta1, τ[l]) for l in 1:L]
    below1 = eta1[eta1 .<= a_init[1]]
    above3 = eta1[eta1 .>= a_init[L]]
    s_init_lo = sum(below1 .- a_init[1])
    s_init_hi = sum(above3 .- a_init[L])
    β_L_init = s_init_lo < -1e-10 ? -length(below1) / s_init_lo : 2.0   # positive rate
    β_R_init = s_init_hi >  1e-10 ?  length(above3) / s_init_hi : 2.0  # positive rate

    # Epsilon: y - η, sample quantiles + tail rates
    eps_all = vec(y .- eta_draw)
    a_eps_raw = [quantile(eps_all, τ[l]) for l in 1:L]
    a_eps_raw .-= mean(a_eps_raw)  # center
    a_eps1 = a_eps_raw[1]; a_eps3 = a_eps_raw[3]
    below_eps = eps_all[eps_all .<= a_eps1]
    above_eps = eps_all[eps_all .>= a_eps3]
    s_eps_lo = sum(below_eps .- a_eps1)
    s_eps_hi = sum(above_eps .- a_eps3)
    β_L_eps = s_eps_lo < -1e-10 ? -length(below_eps) / s_eps_lo : 2.0   # positive rate
    β_R_eps = s_eps_hi >  1e-10 ?  length(above_eps) / s_eps_hi : 2.0  # positive rate

    (a_Q=a_Q, a_init=a_init, a_eps1=a_eps1, a_eps3=a_eps3,
     β_L_Q=β_L_Q, β_R_Q=β_R_Q,
     β_L_init=β_L_init, β_R_init=β_R_init,
     β_L_eps=β_L_eps, β_R_eps=β_R_eps)
end

# ================================================================
#  STOCHASTIC EM WITH QR M-STEP (CUBIC SPLINE)
#
#  Iterate: E-step (FFBS with C² density) → QR M-step
#  Curvature parameters (M_Q, M_init, M_eps) held fixed at true values.
# ================================================================

function estimate_cspline_qr(y::Matrix{Float64}, K::Int, σy::Float64,
                              a_Q0::Matrix{Float64}, M_Q::Float64,
                              a_init0::Vector{Float64}, M_init::Float64,
                              a_eps10::Float64, a_eps30::Float64, M_eps::Float64,
                              τ::Vector{Float64};
                              G::Int=201, S_em::Int=50, M_draws::Int=20,
                              verbose::Bool=true, seed::Int=1)
    N, T_obs = size(y)
    rng = MersenneTwister(seed)

    a_Q = copy(a_Q0)
    a_init = copy(a_init0)
    a_eps1 = a_eps10; a_eps3 = a_eps30

    eta_draw = zeros(N, T_obs)

    for iter in 1:S_em
        a_Q_sum = zeros(K+1, length(τ))
        a_init_sum = zeros(length(τ))
        ae1_sum = 0.0; ae3_sum = 0.0

        for m in 1:M_draws
            cspline_ffbs!(eta_draw, a_Q, M_Q, a_init, M_init,
                           a_eps1, a_eps3, M_eps,
                           y, K, σy, τ, rng; G=G)
            qr_est = cspline_qr_mstep(eta_draw, y, K, σy, τ)
            a_Q_sum .+= qr_est.a_Q
            a_init_sum .+= qr_est.a_init
            ae1_sum += qr_est.a_eps1
            ae3_sum += qr_est.a_eps3
        end

        a_Q .= a_Q_sum ./ M_draws
        a_init .= a_init_sum ./ M_draws
        a_eps1 = ae1_sum / M_draws
        a_eps3 = ae3_sum / M_draws

        if verbose && (iter <= 5 || iter % 10 == 0)
            @printf("  QR iter %3d: ρ=%.4f  a_init=[%.3f,%.3f,%.3f]  a_eps=[%.3f,%.3f]\n",
                    iter, a_Q[2,2], a_init..., a_eps1, a_eps3)
            flush(stdout)
        end
    end

    (a_Q=a_Q, a_init=a_init, a_eps1=a_eps1, a_eps3=a_eps3)
end

# ================================================================
#  TEST
# ================================================================

function mc_comparison(; S::Int=20, N::Int=300, G::Int=201,
                        ml_maxiter::Int=200, qr_S_em::Int=30, qr_M_draws::Int=10)
    K = 2; σy = 1.0; τ = [0.25, 0.50, 0.75]
    tp = make_true_cspline()

    println("="^70)
    @printf("  PAIRED MC: MLE vs QR  (S=%d, N=%d, G=%d)\n", S, N, G)
    println("="^70)
    @printf("True: ρ=%.4f  aQ23=%.4f  ae3=%.4f  M_Q=%.4f\n",
            tp.a_Q[2,2], tp.a_Q[3,2]-tp.a_Q[3,1], tp.a_eps3, tp.M_Q)
    flush(stdout)

    v_true = pack_cspline(tp.a_Q, tp.M_Q, tp.a_init, tp.M_init,
                           tp.a_eps1, tp.a_eps3, tp.M_eps)

    # Storage for key parameters: ρ (=a_Q[2,2]) and a_eps3
    ml_rho = zeros(S); qr_rho = zeros(S)
    ml_ae3 = zeros(S); qr_ae3 = zeros(S)
    ml_aQ23 = zeros(S); qr_aQ23 = zeros(S)
    ml_nll = zeros(S)
    ml_time = zeros(S); qr_time = zeros(S)

    ws = CSplineWorkspace(G, K)

    for s in 1:S
        # Generate data
        y, _ = generate_data_cspline(N, tp.a_Q, tp.M_Q,
                                      tp.a_init, tp.M_init,
                                      tp.a_eps1, tp.a_eps3, tp.M_eps,
                                      K, σy, τ; seed=s)

        # MLE (warm start from truth)
        ml_time[s] = @elapsed begin
            v_opt, nll = estimate_cspline_ml(y, K, σy, v_true, τ;
                                              G=G, maxiter=ml_maxiter, verbose=false)
        end
        a_Q_ml, _, a_init_ml, _, ae1_ml, ae3_ml_val, _ = unpack_cspline(v_opt, K)
        ml_rho[s] = a_Q_ml[2,2]
        ml_aQ23[s] = a_Q_ml[3,2] - a_Q_ml[3,1]
        ml_ae3[s] = ae3_ml_val
        ml_nll[s] = nll

        # QR (warm start from truth, curvatures fixed at truth)
        qr_time[s] = @elapsed begin
            qr_est = estimate_cspline_qr(y, K, σy, tp.a_Q, tp.M_Q,
                                           tp.a_init, tp.M_init,
                                           tp.a_eps1, tp.a_eps3, tp.M_eps, τ;
                                           G=G, S_em=qr_S_em, M_draws=qr_M_draws,
                                           verbose=false, seed=s)
        end
        qr_rho[s] = qr_est.a_Q[2,2]
        qr_aQ23[s] = qr_est.a_Q[3,2] - qr_est.a_Q[3,1]
        qr_ae3[s] = qr_est.a_eps3

        @printf("s=%2d: nll=%.4f  ML(ρ=%.4f ae3=%.4f t=%.0fs)  QR(ρ=%.4f ae3=%.4f t=%.0fs)\n",
                s, nll, ml_rho[s], ml_ae3[s], ml_time[s],
                qr_rho[s], qr_ae3[s], qr_time[s])
        flush(stdout)
    end

    # Summary statistics
    true_rho = tp.a_Q[2,2]
    true_aQ23 = tp.a_Q[3,2] - tp.a_Q[3,1]
    true_ae3 = tp.a_eps3

    println("\n", "="^70)
    println("  SUMMARY")
    println("="^70)

    function report(name, ml_vals, qr_vals, truth)
        ml_bias = mean(ml_vals) - truth
        qr_bias = mean(qr_vals) - truth
        ml_std = std(ml_vals)
        qr_std = std(qr_vals)
        ml_rmse = sqrt(ml_bias^2 + ml_std^2)
        qr_rmse = sqrt(qr_bias^2 + qr_std^2)
        @printf("%-8s true=%.4f\n", name, truth)
        @printf("  MLE:  mean=%.4f  bias=%+.4f  std=%.4f  RMSE=%.4f\n",
                mean(ml_vals), ml_bias, ml_std, ml_rmse)
        @printf("  QR:   mean=%.4f  bias=%+.4f  std=%.4f  RMSE=%.4f\n",
                mean(qr_vals), qr_bias, qr_std, qr_rmse)
        @printf("  efficiency (QR_RMSE/ML_RMSE): %.2f\n", qr_rmse / ml_rmse)
    end

    report("ρ", ml_rho, qr_rho, true_rho)
    println()
    report("aQ23", ml_aQ23, qr_aQ23, true_aQ23)
    println()
    report("ae3", ml_ae3, qr_ae3, true_ae3)
    println()
    @printf("Time:  MLE mean=%.1fs  QR mean=%.1fs\n", mean(ml_time), mean(qr_time))
    flush(stdout)
end

if abspath(PROGRAM_FILE) == @__FILE__
    mc_comparison(S=20, N=300)
end

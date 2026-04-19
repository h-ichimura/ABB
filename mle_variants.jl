#=
mle_variants.jl — Three MLE M-step implementations with non-crossing constraint

Approach 1a: Optim.jl IPNewton on full 9-param problem with linear constraints
Approach 1b: Coordinate descent with analytical feasible interval per parameter
Approach 2:  JuMP.jl + Ipopt

Constraint: (a_{l+1} - a_l)' h(η_j) ≥ 0 for all observations j and gaps l=1..L-1
This is a polyhedral constraint in the parameter vector vec(a_Q) ∈ ℝ^{(K+1)*L}.
=#

include("ABB_three_period.jl")
using Optim, LinearAlgebra, SparseArrays

# Lazy import for JuMP (heavy dependency)
const HAS_JUMP = try
    @eval using JuMP
    @eval using Ipopt
    true
catch e
    @warn "JuMP/Ipopt not available: $e"
    false
end

# ================================================================
#  Shared: profiled negative log-likelihood (no constraint handling)
# ================================================================

"""
Compute -avg_CDLL for given a_Q, with tail rates profiled out.
Assumes a_Q respects non-crossing at all observations.
Used by all three approaches.
"""
function compute_neg_profiled_ll(a_Q::Matrix{Float64},
                                  eta_t::Vector{Float64},
                                  H::Matrix{Float64},
                                  tau::Vector{Float64},
                                  b1_fallback::Float64, bL_fallback::Float64)
    n_obs = length(eta_t)
    L = size(a_Q, 2)
    Q_mat = H * a_Q  # n_obs × L

    # Tail rates: closed-form MLE
    r1 = eta_t .- view(Q_mat, :, 1)
    rL = eta_t .- view(Q_mat, :, L)
    ml = r1 .<= 0; mh = rL .>= 0
    s1 = sum(r1[ml]); sL = sum(rL[mh])
    b1v = s1 < -1e-10 ? -count(ml)/s1 : b1_fallback
    bLv = sL >  1e-10 ?  count(mh)/sL : bL_fallback

    # Log-likelihood
    ll = 0.0
    @inbounds for j in 1:n_obs
        ll += pw_logdens(eta_t[j], view(Q_mat, j, :), tau, b1v, bLv)
    end
    (-ll / n_obs, b1v, bLv)
end

"""Check non-crossing at all observations."""
function check_non_crossing(a_Q::Matrix{Float64}, H::Matrix{Float64})
    Q_mat = H * a_Q
    L = size(a_Q, 2)
    n_obs = size(H, 1)
    for j in 1:n_obs, l in 1:L-1
        Q_mat[j, l+1] < Q_mat[j, l] && return false
    end
    true
end

# ================================================================
#  APPROACH 1a: Optim.jl IPNewton (interior-point Newton)
# ================================================================

function m_step_mle_ipnewton!(par::Params, eta_all::Array{Float64,3},
                               y::Matrix{Float64}, cfg::Config;
                               verbose::Bool=false)
    K, L = cfg.K, cfg.L

    # Marginals (same as QR)
    eta1_all = stack_initial(eta_all, cfg)
    for l in 1:L; par.a_init[l] = quantile(eta1_all, cfg.tau[l]); end
    par.b1_init, par.bL_init = update_tails(eta1_all, par.a_init[1], par.a_init[L])

    eps_all = stack_eps(eta_all, y, cfg)
    for l in 1:L; par.a_eps[l] = quantile(eps_all, cfg.tau[l]); end
    par.a_eps .-= mean(par.a_eps)
    par.b1_eps, par.bL_eps = update_tails(eps_all, par.a_eps[1], par.a_eps[L])

    # Transition: IPNewton on full (K+1)*L parameter vector
    eta_t, eta_lag = stack_transition(eta_all, cfg)
    H = hermite_basis(eta_lag, K, cfg.sigma_y)
    n_obs = length(eta_t)
    P = K + 1  # params per column
    n_params = P * L

    # Warm start from QR
    m_step_qr!(par, eta_all, y, cfg)
    theta0 = vec(copy(par.a_Q))  # column-major: theta[(l-1)*P+k] = a_Q[k, l]

    # Objective
    function neg_ll(theta)
        a_Q = reshape(theta, P, L)
        # Penalty for crossing (IPNewton should avoid this via constraints,
        # but objective must be defined everywhere)
        Q_mat = H * a_Q
        for j in 1:n_obs, l in 1:L-1
            Q_mat[j, l+1] < Q_mat[j, l] && return 1e10
        end
        val, _, _ = compute_neg_profiled_ll(a_Q, eta_t, H, cfg.tau,
                                             par.b1_Q, par.bL_Q)
        val
    end

    # Linear inequality constraints: for each obs j, each gap l:
    #   Σ_k h[j,k] * (a_Q[k, l+1] - a_Q[k, l]) ≥ 0
    # Build constraint function c(θ) and return values in [0, Inf]
    n_con = n_obs * (L - 1)

    function con_c!(c, theta)
        a_Q = reshape(theta, P, L)
        idx = 0
        for j in 1:n_obs, l in 1:L-1
            idx += 1
            s = 0.0
            for k in 1:P
                s += H[j, k] * (a_Q[k, l+1] - a_Q[k, l])
            end
            c[idx] = s
        end
        c
    end

    # Jacobian of constraints (sparse, fixed): c_i = A_i' θ where A is (n_con × n_params)
    # Jacobian J[i, p] = A[i, p] — each constraint i depends on 2*P parameters
    function con_jac!(J, theta)
        fill!(J, 0.0)
        idx = 0
        for j in 1:n_obs, l in 1:L-1
            idx += 1
            for k in 1:P
                J[idx, (l-1)*P + k] = -H[j, k]
                J[idx, l*P + k] = H[j, k]
            end
        end
        J
    end

    # IPNewton requires TwiceDifferentiableConstraints
    lx = fill(-Inf, n_params); ux = fill(Inf, n_params)
    lc = zeros(n_con); uc = fill(Inf, n_con)

    # Constraint Hessian: each constraint is linear, so Hessian is zero
    function con_h!(h, theta, lambda)
        # All linear: Hessian of each constraint is 0, so add nothing
        h
    end

    # Use default (finite-difference) gradient/hessian
    df = TwiceDifferentiable(neg_ll, theta0)
    dfc = TwiceDifferentiableConstraints(con_c!, con_jac!, con_h!,
                                          lx, ux, lc, uc)

    res = optimize(df, dfc, theta0, IPNewton(),
                   Optim.Options(iterations=200, show_trace=verbose))

    par.a_Q .= reshape(Optim.minimizer(res), P, L)

    # Final tail rates from fitted a_Q
    Q_mat = H * par.a_Q
    r1 = eta_t .- view(Q_mat, :, 1); rL = eta_t .- view(Q_mat, :, L)
    ml = r1 .<= 0; mh = rL .>= 0
    s1 = sum(r1[ml]); sL = sum(rL[mh])
    s1 < -1e-10 && (par.b1_Q = -count(ml)/s1)
    sL >  1e-10 && (par.bL_Q =  count(mh)/sL)
    Optim.minimum(res)
end

# ================================================================
#  APPROACH 1b: Coordinate descent with analytical feasible intervals
# ================================================================

function m_step_mle_cdfeas!(par::Params, eta_all::Array{Float64,3},
                             y::Matrix{Float64}, cfg::Config;
                             verbose::Bool=false)
    K, L = cfg.K, cfg.L

    # Marginals
    eta1_all = stack_initial(eta_all, cfg)
    for l in 1:L; par.a_init[l] = quantile(eta1_all, cfg.tau[l]); end
    par.b1_init, par.bL_init = update_tails(eta1_all, par.a_init[1], par.a_init[L])

    eps_all = stack_eps(eta_all, y, cfg)
    for l in 1:L; par.a_eps[l] = quantile(eps_all, cfg.tau[l]); end
    par.a_eps .-= mean(par.a_eps)
    par.b1_eps, par.bL_eps = update_tails(eps_all, par.a_eps[1], par.a_eps[L])

    # Transition
    eta_t, eta_lag = stack_transition(eta_all, cfg)
    H = hermite_basis(eta_lag, K, cfg.sigma_y)
    n_obs = length(eta_t)

    # Warm start from QR
    m_step_qr!(par, eta_all, y, cfg)
    a_cur = copy(par.a_Q)

    """
    Compute feasible interval [lo, hi] for a_cur[k, l_target] such that
    non-crossing Q_{l+1}(η_j) ≥ Q_l(η_j) holds at all observations.
    Other entries of a_cur are held fixed.
    """
    function feasible_interval(k::Int, l_target::Int)
        lo = -Inf; hi = Inf
        cur = a_cur[k, l_target]

        # Gap l_target → l_target+1 (if l_target < L):
        # Σ_m h[j,m] * (a[m, l+1] - a[m, l]) ≥ 0
        # Changing a[k, l_target] from cur to v by Δ = v - cur:
        # Δ * (-h[j,k]) added → original_gap - h[j,k]*Δ ≥ 0 → h[j,k]*Δ ≤ original_gap
        if l_target < L
            for j in 1:n_obs
                g = 0.0
                @inbounds for m in 1:K+1
                    g += H[j, m] * (a_cur[m, l_target+1] - a_cur[m, l_target])
                end
                h_jk = H[j, k]
                if h_jk > 1e-14
                    hi = min(hi, cur + g / h_jk)
                elseif h_jk < -1e-14
                    lo = max(lo, cur + g / h_jk)
                else
                    g < -1e-14 && return (Inf, -Inf)  # infeasible (shouldn't happen)
                end
            end
        end

        # Gap l_target-1 → l_target (if l_target > 1):
        # Σ_m h[j,m] * (a[m, l] - a[m, l-1]) ≥ 0
        # Changing a[k, l_target] by Δ = v - cur: h[j,k]*Δ added
        # → original_gap + h[j,k]*Δ ≥ 0 → h[j,k]*Δ ≥ -original_gap
        if l_target > 1
            for j in 1:n_obs
                g = 0.0
                @inbounds for m in 1:K+1
                    g += H[j, m] * (a_cur[m, l_target] - a_cur[m, l_target-1])
                end
                h_jk = H[j, k]
                if h_jk > 1e-14
                    lo = max(lo, cur - g / h_jk)
                elseif h_jk < -1e-14
                    hi = min(hi, cur - g / h_jk)
                else
                    g < -1e-14 && return (Inf, -Inf)
                end
            end
        end

        (lo, hi)
    end

    function obj(a_Q_mat)
        val, _, _ = compute_neg_profiled_ll(a_Q_mat, eta_t, H, cfg.tau,
                                             par.b1_Q, par.bL_Q)
        val
    end

    # Coordinate descent
    n_cycles = 5
    for cycle in 1:n_cycles
        max_step = 0.0
        for l in 1:L, k in 1:K+1
            cur = a_cur[k, l]
            lo_feas, hi_feas = feasible_interval(k, l)
            # Add soft exploration bound (relative to current value)
            w = cycle <= 2 ? 0.5 : 0.15
            lo = max(lo_feas + 1e-8, cur - w)
            hi = min(hi_feas - 1e-8, cur + w)
            lo >= hi && continue  # no room to search

            function f1d(v)
                a_cur[k, l] = v
                obj(a_cur)
            end

            res1d = optimize(f1d, lo, hi)
            new_val = Optim.minimizer(res1d)
            max_step = max(max_step, abs(new_val - cur))
            a_cur[k, l] = new_val
        end
        verbose && @printf("  cycle %d: max step = %.6f\n", cycle, max_step)
        max_step < 1e-6 && break
    end

    par.a_Q .= a_cur

    # Final tail rates
    Q_mat = H * par.a_Q
    r1 = eta_t .- view(Q_mat, :, 1); rL = eta_t .- view(Q_mat, :, L)
    ml = r1 .<= 0; mh = rL .>= 0
    s1 = sum(r1[ml]); sL = sum(rL[mh])
    s1 < -1e-10 && (par.b1_Q = -count(ml)/s1)
    sL >  1e-10 && (par.bL_Q =  count(mh)/sL)
    obj(par.a_Q)
end

# ================================================================
#  APPROACH 2: JuMP.jl + Ipopt (loaded lazily since heavy dependencies)
# ================================================================

function m_step_mle_jump!(par::Params, eta_all::Array{Float64,3},
                           y::Matrix{Float64}, cfg::Config;
                           verbose::Bool=false)
    HAS_JUMP || error("JuMP/Ipopt not installed. Run Pkg.add([\"JuMP\", \"Ipopt\"])")

    K, L = cfg.K, cfg.L
    P = K + 1

    # Marginals (same)
    eta1_all = stack_initial(eta_all, cfg)
    for l in 1:L; par.a_init[l] = quantile(eta1_all, cfg.tau[l]); end
    par.b1_init, par.bL_init = update_tails(eta1_all, par.a_init[1], par.a_init[L])

    eps_all = stack_eps(eta_all, y, cfg)
    for l in 1:L; par.a_eps[l] = quantile(eps_all, cfg.tau[l]); end
    par.a_eps .-= mean(par.a_eps)
    par.b1_eps, par.bL_eps = update_tails(eps_all, par.a_eps[1], par.a_eps[L])

    # Transition
    eta_t, eta_lag = stack_transition(eta_all, cfg)
    H = hermite_basis(eta_lag, K, cfg.sigma_y)
    n_obs = length(eta_t)

    # Warm start from QR
    m_step_qr!(par, eta_all, y, cfg)
    a_init = copy(par.a_Q)

    # Profile tail rates at warm start and FIX them (for JuMP's differentiability)
    Q_mat0 = H * a_init
    r1_0 = eta_t .- view(Q_mat0, :, 1); rL_0 = eta_t .- view(Q_mat0, :, L)
    ml_0 = r1_0 .<= 0; mh_0 = rL_0 .>= 0
    s1_0 = sum(r1_0[ml_0]); sL_0 = sum(rL_0[mh_0])
    b1_0 = s1_0 < -1e-10 ? -count(ml_0)/s1_0 : par.b1_Q
    bL_0 = sL_0 >  1e-10 ?  count(mh_0)/sL_0 : par.bL_Q

    # Fix segment assignment based on QR warm start (for smooth objective)
    # segment[j] ∈ {0,1,...,L} where 0=left tail, L=right tail, l=between Q_l and Q_{l+1}
    seg = zeros(Int, n_obs)
    for j in 1:n_obs
        if eta_t[j] <= Q_mat0[j, 1]
            seg[j] = 0
        elseif eta_t[j] > Q_mat0[j, L]
            seg[j] = L
        else
            for l in 1:L-1
                if eta_t[j] <= Q_mat0[j, l+1]
                    seg[j] = l
                    break
                end
            end
        end
    end

    # Build JuMP model
    m = Model(Ipopt.Optimizer)
    verbose || set_silent(m)

    # Box constraints around warm start prevent unbounded optimization
    # (segments are fixed at warm start values; with unbounded variables,
    # the fixed-segment objective is unbounded)
    @variable(m, a_init[k,l] - 0.5 <= a[k=1:P, l=1:L] <= a_init[k,l] + 0.5)
    for k in 1:P, l in 1:L; set_start_value(a[k, l], a_init[k, l]); end

    # Non-crossing constraints: Σ_k H[j,k] * (a[k, l+1] - a[k, l]) ≥ 0
    @constraint(m, [j=1:n_obs, l=1:L-1],
                sum(H[j, k] * (a[k, l+1] - a[k, l]) for k in 1:P) >= 0)

    tau = cfg.tau
    idx_left = findall(seg .== 0)
    idx_right = findall(seg .== L)
    idx_mid = [findall(seg .== l) for l in 1:L-1]

    # Q values are LINEAR in a — use @expression (not @NLexpression)
    # Q[j,l] = Σ_k H[j,k] * a[k,l]
    # Linear contribution to objective from left/right tails:
    #   Left:  b1_0 * Σ_j Q[j,1]  (constants dropped)
    #   Right: -bL_0 * Σ_j Q[j,L]
    # Both are linear in a; combine into a single linear objective term.

    # Compute total contribution coefficient for each a[k,l]
    # Left tail contribution: +b1_0 · H[j,1] on a[k,1]
    # Right tail: -bL_0 · H[j,L-index] on a[k,L]
    coef_lin = zeros(P, L)  # coefficient of a[k,l] in linear part
    for j in idx_left, k in 1:P
        coef_lin[k, 1] += b1_0 * H[j, k]
    end
    for j in idx_right, k in 1:P
        coef_lin[k, L] -= bL_0 * H[j, k]
    end

    @expression(m, lin_term, sum(coef_lin[k,l] * a[k,l] for k in 1:P, l in 1:L))

    # Nonlinear part: for each middle observation j in segment l,
    # add log(Q[j,l+1] - Q[j,l]) = log(Σ_k H[j,k]*(a[k,l+1]-a[k,l]))
    # Use @NLobjective with explicit sum
    @NLobjective(m, Min,
        (
          lin_term
          + sum(
              sum(log(sum(H[j, k] * (a[k, l+1] - a[k, l]) for k in 1:P)) for j in idx_mid[l])
              for l in 1:L-1
            )
        ) / n_obs
    )

    optimize!(m)

    par.a_Q .= value.(a)

    # Final tail rates
    Q_mat = H * par.a_Q
    r1 = eta_t .- view(Q_mat, :, 1); rL = eta_t .- view(Q_mat, :, L)
    ml = r1 .<= 0; mh = rL .>= 0
    s1 = sum(r1[ml]); sL = sum(rL[mh])
    s1 < -1e-10 && (par.b1_Q = -count(ml)/s1)
    sL >  1e-10 && (par.bL_Q =  count(mh)/sL)
    objective_value(m)
end

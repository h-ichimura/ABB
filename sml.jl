#=
sml.jl — Simulated Maximum Likelihood for the ABB three-period model

Instead of EM, use SML:
1. For a candidate θ, simulate M sequences (η_1, η_2, η_3) from the model
   using fixed uniform random numbers U^(m) (common random numbers).
2. The simulated likelihood for observation i is:
     L_i(θ) ≈ (1/M) Σ_m Π_t f_ε(y_{it} - η_{it}^(m)(θ, U^(m)))
3. Maximize Σ_i log L_i(θ) over θ.

Feasibility: if any simulated η pair gives crossing knots, reject θ
(return -∞).

Common random numbers: the uniform draws U^(m) = (u_1^(m), u_2^(m), u_3^(m))
are fixed across θ evaluations. Only the mapping η(θ, U) changes with θ.
This makes the simulated likelihood a smooth function of θ.
=#

include("ABB_three_period.jl")
using Optim, Random, Printf

# ================================================================
#  FORWARD SIMULATION OF η GIVEN θ AND COMMON RANDOM NUMBERS
#
# For the ABB piecewise-uniform model:
# η_{i1} ~ inverse CDF of F_init at u_{i1}
# η_{it} = inverse CDF of F(· | η_{i,t-1}) at u_{it}
# ================================================================

"""
Inverse CDF of the piecewise-uniform + exponential-tail distribution.
Given quantile knots q[1]<q[2]<q[3] at τ=(0.25, 0.50, 0.75) and tail
rates b1, bL, returns the value x such that P(X ≤ x) = u.
"""
function pw_inverse_cdf(u::Float64, q::Vector{Float64}, tau::Vector{Float64},
                        b1::Float64, bL::Float64)
    L = length(q)
    if u < tau[1]
        # Left exponential tail: P(X ≤ x) = τ_1 exp(b1(x - q_1)) for x ≤ q_1
        # → x = q_1 + log(u/τ_1) / b1
        return q[1] + log(u / tau[1]) / b1
    end
    if u >= tau[L]
        # Right exponential tail: P(X > x) = (1-τ_L) exp(-bL(x - q_L))
        # → x = q_L + log((1-τ_L)/(1-u)) / bL
        return q[L] + log((1.0 - tau[L]) / (1.0 - u)) / bL
    end
    # Interior: piecewise linear in u between τ_l and τ_{l+1}
    @inbounds for l in 1:L-1
        if u < tau[l+1]
            frac = (u - tau[l]) / (tau[l+1] - tau[l])
            return q[l] + frac * (q[l+1] - q[l])
        end
    end
    q[L]
end

"""
Given θ (Params) and a matrix of common uniform draws U of size N×T×M×2,
simulate the η and ε sequences for each observation i, draw m.
U[i,t,m,1] for η draw, U[i,t,m,2] for ε draw.

Returns:
  eta_sim : N × T × M array of η values
  eps_sim : N × T × M array of ε values
  feasible : Bool, true if all knot configurations were ordered
"""
function forward_simulate(par::Params, cfg::Config, U::Array{Float64,4})
    N, T, M = cfg.N, cfg.T, cfg.M
    eta_sim = zeros(N, T, M)
    eps_sim = zeros(N, T, M)
    q_buf = zeros(cfg.L)

    feasible = true

    for m in 1:M
        for i in 1:N
            # η_1 from initial distribution
            eta_sim[i, 1, m] = pw_inverse_cdf(U[i, 1, m, 1], par.a_init,
                                                cfg.tau, par.b1_init, par.bL_init)
            # η_t from transition for t = 2, 3
            for t in 2:T
                transition_quantiles!(q_buf, eta_sim[i, t-1, m],
                                       par.a_Q, cfg.K, cfg.sigma_y)
                # Check feasibility (non-crossing)
                for l in 1:cfg.L-1
                    if q_buf[l+1] < q_buf[l]
                        feasible = false
                    end
                end
                eta_sim[i, t, m] = pw_inverse_cdf(U[i, t, m, 1], q_buf,
                                                    cfg.tau, par.b1_Q, par.bL_Q)
            end
            # ε_t (iid)
            for t in 1:T
                eps_sim[i, t, m] = pw_inverse_cdf(U[i, t, m, 2], par.a_eps,
                                                    cfg.tau, par.b1_eps, par.bL_eps)
            end
        end
    end

    eta_sim, eps_sim, feasible
end

# ================================================================
#  SIMULATED LIKELIHOOD
#
# Key insight: for SML, we need P(y_i | θ) = E[∏_t f_ε(y_{it} - η_{it})]
# where expectation is over η_i drawn from the model.
# The η_i draws do NOT depend on y_i.
#
# Using common random numbers U, simulate η_i = g(θ, U_i). Then:
#   L_i(θ) ≈ (1/M) Σ_m ∏_t f_ε(y_{it} - g(θ, U_i^(m))_t)
#
# For the LOG-likelihood we use the log-sum-exp trick for numerical stability.
# ================================================================

function simulated_neg_loglik(par::Params, y::Matrix{Float64}, cfg::Config,
                              U::Array{Float64,4})
    N, T, M = cfg.N, cfg.T, cfg.M
    eta_sim, _, feasible = forward_simulate(par, cfg, U)
    # Non-crossing violation → large but finite penalty so NM can still move
    if !feasible
        return 1e6
    end

    # Compute log-likelihood for each (i, m) as sum of log f_ε(y_{it} - η_{it}^m)
    # Then marginalize over m by log-sum-exp / M.
    ll_total = 0.0
    log_m_terms = zeros(M)

    for i in 1:N
        for m in 1:M
            ll_m = 0.0
            for t in 1:T
                eps_val = y[i, t] - eta_sim[i, t, m]
                ll_m += pw_logdens(eps_val, par.a_eps, cfg.tau,
                                    par.b1_eps, par.bL_eps)
            end
            log_m_terms[m] = ll_m
        end
        # log L_i = log((1/M) Σ_m exp(log_m_terms[m]))
        max_log = maximum(log_m_terms)
        ll_i = max_log + log(sum(exp(log_m_terms[m] - max_log) for m in 1:M) / M)
        ll_total += ll_i
    end

    -ll_total / N
end

# ================================================================
#  MAXIMIZATION OVER θ
#
# For SML, parameterize θ as the flattened parameter vector including
# all quantile coefficients, tail rates, and marginal parameters.
# Use LBFGS on the full θ.
# ================================================================

"""
Pack Params into a vector (for optimization).
Layout: a_Q (K+1 × L), b1_Q, bL_Q, a_init (L), b1_init, bL_init, a_eps (L), b1_eps, bL_eps.
Tail rates stored as log (to keep positive).
"""
function params_to_vec(par::Params)
    v = Float64[]
    append!(v, vec(par.a_Q))
    push!(v, log(par.b1_Q), log(par.bL_Q))
    append!(v, par.a_init)
    push!(v, log(par.b1_init), log(par.bL_init))
    append!(v, par.a_eps)
    push!(v, log(par.b1_eps), log(par.bL_eps))
    v
end

function vec_to_params!(par::Params, v::Vector{Float64}, K::Int, L::Int)
    np = (K + 1) * L
    par.a_Q .= reshape(view(v, 1:np), K + 1, L)
    par.b1_Q = exp(v[np + 1])
    par.bL_Q = exp(v[np + 2])
    par.a_init .= view(v, np + 3:np + 2 + L)
    par.b1_init = exp(v[np + 3 + L])
    par.bL_init = exp(v[np + 4 + L])
    par.a_eps .= view(v, np + 5 + L:np + 4 + 2L)
    par.b1_eps = exp(v[np + 5 + 2L])
    par.bL_eps = exp(v[np + 6 + 2L])
    par
end

"""
For parameter v[i] (a_Q entry), compute the feasible interval: the range
of values such that all simulated η_{t-1}^(m) still give ordered quantile
knots q_1 < q_2 < q_3.

For each simulated η_{t-1}^(m) and each adjacent pair (l, l+1):
  constraint: Σ_k H_j[k] * (a[k,l+1] - a[k,l]) > 0
where H_j = hermite_basis(η_{t-1}^(m)).

When varying a[k, l_target], the gap changes linearly in v. We solve
for the range of v that keeps all gaps positive.

Returns (lo, hi) for parameter index i corresponding to a_Q[k, l_target].
Non-aQ parameters (tail rates, marginals) have no crossing constraint.
"""
function feasible_interval_sml(v::Vector{Float64}, i::Int,
                                eta_lag_all::Vector{Float64},
                                K::Int, L::Int, sigma_y::Float64)
    np_aQ = (K + 1) * L
    # Non-aQ parameters: no crossing constraint (wide search)
    if i > np_aQ
        if i == np_aQ + 1 || i == np_aQ + 2 ||
           (i >= np_aQ + 3 + L && i <= np_aQ + 4 + L) ||
           (i >= np_aQ + 5 + 2L && i <= np_aQ + 6 + 2L)
            return (v[i] - 0.5, v[i] + 0.5)   # log tail rates
        else
            return (v[i] - 0.3, v[i] + 0.3)   # marginal quantile levels
        end
    end

    # aQ parameter: find (k, l_target) from i (Julia 1-indexed column major)
    k = ((i - 1) % (K + 1)) + 1
    l_target = ((i - 1) ÷ (K + 1)) + 1

    cur = v[i]
    lo = cur - 1.0
    hi = cur + 1.0

    a_Q = reshape(v[1:np_aQ], K + 1, L)
    n_obs = length(eta_lag_all)

    # For each η_lag simulation, compute h(η_lag) and check constraints
    hv = zeros(K + 1)
    for j in 1:n_obs
        z = eta_lag_all[j] / sigma_y
        hv[1] = 1.0
        K >= 1 && (hv[2] = z)
        for kk in 2:K
            hv[kk+1] = z * hv[kk] - (kk - 1) * hv[kk-1]
        end

        # Check adjacent gaps around l_target
        # Gap (l_target, l_target+1) = h' * (a[:,l+1] - a[:,l]) must be > 0
        # Changing a[k, l_target] by Δ changes this gap by -h[k]*Δ
        if l_target < L
            g = 0.0
            for m in 1:K+1
                g += hv[m] * (a_Q[m, l_target+1] - a_Q[m, l_target])
            end
            h_k = hv[k]
            if h_k > 1e-14
                # g - h_k*Δ > 0 → Δ < g/h_k → hi ≤ cur + g/h_k
                hi = min(hi, cur + g / h_k - 1e-8)
            elseif h_k < -1e-14
                lo = max(lo, cur + g / h_k + 1e-8)
            end
        end
        # Gap (l_target-1, l_target) = h' * (a[:,l_target] - a[:,l_target-1]) must be > 0
        if l_target > 1
            g = 0.0
            for m in 1:K+1
                g += hv[m] * (a_Q[m, l_target] - a_Q[m, l_target-1])
            end
            h_k = hv[k]
            if h_k > 1e-14
                # g + h_k*Δ > 0 → Δ > -g/h_k
                lo = max(lo, cur - g / h_k + 1e-8)
            elseif h_k < -1e-14
                hi = min(hi, cur - g / h_k - 1e-8)
            end
        end
    end

    (lo, hi)
end

function estimate_sml(y::Matrix{Float64}, cfg::Config, par0::Params;
                      M_sim::Int=500, seed::Int=42, verbose::Bool=true,
                      maxiter::Int=30)
    N, T, L, K = cfg.N, cfg.T, cfg.L, cfg.K
    cfg_sim = Config(N, T, K, L, cfg.tau, cfg.sigma_y,
                     cfg.maxiter, cfg.n_draws, M_sim, cfg.var_prop)

    # Common random numbers
    rng = MersenneTwister(seed)
    U = rand(rng, N, T, M_sim, 2)

    par_work = copy_params(par0)
    v = params_to_vec(par0)

    verbose && @printf("  SML: initial obj = %.6f\n",
                       simulated_neg_loglik(par_work, y, cfg_sim, U))

    total_params = length(v)

    for cyc in 1:maxiter
        max_step = 0.0
        # Pre-simulate η_lag at current θ to compute feasible intervals
        # (only η_1 and η_2 serve as lag variables)
        eta_sim, _, _ = forward_simulate(par_work, cfg_sim, U)
        eta_lag_all = vec(eta_sim[:, 1:T-1, :])  # N * (T-1) * M

        for i in 1:total_params
            cur = v[i]
            lo, hi = feasible_interval_sml(v, i, eta_lag_all, K, L, cfg.sigma_y)
            if hi - lo < 1e-6
                continue  # no feasible move
            end

            function f1d(vi)
                v[i] = vi
                vec_to_params!(par_work, v, K, L)
                simulated_neg_loglik(par_work, y, cfg_sim, U)
            end

            res1d = optimize(f1d, lo, hi)
            new_val = Optim.minimizer(res1d)
            max_step = max(max_step, abs(new_val - cur))
            v[i] = new_val
        end
        vec_to_params!(par_work, v, K, L)

        if verbose
            @printf("  SML cycle %3d: obj = %.6f  max_step = %.2e\n",
                    cyc, simulated_neg_loglik(par_work, y, cfg_sim, U), max_step)
        end
        max_step < 1e-5 && break
    end

    vec_to_params!(par_work, v, K, L)
    @printf("  SML: final obj = %.6f\n",
            simulated_neg_loglik(par_work, y, cfg_sim, U))
    par_work
end

# ================================================================
#  TEST
# ================================================================

function test_sml()
    K = 2; L = 3; sigma_y = 1.0; T = 3; N = 200; M_sim = 500
    tau = [0.25, 0.50, 0.75]
    par_true = make_true_params_linear(tau=tau, sigma_y=sigma_y, K=K)
    y, eta_true = generate_data_abb(N, par_true, tau, sigma_y, K; seed=42)

    cfg = Config(N, T, K, L, tau, sigma_y, 1, 1, M_sim, fill(0.05, T))

    # Warm start from QR to get a feasible point
    par0 = init_params(y, cfg)
    eta_all = zeros(N, T, 20)
    for m in 1:20; eta_all[:, :, m] .= 0.6 .* y; end
    cfg_em = Config(N, T, K, L, tau, sigma_y, 5, 100, 20, fill(0.05, T))
    # A few EM-QR iterations to get a feasible starting point
    for iter in 1:5
        e_step!(eta_all, y, par0, cfg_em)
        m_step_qr!(par0, eta_all, y, cfg_em)
    end

    println("="^60)
    println("  SML TEST (N=$N, M_sim=$M_sim)")
    println("="^60)
    println("\nTrue parameters:")
    @printf("  slopes:     [%.4f, %.4f, %.4f]\n", par_true.a_Q[2,:]...)
    @printf("  intercepts: [%.4f, %.4f, %.4f]\n", par_true.a_Q[1,:]...)

    @printf("\nStarting parameters:\n")
    @printf("  slopes:     [%.4f, %.4f, %.4f]\n", par0.a_Q[2,:]...)
    @printf("  intercepts: [%.4f, %.4f, %.4f]\n", par0.a_Q[1,:]...)

    t = @elapsed par_sml = estimate_sml(y, cfg, par0;
                                        M_sim=M_sim, maxiter=500, verbose=false)

    @printf("\nSML estimates (%.1f s):\n", t)
    @printf("  slopes:     [%.4f, %.4f, %.4f]\n", par_sml.a_Q[2,:]...)
    @printf("  intercepts: [%.4f, %.4f, %.4f]\n", par_sml.a_Q[1,:]...)
    @printf("  b1_Q=%.4f  bL_Q=%.4f\n", par_sml.b1_Q, par_sml.bL_Q)
    @printf("  ε quants:   [%.4f, %.4f, %.4f]\n", par_sml.a_eps...)
end

if abspath(PROGRAM_FILE) == @__FILE__
    test_sml()
end

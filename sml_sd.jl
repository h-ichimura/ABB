#=
sml_sd.jl — Simulated Maximum Likelihood with Steepest Descent

Uses:
  - SML objective with common random numbers (forward simulation + eps density)
  - Steepest descent with backtracking Armijo line search
  - Numerical gradient by central differences

No LBFGS (Hessian-free), no coordinate descent — pure gradient descent.
=#

include("ABB_three_period.jl")
using Printf, Random, LinearAlgebra

# ================================================================
#  INVERSE CDF FOR PIECEWISE-UNIFORM + EXP-TAIL DISTRIBUTION
# ================================================================

function pw_inverse_cdf(u::Float64, q::AbstractVector{Float64},
                        tau::Vector{Float64}, b1::Float64, bL::Float64)
    L = length(q)
    if u < tau[1]
        return q[1] + log(u / tau[1]) / b1
    end
    if u >= tau[L]
        return q[L] + log((1.0 - tau[L]) / (1.0 - u)) / bL
    end
    @inbounds for l in 1:L-1
        if u < tau[l+1]
            frac = (u - tau[l]) / (tau[l+1] - tau[l])
            return q[l] + frac * (q[l+1] - q[l])
        end
    end
    q[L]
end

# ================================================================
#  SIMULATED LIKELIHOOD
# ================================================================

"""
Compute simulated neg-log-likelihood given params and common uniform draws U.
U[i, t, m] = uniform draw for η_t of household i, simulation m.
For T=3, we simulate η_1, η_2, η_3 via inverse-CDF.
"""
function simulated_neg_loglik(par::Params, y::Matrix{Float64}, cfg::Config,
                              U::Array{Float64,3})
    N, T, M = cfg.N, cfg.T, cfg.M
    q_buf = zeros(cfg.L)
    log_weights = zeros(M)

    total_ll = 0.0
    for i in 1:N
        for m in 1:M
            # Simulate η_1
            eta_1 = pw_inverse_cdf(U[i, 1, m], par.a_init, cfg.tau,
                                    par.b1_init, par.bL_init)
            # Simulate η_2 | η_1
            transition_quantiles!(q_buf, eta_1, par.a_Q, cfg.K, cfg.sigma_y)
            sort!(q_buf)
            eta_2 = pw_inverse_cdf(U[i, 2, m], q_buf, cfg.tau,
                                    par.b1_Q, par.bL_Q)
            # Simulate η_3 | η_2
            transition_quantiles!(q_buf, eta_2, par.a_Q, cfg.K, cfg.sigma_y)
            sort!(q_buf)
            eta_3 = pw_inverse_cdf(U[i, 3, m], q_buf, cfg.tau,
                                    par.b1_Q, par.bL_Q)

            # log f(y | η) = sum of log f_eps(y_t - η_t)
            log_weights[m] = (
                pw_logdens(y[i,1] - eta_1, par.a_eps, cfg.tau,
                           par.b1_eps, par.bL_eps) +
                pw_logdens(y[i,2] - eta_2, par.a_eps, cfg.tau,
                           par.b1_eps, par.bL_eps) +
                pw_logdens(y[i,3] - eta_3, par.a_eps, cfg.tau,
                           par.b1_eps, par.bL_eps))
        end
        # log L_i = log((1/M) Σ exp(log_weights)) = max + log(sum exp(w-max)/M)
        mx = maximum(log_weights)
        Li = mx + log(sum(exp(log_weights[m] - mx) for m in 1:M) / M)
        total_ll += Li
    end
    -total_ll / N
end

# ================================================================
#  PARAMETER VEC <-> PARAMS
# ================================================================

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

# ================================================================
#  NUMERICAL GRADIENT (central differences)
# ================================================================

function num_gradient(v::Vector{Float64}, obj_fn, h::Float64=1e-4)
    g = zeros(length(v))
    for i in eachindex(v)
        v_p = copy(v); v_p[i] += h
        v_m = copy(v); v_m[i] -= h
        g[i] = (obj_fn(v_p) - obj_fn(v_m)) / (2h)
    end
    g
end

# ================================================================
#  STEEPEST DESCENT WITH BACKTRACKING
# ================================================================

function steepest_descent(obj_fn, v0::Vector{Float64};
                         maxiter::Int=100, tol::Float64=1e-5,
                         alpha_init::Float64=0.1, verbose::Bool=true)
    v = copy(v0)
    obj_val = obj_fn(v)
    alpha = alpha_init

    for iter in 1:maxiter
        # Compute gradient
        g = num_gradient(v, obj_fn)
        gnorm = norm(g)
        if gnorm < tol
            verbose && @printf("  Converged (|g|=%.2e) at iter %d\n", gnorm, iter)
            break
        end

        # Backtracking line search (Armijo with c=1e-4)
        direction = -g / gnorm  # unit direction
        step = alpha
        obj_new = obj_fn(v .+ step .* direction)
        n_back = 0
        while obj_new > obj_val - 1e-4 * step * gnorm && n_back < 20
            step *= 0.5
            n_back += 1
            obj_new = obj_fn(v .+ step .* direction)
        end

        if n_back >= 20 && obj_new >= obj_val
            verbose && @printf("  Line search failed at iter %d (|g|=%.2e, obj=%.6f)\n",
                               iter, gnorm, obj_val)
            break
        end

        v .+= step .* direction
        obj_val = obj_new
        # Slight step-size adaptation
        alpha = min(step * 1.5, 0.5)

        if verbose && (iter <= 5 || iter % 5 == 0)
            @printf("  SD iter %3d: obj=%.6f  |g|=%.4e  step=%.4f  back=%d\n",
                    iter, obj_val, gnorm, step, n_back)
        end
    end
    v, obj_val
end

# ================================================================
#  ESTIMATION
# ================================================================

function estimate_sml_sd(y::Matrix{Float64}, cfg::Config, par0::Params;
                         M_sim::Int=500, seed::Int=42, verbose::Bool=true,
                         maxiter::Int=100)
    N, T = cfg.N, cfg.T
    L, K = cfg.L, cfg.K
    cfg_sim = Config(N, T, K, L, cfg.tau, cfg.sigma_y,
                     cfg.maxiter, cfg.n_draws, M_sim, cfg.var_prop)

    # Common random numbers
    rng = MersenneTwister(seed)
    U = rand(rng, N, T, M_sim)

    par_work = copy_params(par0)
    function obj(v)
        vec_to_params!(par_work, v, K, L)
        simulated_neg_loglik(par_work, y, cfg_sim, U)
    end

    v0 = params_to_vec(par0)
    verbose && @printf("  SML-SD: initial obj = %.6f\n", obj(v0))

    v_opt, obj_opt = steepest_descent(obj, v0; maxiter=maxiter, verbose=verbose)
    vec_to_params!(par_work, v_opt, K, L)
    @printf("  SML-SD: final obj = %.6f\n", obj_opt)
    par_work
end

# ================================================================
#  TEST
# ================================================================

function main()
    K = 2; L = 3; sigma_y = 1.0; T = 3; N = 200; M_sim = 500
    tau = [0.25, 0.50, 0.75]
    par_true = make_true_params_linear(tau=tau, sigma_y=sigma_y, K=K)
    y, eta_true = generate_data_abb(N, par_true, tau, sigma_y, K; seed=42)
    cfg = Config(N, T, K, L, tau, sigma_y, 1, 1, 1, fill(0.05, T))

    # QR warm start
    println("Getting QR warm start...")
    par0 = init_params(y, cfg)
    eta_all = zeros(N, T, 20)
    for m in 1:20; eta_all[:,:,m] .= 0.6 .* y; end
    cfg_em = Config(N, T, K, L, tau, sigma_y, 5, 100, 20, fill(0.05, T))
    for _ in 1:5
        e_step!(eta_all, y, par0, cfg_em)
        m_step_qr!(par0, eta_all, y, cfg_em)
    end

    println("\nRunning SML with Steepest Descent...")
    t = @elapsed par_sml = estimate_sml_sd(y, cfg, par0;
                                             M_sim=M_sim, maxiter=100)

    println("\n==== TRANSITION ====")
    @printf("True slopes:      [%.4f, %.4f, %.4f]\n", par_true.a_Q[2,:]...)
    @printf("QR slopes:        [%.4f, %.4f, %.4f]\n", par0.a_Q[2,:]...)
    @printf("SML-SD slopes:    [%.4f, %.4f, %.4f]\n", par_sml.a_Q[2,:]...)
    @printf("True intercepts:  [%.4f, %.4f, %.4f]\n", par_true.a_Q[1,:]...)
    @printf("QR intercepts:    [%.4f, %.4f, %.4f]\n", par0.a_Q[1,:]...)
    @printf("SML-SD intcpts:   [%.4f, %.4f, %.4f]\n", par_sml.a_Q[1,:]...)

    println("\n==== ε QUANTILES ====")
    @printf("True:    [%.4f, %.4f, %.4f]  b1=%.4f  bL=%.4f\n",
            par_true.a_eps..., par_true.b1_eps, par_true.bL_eps)
    @printf("QR:      [%.4f, %.4f, %.4f]  b1=%.4f  bL=%.4f\n",
            par0.a_eps..., par0.b1_eps, par0.bL_eps)
    @printf("SML-SD:  [%.4f, %.4f, %.4f]  b1=%.4f  bL=%.4f\n",
            par_sml.a_eps..., par_sml.b1_eps, par_sml.bL_eps)

    println("\n==== η_1 QUANTILES ====")
    @printf("True:    [%.4f, %.4f, %.4f]  b1=%.4f  bL=%.4f\n",
            par_true.a_init..., par_true.b1_init, par_true.bL_init)
    @printf("QR:      [%.4f, %.4f, %.4f]  b1=%.4f  bL=%.4f\n",
            par0.a_init..., par0.b1_init, par0.bL_init)
    @printf("SML-SD:  [%.4f, %.4f, %.4f]  b1=%.4f  bL=%.4f\n",
            par_sml.a_init..., par_sml.b1_init, par_sml.bL_init)

    @printf("\nTime: %.1f s\n", t)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end

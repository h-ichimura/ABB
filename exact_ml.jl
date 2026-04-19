#=
exact_ml.jl — Exact maximum likelihood via forward filtering

The piecewise-uniform + exponential-tail structure allows efficient
numerical integration. Use a fine grid on η_t and forward filter:

  f(η_1 | y_1)        ∝ f_ε(y_1 - η_1) · f_init(η_1)
  f(η_2 | y_1, y_2)   ∝ f_ε(y_2 - η_2) · ∫ f(η_2 | η_1) f(η_1 | y_1) dη_1
  f(η_3 | y_1, y_2, y_3) similarly

Likelihood:
  L_i = f_ε(y_1-η_1) f_init(η_1) · [at step 2, normalizing constant]
      · [at step 3, normalizing constant]

More precisely: the marginal likelihood is
  L_i = ∫ f_init(η_1) f_ε(y_1-η_1) · [∫ f(η_2|η_1) f_ε(y_2-η_2)
        · [∫ f(η_3|η_2) f_ε(y_3-η_3) dη_3] dη_2] dη_1

Computed on a grid: O(T·G²) per observation where G is the grid size.
No simulation, no feasibility issues, smooth in θ.
=#

include("ABB_three_period.jl")
using Optim, Printf

# ================================================================
#  DENSITY EVALUATION ON A GRID
# ================================================================

"""Evaluate piecewise-uniform density at all grid points."""
function pw_dens_grid(grid::Vector{Float64}, q::Vector{Float64},
                     tau::Vector{Float64}, b1::Float64, bL::Float64)
    vals = zeros(length(grid))
    for i in eachindex(grid)
        vals[i] = exp(pw_logdens(grid[i], q, tau, b1, bL))
    end
    vals
end

# ================================================================
#  FORWARD FILTER FOR ONE OBSERVATION
# ================================================================

"""
Precompute the G×G transition density matrix T_mat[g1, g2] = f(grid[g2] | grid[g1]).
This is computed ONCE per likelihood evaluation and shared across all N obs.

The dependence on θ is entirely through par.a_Q and the tail rates.
"""
function compute_transition_matrix!(T_mat::Matrix{Float64},
                                    grid::Vector{Float64},
                                    par::Params, cfg::Config)
    G = length(grid)
    q_buf = zeros(cfg.L)
    @inbounds for g1 in 1:G
        transition_quantiles!(q_buf, grid[g1], par.a_Q, cfg.K, cfg.sigma_y)
        sort!(q_buf)
        for g2 in 1:G
            T_mat[g1, g2] = exp(pw_logdens(grid[g2], q_buf, cfg.tau,
                                           par.b1_Q, par.bL_Q))
        end
    end
end

"""
Compute f_eps(y - grid[g]) for all g; used to condition on y_t.
"""
function eps_likelihood_vec!(vec::Vector{Float64}, y_t::Float64,
                             grid::Vector{Float64}, par::Params,
                             tau::Vector{Float64})
    @inbounds for g in eachindex(grid)
        vec[g] = exp(pw_logdens(y_t - grid[g], par.a_eps, tau,
                                par.b1_eps, par.bL_eps))
    end
end

"""
Compute log-likelihood for all N observations using precomputed T_mat.
T_mat[g1, g2] = f(grid[g2] | grid[g1]).
"""
function loglik_all(y::Matrix{Float64}, par::Params, cfg::Config,
                    grid::Vector{Float64}, dgrid::Float64,
                    T_mat::Matrix{Float64})
    N, T = cfg.N, cfg.T
    G = length(grid)

    # f_init on grid
    f_init = pw_dens_grid(grid, par.a_init, cfg.tau, par.b1_init, par.bL_init)

    # Per-observation working vectors
    p = zeros(G)
    p_new = zeros(G)
    eps_vec = zeros(G)

    total_ll = 0.0
    for i in 1:N
        # Step 1: p = f_init * f_eps(y_1 - η_1)
        eps_likelihood_vec!(eps_vec, y[i, 1], grid, par, cfg.tau)
        @inbounds for g in 1:G; p[g] = f_init[g] * eps_vec[g]; end
        L1 = sum(p) * dgrid
        L1 < 1e-300 && return -Inf
        total_ll += log(L1)
        p ./= L1

        # Step 2: propagate through transition then multiply by eps(y_2)
        # p_new[g2] = f_eps(y_2 - g2) * Σ_g1 T_mat[g1, g2] * p[g1] * dgrid
        mul!(p_new, transpose(T_mat), p)  # p_new[g2] = Σ g1 T[g1,g2]*p[g1]
        p_new .*= dgrid
        eps_likelihood_vec!(eps_vec, y[i, 2], grid, par, cfg.tau)
        @inbounds for g in 1:G; p_new[g] *= eps_vec[g]; end
        L2 = sum(p_new) * dgrid
        L2 < 1e-300 && return -Inf
        total_ll += log(L2)
        p_new ./= L2
        p, p_new = p_new, p  # swap

        # Step 3
        if T >= 3
            mul!(p_new, transpose(T_mat), p)
            p_new .*= dgrid
            eps_likelihood_vec!(eps_vec, y[i, 3], grid, par, cfg.tau)
            @inbounds for g in 1:G; p_new[g] *= eps_vec[g]; end
            L3 = sum(p_new) * dgrid
            L3 < 1e-300 && return -Inf
            total_ll += log(L3)
        end
    end

    total_ll
end

"""
Neg avg log-likelihood. Precomputes transition matrix once.
"""
function exact_neg_loglik(par::Params, y::Matrix{Float64}, cfg::Config;
                          grid_min::Float64=-5.0, grid_max::Float64=5.0,
                          G::Int=100)
    N = cfg.N
    grid = collect(range(grid_min, grid_max, length=G))
    dgrid = (grid_max - grid_min) / (G - 1)

    # Precompute transition density matrix: G × G
    T_mat = zeros(G, G)
    compute_transition_matrix!(T_mat, grid, par, cfg)

    ll = loglik_all(y, par, cfg, grid, dgrid, T_mat)
    -ll / N
end

# ================================================================
#  TEST: EVALUATE AT TRUE THETA
# ================================================================

function test_exact()
    K = 2; L = 3; sigma_y = 1.0; T = 3; N = 100
    tau = [0.25, 0.50, 0.75]
    par_true = make_true_params_linear(tau=tau, sigma_y=sigma_y, K=K)
    y, eta_true = generate_data_abb(N, par_true, tau, sigma_y, K; seed=42)
    cfg = Config(N, T, K, L, tau, sigma_y, 1, 1, 1, fill(0.05, T))

    println("="^60)
    println("  EXACT ML TEST (N=$N)")
    println("="^60)

    # Evaluate at true θ
    t_true = @elapsed ll_true = exact_neg_loglik(par_true, y, cfg; G=100)
    @printf("\nNeg loglik at truth: %.6f  (%.2f s)\n", ll_true, t_true)

    # Evaluate at perturbed θ
    par_pert = copy_params(par_true)
    par_pert.a_Q[2, :] .*= 0.9  # slopes 0.72 instead of 0.8
    ll_pert = exact_neg_loglik(par_pert, y, cfg; G=100)
    @printf("Neg loglik at 0.9x slopes: %.6f  (diff = %+.6f)\n",
            ll_pert, ll_pert - ll_true)

    par_pert.a_Q[2, :] .= par_true.a_Q[2, :]
    par_pert.a_Q[1, :] .*= 0.5  # intercepts halved
    ll_pert = exact_neg_loglik(par_pert, y, cfg; G=100)
    @printf("Neg loglik at 0.5x intercepts: %.6f  (diff = %+.6f)\n",
            ll_pert, ll_pert - ll_true)
end

function estimate_exact_ml(y::Matrix{Float64}, cfg::Config, par0::Params;
                            G::Int=100, grid_min=-5.0, grid_max=5.0,
                            maxiter::Int=100, verbose::Bool=true)
    K, L = cfg.K, cfg.L
    par_work = copy_params(par0)
    v = params_to_vec(par0)

    function obj(v)
        vec_to_params!(par_work, v, K, L)
        exact_neg_loglik(par_work, y, cfg; grid_min=grid_min, grid_max=grid_max, G=G)
    end

    verbose && (@printf("  Exact ML: initial obj = %.6f\n", obj(v)); flush(stdout))

    # LBFGS with numerical gradient (forward differences)
    res = optimize(obj, v, LBFGS(),
                   Optim.Options(iterations=maxiter, g_tol=1e-5, show_trace=verbose,
                                 show_every=5))

    v_opt = Optim.minimizer(res)
    vec_to_params!(par_work, v_opt, K, L)
    @printf("  Exact ML: final obj = %.6f  (iters=%d)\n",
            Optim.minimum(res), Optim.iterations(res))
    par_work
end

# Param packing (same layout as sml.jl)
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

function full_estimate_exact_ml()
    K = 2; L = 3; sigma_y = 1.0; T = 3; N = 200
    tau = [0.25, 0.50, 0.75]
    par_true = make_true_params_linear(tau=tau, sigma_y=sigma_y, K=K)
    y, eta_true = generate_data_abb(N, par_true, tau, sigma_y, K; seed=42)
    cfg = Config(N, T, K, L, tau, sigma_y, 1, 1, 1, fill(0.05, T))

    # Warm start from QR
    println("Getting QR warm start...")
    par0 = init_params(y, cfg)
    eta_all = zeros(N, T, 20)
    for m in 1:20; eta_all[:,:,m] .= 0.6 .* y; end
    cfg_em = Config(N, T, K, L, tau, sigma_y, 5, 100, 20, fill(0.05, T))
    for _ in 1:5
        e_step!(eta_all, y, par0, cfg_em)
        m_step_qr!(par0, eta_all, y, cfg_em)
    end
    @printf("QR slopes:     [%.4f, %.4f, %.4f]\n", par0.a_Q[2,:]...)
    @printf("QR intercepts: [%.4f, %.4f, %.4f]\n", par0.a_Q[1,:]...)

    # Exact ML with smaller grid
    println("\nRunning exact ML (LBFGS, G=60)...")
    t = @elapsed par_ml = estimate_exact_ml(y, cfg, par0; G=60, maxiter=50,
                                              verbose=true)

    @printf("\n==== TRANSITION RESULTS ====\n")
    @printf("True slopes:       [%.4f, %.4f, %.4f]\n", par_true.a_Q[2,:]...)
    @printf("QR slopes:         [%.4f, %.4f, %.4f]\n", par0.a_Q[2,:]...)
    @printf("Exact ML slopes:   [%.4f, %.4f, %.4f]\n", par_ml.a_Q[2,:]...)
    @printf("True intercepts:   [%.4f, %.4f, %.4f]\n", par_true.a_Q[1,:]...)
    @printf("QR intercepts:     [%.4f, %.4f, %.4f]\n", par0.a_Q[1,:]...)
    @printf("Exact ML intcpts:  [%.4f, %.4f, %.4f]\n", par_ml.a_Q[1,:]...)
    @printf("\n==== ε QUANTILES ====\n")
    @printf("True:              [%.4f, %.4f, %.4f]  b1=%.4f  bL=%.4f\n",
            par_true.a_eps..., par_true.b1_eps, par_true.bL_eps)
    @printf("QR:                [%.4f, %.4f, %.4f]  b1=%.4f  bL=%.4f\n",
            par0.a_eps..., par0.b1_eps, par0.bL_eps)
    @printf("Exact ML:          [%.4f, %.4f, %.4f]  b1=%.4f  bL=%.4f\n",
            par_ml.a_eps..., par_ml.b1_eps, par_ml.bL_eps)
    @printf("\n==== η_1 QUANTILES ====\n")
    @printf("True:              [%.4f, %.4f, %.4f]  b1=%.4f  bL=%.4f\n",
            par_true.a_init..., par_true.b1_init, par_true.bL_init)
    @printf("QR:                [%.4f, %.4f, %.4f]  b1=%.4f  bL=%.4f\n",
            par0.a_init..., par0.b1_init, par0.bL_init)
    @printf("Exact ML:          [%.4f, %.4f, %.4f]  b1=%.4f  bL=%.4f\n",
            par_ml.a_init..., par_ml.b1_init, par_ml.bL_init)
    @printf("\nTime: %.1f s\n", t)
end

if abspath(PROGRAM_FILE) == @__FILE__
    test_exact()
    println("\n\n")
    full_estimate_exact_ml()
end

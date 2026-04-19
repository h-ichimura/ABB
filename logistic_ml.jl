#=
logistic_ml.jl — Exact Maximum Likelihood for logistic ABB-style model

Uses forward filter on a grid to compute the observed-data likelihood:
  L(θ; y) = ∏_i ∫ f_init(η_1) · f_trans(η_2|η_1) · f_trans(η_3|η_2)
                    · Π_t f_ε(y_t - η_t) dη

Parameters (10 total):
  μ_Q[1:3]  — conditional median Hermite coeffs
  α_Q[1:3]  — conditional scale Hermite coeffs (α must be positive pointwise)
  μ_init, α_init  — initial η_1 distribution
  μ_eps, α_eps   — noise distribution

QR ignores the cross-percentile restrictions inherent in this parameterization;
logistic MLE exploits them. QR would need 9 a_Q params + 2 tail rates = 11
independent parameters, vs 10 for logistic MLE.
=#

include("logistic_abb_dgp.jl")
using Optim, Printf

# ================================================================
#  FORWARD FILTER (EXACT LIKELIHOOD)
# ================================================================

"""
Compute negative average log-likelihood -1/N · log L(θ; y) via forward filter.

Uses a fixed grid over η support [grid_min, grid_max] with G points.
T_mat[g1, g2] = f_trans(grid[g2] | grid[g1]; θ)  — precomputed per likelihood eval.
"""
function logistic_neg_loglik(p::LogisticParams, y::Matrix{Float64},
                              K::Int, σy::Float64;
                              grid_min::Float64=-5.0, grid_max::Float64=5.0,
                              G::Int=100)
    N, T = size(y, 1), size(y, 2)
    grid = collect(range(grid_min, grid_max, length=G))
    dgrid = (grid_max - grid_min) / (G - 1)

    # Precompute transition density: T_mat[g1, g2] = f(grid[g2] | grid[g1])
    T_mat = zeros(G, G)
    @inbounds for g1 in 1:G
        μ, α = cond_μ_α(p, grid[g1], K, σy)
        for g2 in 1:G
            T_mat[g1, g2] = logistic_pdf(grid[g2], μ, α)
        end
    end

    # f_init on grid
    f_init = [logistic_pdf(grid[g], p.μ_init, p.α_init) for g in 1:G]

    # For each individual: forward filter
    p_vec = zeros(G); p_new = zeros(G); eps_vec = zeros(G)
    total_ll = 0.0

    for i in 1:N
        # t=1: p[g] ∝ f_init(η=g) · f_ε(y[1] - η=g)
        for g in 1:G
            eps_vec[g] = logistic_pdf(y[i, 1] - grid[g], p.μ_eps, p.α_eps)
            p_vec[g] = f_init[g] * eps_vec[g]
        end
        L1 = sum(p_vec) * dgrid
        L1 < 1e-300 && return Inf
        total_ll += log(L1)
        p_vec ./= L1

        # t=2, 3: propagate and multiply by f_ε
        for t in 2:T
            mul!(p_new, transpose(T_mat), p_vec)
            p_new .*= dgrid
            for g in 1:G
                eps_vec[g] = logistic_pdf(y[i, t] - grid[g], p.μ_eps, p.α_eps)
                p_new[g] *= eps_vec[g]
            end
            Lt = sum(p_new) * dgrid
            Lt < 1e-300 && return Inf
            total_ll += log(Lt)
            p_new ./= Lt
            p_vec, p_new = p_new, p_vec
        end
    end

    -total_ll / N
end

# ================================================================
#  PARAMETER VECTOR ⇔ LogisticParams
# ================================================================

"""Flatten params to a vector. α's are log-transformed (positivity)."""
function pack_logistic(p::LogisticParams)
    # μ_Q: 3 raw
    # α_Q: 3 — but α(η) must be > 0 everywhere; constrain α_Q[1] > 0 via log
    #         α_Q[2], α_Q[3] are raw (can be any sign, but α must stay positive;
    #         we soften this via `max(α, 1e-6)` in cond_μ_α)
    # μ_init, log(α_init), μ_eps, log(α_eps): 4
    [p.μ_Q...,
     log(max(p.α_Q[1], 1e-8)), p.α_Q[2], p.α_Q[3],
     p.μ_init, log(max(p.α_init, 1e-8)),
     p.μ_eps, log(max(p.α_eps, 1e-8))]
end

function unpack_logistic(v::Vector{Float64})
    μ_Q = [v[1], v[2], v[3]]
    α_Q = [exp(v[4]), v[5], v[6]]
    LogisticParams(μ_Q, α_Q,
                   v[7], exp(v[8]),
                   v[9], exp(v[10]))
end

# ================================================================
#  ESTIMATION
# ================================================================

"""
Fit logistic MLE by maximizing observed-data log-likelihood via L-BFGS.
Starts from `p0`. Returns fitted LogisticParams.
"""
function estimate_logistic_ml(y::Matrix{Float64}, K::Int, σy::Float64,
                               p0::LogisticParams;
                               G::Int=100, maxiter::Int=100,
                               verbose::Bool=false)
    v0 = pack_logistic(p0)

    function obj(v)
        p = unpack_logistic(v)
        logistic_neg_loglik(p, y, K, σy; G=G)
    end

    res = optimize(obj, v0, LBFGS(),
                   Optim.Options(iterations=maxiter, g_tol=1e-6,
                                 show_trace=verbose, show_every=5))
    v_opt = Optim.minimizer(res)
    verbose && @printf("  Final neg-ll: %.6f  (iters=%d)\n",
                       Optim.minimum(res), Optim.iterations(res))
    unpack_logistic(v_opt)
end

# ================================================================
#  QR ON LOGISTIC DATA (ABB-style piecewise-uniform M-step)
# ================================================================
"""
QR estimator: run ABB's EM with QR M-step on logistic data.
Returns fitted ABB Params (will be misspecified — data is logistic, not PU).
"""
function estimate_qr_on_logistic(y::Matrix{Float64}, K::Int, L::Int, σy::Float64,
                                  τ::Vector{Float64};
                                  S::Int=50, M::Int=20, n_draws::Int=100,
                                  verbose::Bool=false)
    T = size(y, 2); N = size(y, 1)
    cfg = Config(N, T, K, L, τ, σy, S, n_draws, M, fill(0.05, T))
    par = init_params(y, cfg)
    eta_all = zeros(N, T, M)
    for m in 1:M; eta_all[:, :, m] .= 0.6 .* y; end

    for iter in 1:S
        e_step!(eta_all, y, par, cfg)
        m_step_qr!(par, eta_all, y, cfg)
        if verbose && iter % 10 == 0
            @printf("  QR iter %d: slopes=[%.3f, %.3f, %.3f]\n",
                    iter, par.a_Q[2, :] ./ σy...)
        end
    end
    par
end

# ================================================================
#  SANITY CHECK
# ================================================================
if abspath(PROGRAM_FILE) == @__FILE__
    K = 2; L = 3; σy = 1.0
    τ = [0.25, 0.50, 0.75]

    println("="^70)
    println("  LOGISTIC MLE — single seed sanity check")
    println("="^70)

    # True params
    p_true = make_true_logistic_params()
    N = 1000

    println("\nGenerating data (N=$N, seed=42)...")
    y, η = generate_data_logistic(N, p_true, K, σy; seed=42)
    @printf("  η range [%.3f, %.3f], y range [%.3f, %.3f]\n",
            extrema(η)..., extrema(y)...)

    # True value of neg-ll
    println("\nNeg-ll at truth...")
    t_eval = @elapsed nll_true = logistic_neg_loglik(p_true, y, K, σy; G=120)
    @printf("  neg-ll(truth) = %.6f  (eval %.2fs)\n", nll_true, t_eval)

    # MLE
    println("\nFitting logistic MLE...")
    # Start from slightly perturbed truth for numerical stability
    p0 = copy_params_log(p_true)
    p0.μ_Q[2] = 0.5  # slope perturbed from 0.8
    p0.α_Q[1] = 3.0  # scale perturbed

    t_mle = @elapsed p_mle = estimate_logistic_ml(y, K, σy, p0; G=100,
                                                   maxiter=200, verbose=true)
    nll_mle = logistic_neg_loglik(p_mle, y, K, σy; G=120)

    @printf("\n--- Logistic MLE results (time %.1fs) ---\n", t_mle)
    @printf("True:\n")
    @printf("  μ_Q: [%.4f, %.4f, %.4f]  (slope of median)\n", p_true.μ_Q...)
    @printf("  α_Q: [%.4f, %.4f, %.4f]  (scale)\n", p_true.α_Q...)
    @printf("  μ_init=%.4f, α_init=%.4f\n", p_true.μ_init, p_true.α_init)
    @printf("  μ_eps=%.4f,  α_eps=%.4f\n", p_true.μ_eps, p_true.α_eps)
    @printf("MLE:\n")
    @printf("  μ_Q: [%.4f, %.4f, %.4f]\n", p_mle.μ_Q...)
    @printf("  α_Q: [%.4f, %.4f, %.4f]\n", p_mle.α_Q...)
    @printf("  μ_init=%.4f, α_init=%.4f\n", p_mle.μ_init, p_mle.α_init)
    @printf("  μ_eps=%.4f,  α_eps=%.4f\n", p_mle.μ_eps, p_mle.α_eps)
    @printf("neg-ll: truth=%.6f, MLE=%.6f, diff=%+.6f\n",
            nll_true, nll_mle, nll_mle - nll_true)

    # Implied a_Q from MLE
    a_Q_implied = logistic_to_aQ(p_mle, τ, K, σy)
    a_Q_true = logistic_to_aQ(p_true, τ, K, σy)
    println("\nImplied ABB-style a_Q (from MLE):")
    @printf("  slopes:    [%.4f, %.4f, %.4f]  (true [%.4f, %.4f, %.4f])\n",
            a_Q_implied[2, :]..., a_Q_true[2, :]...)
    @printf("  intercepts:[%.4f, %.4f, %.4f]  (true [%.4f, %.4f, %.4f])\n",
            a_Q_implied[1, :]..., a_Q_true[1, :]...)

    # QR fit
    println("\nFitting ABB QR on the same data...")
    t_qr = @elapsed par_qr = estimate_qr_on_logistic(y, K, L, σy, τ; S=50, M=20,
                                                       verbose=false)
    @printf("QR (time %.1fs):\n", t_qr)
    @printf("  slopes:    [%.4f, %.4f, %.4f]\n", par_qr.a_Q[2, :] ./ σy...)
    @printf("  intercepts:[%.4f, %.4f, %.4f]\n", par_qr.a_Q[1, :]...)
    @printf("  quadratic: [%.4f, %.4f, %.4f]\n", par_qr.a_Q[3, :]...)
    @printf("  b1_Q=%.4f, bL_Q=%.4f\n", par_qr.b1_Q, par_qr.bL_Q)
end

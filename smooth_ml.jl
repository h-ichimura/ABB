#=
smooth_ml.jl — Exact ML with SMOOTH ABB density (cubic log-density interior,
exponential tails), using the same Hermite quantile knots as ABB.

The transition log-density is:
  log f(η_t | η_{t-1}) = β_0 + β_1 η_t + Σ_l γ_l (η_t - q_l(η_{t-1}))₊³ - log C
where:
  - q_l(η_{t-1}) = Σ_k a_{kl} He_k(η_{t-1}/σ_y) is ABB's quantile knot at τ_l
  - β_0, β_1, γ_l are spline coefficients (depend on η_{t-1} via Hermite basis too)
  - C is the normalizing constant

The density is smooth (C² at knots, exponential tails), so MLE is
a smooth optimization problem.

Marginals (f_init, f_eps) kept as piecewise-uniform (ABB form).
=#

include("ABB_three_period.jl")
using Optim, Printf, LinearAlgebra

# ================================================================
#  GAUSS-LEGENDRE QUADRATURE
# ================================================================

function gauss_legendre_nodes(n::Int)
    beta_coef = [i / sqrt(4i^2 - 1) for i in 1:n-1]
    J = SymTridiagonal(zeros(n), beta_coef)
    lam, V = eigen(J)
    w = 2.0 .* V[1,:].^2
    lam, w
end

const GL16_n, GL16_w = gauss_legendre_nodes(16)

function gl_int(f, a::Float64, b::Float64;
                nodes=GL16_n, weights=GL16_w)
    mid = (a + b) / 2; half = (b - a) / 2
    s = 0.0
    for i in eachindex(nodes)
        s += weights[i] * f(mid + half * nodes[i])
    end
    s * half
end

# ================================================================
#  SMOOTH TRANSITION DENSITY
# ================================================================

"""
Evaluate log s(x; η_{t-1}, a_Q, b_coef) = β_0 + β_1 x + Σ_l γ_l (x-q_l)₊³.
b_coef: (K+1) × (L+2) coefficients for β_0, β_1, γ_1,...,γ_L

Returns (s_value, qknots_sorted) for use in normalization.
"""
function smooth_s(x::Float64, eta_lag::Float64,
                  a_Q::Matrix{Float64}, b_coef::Matrix{Float64},
                  K::Int, L::Int, sigma_y::Float64,
                  hv::Vector{Float64}, qknots::Vector{Float64})
    # Hermite basis at η_{t-1}
    z = eta_lag / sigma_y
    hv[1] = 1.0
    K >= 1 && (hv[2] = z)
    for k in 2:K; hv[k+1] = z * hv[k] - (k - 1) * hv[k-1]; end
    # Quantile knots
    for l in 1:L
        s = 0.0
        for k in 1:K+1; s += a_Q[k, l] * hv[k]; end
        qknots[l] = s
    end
    sort!(qknots)
    # Spline coefficients
    beta0 = 0.0; beta1 = 0.0
    for k in 1:K+1
        beta0 += b_coef[k, 1] * hv[k]
        beta1 += b_coef[k, 2] * hv[k]
    end
    # s(x) = β_0 + β_1 x + Σ γ_l (x - q_l)₊³
    sv = beta0 + beta1 * x
    for l in 1:L
        d = x - qknots[l]
        if d > 0
            g = 0.0
            for k in 1:K+1; g += b_coef[k, l+2] * hv[k]; end
            sv += g * d * d * d
        end
    end
    (sv, beta0, beta1, qknots)
end

"""
Normalizing constant C = ∫ exp(s(x)) dx.
Left tail exponential (β_1 > 0): integrates to exp(β_0 + β_1 q_1) / β_1
Right tail: if γ_sum < 0, cubic decay, integrate numerically.
"""
function smooth_C(eta_lag::Float64, a_Q::Matrix{Float64}, b_coef::Matrix{Float64},
                  K::Int, L::Int, sigma_y::Float64;
                  hi::Float64=10.0)
    hv = zeros(K + 1)
    qknots = zeros(L)
    # Get coefficients
    z = eta_lag / sigma_y
    hv[1] = 1.0; K >= 1 && (hv[2] = z)
    for k in 2:K; hv[k+1] = z * hv[k] - (k - 1) * hv[k-1]; end
    for l in 1:L
        s = 0.0
        for k in 1:K+1; s += a_Q[k, l] * hv[k]; end
        qknots[l] = s
    end
    sort!(qknots)
    beta0 = 0.0; beta1 = 0.0
    for k in 1:K+1
        beta0 += b_coef[k, 1] * hv[k]
        beta1 += b_coef[k, 2] * hv[k]
    end
    gamma = zeros(L)
    for l in 1:L
        for k in 1:K+1; gamma[l] += b_coef[k, l+2] * hv[k]; end
    end

    # Left tail: x < q_1, log f = β_0 + β_1 x (cubics all 0)
    # Need β_1 > 0 for integrable left tail
    C_left = if beta1 > 1e-8
        exp(beta0 + beta1 * qknots[1]) / beta1
    else
        return Inf  # improper
    end

    # Interior segments: numerical integration
    function s_fn(x)
        v = beta0 + beta1 * x
        for l in 1:L
            d = x - qknots[l]
            d > 0 && (v += gamma[l] * d * d * d)
        end
        exp(v)
    end
    C_interior = 0.0
    for i in 1:L-1
        C_interior += gl_int(s_fn, qknots[i], qknots[i+1])
    end

    # Right tail: x > q_L, need γ_sum < 0 for integrable
    gamma_sum = sum(gamma)
    C_right = if gamma_sum < -1e-8
        gl_int(s_fn, qknots[L], hi)
    else
        return Inf
    end

    C_left + C_interior + C_right
end

# ================================================================
#  TRANSITION DENSITY ON A GRID (precomputed)
# ================================================================

"""
Precompute G×G transition density matrix: T_mat[g1, g2] = f(grid[g2] | grid[g1]).
"""
function smooth_transition_matrix!(T_mat::Matrix{Float64},
                                   grid::Vector{Float64},
                                   a_Q::Matrix{Float64},
                                   b_coef::Matrix{Float64},
                                   K::Int, L::Int, sigma_y::Float64)
    G = length(grid)
    hv = zeros(K + 1)
    qknots = zeros(L)
    gamma = zeros(L)

    @inbounds for g1 in 1:G
        η_lag = grid[g1]
        # Coefficients at this η_lag
        z = η_lag / sigma_y
        hv[1] = 1.0; K >= 1 && (hv[2] = z)
        for k in 2:K; hv[k+1] = z * hv[k] - (k - 1) * hv[k-1]; end
        for l in 1:L
            q = 0.0
            for k in 1:K+1; q += a_Q[k, l] * hv[k]; end
            qknots[l] = q
        end
        sort!(qknots)
        beta0 = 0.0; beta1 = 0.0
        for k in 1:K+1
            beta0 += b_coef[k, 1] * hv[k]
            beta1 += b_coef[k, 2] * hv[k]
        end
        for l in 1:L
            gamma[l] = 0.0
            for k in 1:K+1; gamma[l] += b_coef[k, l+2] * hv[k]; end
        end

        # Normalizing constant
        if beta1 <= 1e-8 || sum(gamma) >= -1e-8
            for g2 in 1:G; T_mat[g1, g2] = 1e-300; end
            continue
        end
        C_left = exp(beta0 + beta1 * qknots[1]) / beta1
        function s_fn(x)
            v = beta0 + beta1 * x
            for l in 1:L
                d = x - qknots[l]
                d > 0 && (v += gamma[l] * d * d * d)
            end
            exp(v)
        end
        C_int = 0.0
        for i in 1:L-1
            C_int += gl_int(s_fn, qknots[i], qknots[i+1])
        end
        C_right = gl_int(s_fn, qknots[L], 10.0)
        C_total = C_left + C_int + C_right

        # Fill row
        for g2 in 1:G
            T_mat[g1, g2] = s_fn(grid[g2]) / max(C_total, 1e-300)
        end
    end
end

# ================================================================
#  EXACT ML WITH SMOOTH TRANSITION
# ================================================================

"""
Compute negative average log-likelihood using forward filter with smooth transition.
a_Q: (K+1) × L quantile knot coefficients (same as ABB)
b_coef: (K+1) × (L+2) spline coefficients (new)
"""
function smooth_neg_loglik(par::Params, b_coef::Matrix{Float64},
                           y::Matrix{Float64}, cfg::Config;
                           grid_min::Float64=-5.0, grid_max::Float64=5.0,
                           G::Int=80)
    N, T = cfg.N, cfg.T
    K, L = cfg.K, cfg.L

    grid = collect(range(grid_min, grid_max, length=G))
    dgrid = (grid_max - grid_min) / (G - 1)

    # Precompute transition matrix
    T_mat = zeros(G, G)
    smooth_transition_matrix!(T_mat, grid, par.a_Q, b_coef, K, L, cfg.sigma_y)

    # f_init on grid (piecewise-uniform, unchanged)
    f_init = [exp(pw_logdens(grid[g], par.a_init, cfg.tau,
                              par.b1_init, par.bL_init)) for g in 1:G]

    p = zeros(G); p_new = zeros(G); eps_vec = zeros(G)
    total_ll = 0.0

    for i in 1:N
        # Step 1: p = f_init * f_ε(y_1 - η_1)
        for g in 1:G
            eps_vec[g] = exp(pw_logdens(y[i, 1] - grid[g], par.a_eps, cfg.tau,
                                         par.b1_eps, par.bL_eps))
            p[g] = f_init[g] * eps_vec[g]
        end
        L1 = sum(p) * dgrid
        L1 < 1e-300 && return Inf
        total_ll += log(L1)
        p ./= L1

        # Step 2
        mul!(p_new, transpose(T_mat), p)
        p_new .*= dgrid
        for g in 1:G
            eps_vec[g] = exp(pw_logdens(y[i, 2] - grid[g], par.a_eps, cfg.tau,
                                         par.b1_eps, par.bL_eps))
            p_new[g] *= eps_vec[g]
        end
        L2 = sum(p_new) * dgrid
        L2 < 1e-300 && return Inf
        total_ll += log(L2)
        p_new ./= L2
        p, p_new = p_new, p

        # Step 3
        if T >= 3
            mul!(p_new, transpose(T_mat), p)
            p_new .*= dgrid
            for g in 1:G
                eps_vec[g] = exp(pw_logdens(y[i, 3] - grid[g], par.a_eps, cfg.tau,
                                             par.b1_eps, par.bL_eps))
                p_new[g] *= eps_vec[g]
            end
            L3 = sum(p_new) * dgrid
            L3 < 1e-300 && return Inf
            total_ll += log(L3)
        end
    end
    -total_ll / N
end

# ================================================================
#  ESTIMATION: PARAMETER VECTOR <-> (Params, b_coef)
# ================================================================

function smooth_params_to_vec(par::Params, b_coef::Matrix{Float64}, K::Int, L::Int)
    v = Float64[]
    append!(v, vec(par.a_Q))
    append!(v, vec(b_coef))  # (K+1) × (L+2)
    append!(v, par.a_init); push!(v, log(par.b1_init), log(par.bL_init))
    append!(v, par.a_eps);  push!(v, log(par.b1_eps),  log(par.bL_eps))
    v
end

function smooth_vec_to_params!(par::Params, b_coef::Matrix{Float64},
                                v::Vector{Float64}, K::Int, L::Int)
    np_aQ = (K + 1) * L
    np_b = (K + 1) * (L + 2)
    par.a_Q .= reshape(view(v, 1:np_aQ), K + 1, L)
    b_coef .= reshape(view(v, np_aQ+1:np_aQ+np_b), K + 1, L + 2)
    i = np_aQ + np_b
    par.a_init .= view(v, i+1:i+L); i += L
    par.b1_init = exp(v[i+1]); par.bL_init = exp(v[i+2]); i += 2
    par.a_eps .= view(v, i+1:i+L); i += L
    par.b1_eps = exp(v[i+1]); par.bL_eps = exp(v[i+2])
    nothing
end

"""
Initial b_coef: chosen so the smooth density approximates ABB's piecewise-uniform.
β_1 > 0 for integrable left tail, γ summing to < 0 for integrable right tail.
"""
function init_smooth_b_coef(K::Int, L::Int)
    b = zeros(K + 1, L + 2)
    b[1, 2] = 2.0      # β_1 (constant in η_lag)
    for l in 1:L
        b[1, l + 2] = -0.3 / L  # γ_l (constant in η_lag)
    end
    b
end

function estimate_smooth_ml(y::Matrix{Float64}, cfg::Config, par0::Params;
                             G::Int=80, maxiter::Int=100, verbose::Bool=true)
    K, L = cfg.K, cfg.L
    par = copy_params(par0)
    b_coef = init_smooth_b_coef(K, L)

    function obj(v)
        smooth_vec_to_params!(par, b_coef, v, K, L)
        smooth_neg_loglik(par, b_coef, y, cfg; G=G)
    end

    v0 = smooth_params_to_vec(par0, b_coef, K, L)
    verbose && @printf("  Smooth ML initial obj: %.6f\n", obj(v0))

    res = optimize(obj, v0, LBFGS(),
                   Optim.Options(iterations=maxiter, g_tol=1e-5,
                                 show_trace=verbose, show_every=5))
    v_opt = Optim.minimizer(res)
    smooth_vec_to_params!(par, b_coef, v_opt, K, L)
    @printf("  Smooth ML final obj: %.6f (iters=%d)\n",
            Optim.minimum(res), Optim.iterations(res))
    par, b_coef
end

# ================================================================
#  TEST
# ================================================================

"""
Draw η from the smooth transition density using inverse-CDF on a fine grid.
"""
function smooth_draw(rng::AbstractRNG, eta_lag::Float64,
                     a_Q::Matrix{Float64}, b_coef::Matrix{Float64},
                     K::Int, L::Int, sigma_y::Float64;
                     grid_min=-5.0, grid_max=5.0, G=200)
    grid = collect(range(grid_min, grid_max, length=G))
    # Compute density on grid
    dens = zeros(G)
    for g in 1:G
        _, _, _, _ = smooth_s(grid[g], eta_lag, a_Q, b_coef, K, L, sigma_y,
                              zeros(K+1), zeros(L))
    end
    # Actually just compute unnormalized then normalize
    hv = zeros(K+1); qknots = zeros(L); gamma = zeros(L)
    z = eta_lag / sigma_y
    hv[1] = 1.0; K >= 1 && (hv[2] = z)
    for k in 2:K; hv[k+1] = z*hv[k] - (k-1)*hv[k-1]; end
    for l in 1:L
        s=0.0; for k in 1:K+1; s += a_Q[k,l]*hv[k]; end
        qknots[l] = s
    end
    sort!(qknots)
    beta0 = 0.0; beta1 = 0.0
    for k in 1:K+1
        beta0 += b_coef[k,1]*hv[k]; beta1 += b_coef[k,2]*hv[k]
    end
    for l in 1:L
        for k in 1:K+1; gamma[l] += b_coef[k,l+2]*hv[k]; end
    end

    for g in 1:G
        x = grid[g]
        s = beta0 + beta1 * x
        for l in 1:L
            d = x - qknots[l]
            d > 0 && (s += gamma[l] * d * d * d)
        end
        dens[g] = exp(s)
    end
    # Inverse CDF via cumulative sum
    cdf = cumsum(dens) .* ((grid_max-grid_min)/(G-1))
    cdf ./= cdf[end]
    u = rand(rng)
    idx = searchsortedfirst(cdf, u)
    idx = clamp(idx, 1, G)
    grid[idx]
end

"""Generate data from smooth transition model with given b_coef."""
function generate_data_smooth(N::Int, par::Params, b_coef::Matrix{Float64},
                              tau, sigma_y, K::Int; seed::Int=42)
    rng = MersenneTwister(seed)
    T = 3; L = length(tau)
    eta = zeros(N, T); y = zeros(N, T)
    for i in 1:N
        eta[i, 1] = pw_draw(rng, par.a_init, tau, par.b1_init, par.bL_init)
    end
    for t in 2:T, i in 1:N
        eta[i, t] = smooth_draw(rng, eta[i, t-1], par.a_Q, b_coef, K, L, sigma_y)
    end
    for t in 1:T, i in 1:N
        y[i, t] = eta[i, t] + pw_draw(rng, par.a_eps, tau, par.b1_eps, par.bL_eps)
    end
    y, eta
end

function main()
    K = 2; L = 3; sigma_y = 1.0; T = 3; N = 200
    tau = [0.25, 0.50, 0.75]
    par_true = make_true_params_linear(tau=tau, sigma_y=sigma_y, K=K)

    # Draw data from SMOOTH model (not piecewise-uniform)
    b_true = init_smooth_b_coef(K, L)
    println("Drawing data from SMOOTH model (b_true = default init)...")
    @printf("  Generating %d × 2 transitions...\n", N); flush(stdout)
    t_gen = @elapsed y, eta_true = generate_data_smooth(N, par_true, b_true, tau, sigma_y, K; seed=42)
    @printf("  Done in %.1fs. η range: [%.3f, %.3f]\n", t_gen, extrema(eta_true)...)

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

    println("\nRunning smooth ML (LBFGS on smooth density)...")
    t = @elapsed par_sml, b_opt = estimate_smooth_ml(y, cfg, par0; G=80,
                                                    maxiter=50, verbose=true)

    println("\n==== TRANSITION ====")
    @printf("True slopes:    [%.4f, %.4f, %.4f]\n", par_true.a_Q[2,:]...)
    @printf("QR slopes:      [%.4f, %.4f, %.4f]\n", par0.a_Q[2,:]...)
    @printf("Smooth slopes:  [%.4f, %.4f, %.4f]\n", par_sml.a_Q[2,:]...)
    @printf("True intcpts:   [%.4f, %.4f, %.4f]\n", par_true.a_Q[1,:]...)
    @printf("QR intcpts:     [%.4f, %.4f, %.4f]\n", par0.a_Q[1,:]...)
    @printf("Smooth intcpts: [%.4f, %.4f, %.4f]\n", par_sml.a_Q[1,:]...)

    println("\n==== η_1 QUANTILES ====")
    @printf("True:    [%.4f, %.4f, %.4f]  b1=%.4f  bL=%.4f\n",
            par_true.a_init..., par_true.b1_init, par_true.bL_init)
    @printf("QR:      [%.4f, %.4f, %.4f]  b1=%.4f  bL=%.4f\n",
            par0.a_init..., par0.b1_init, par0.bL_init)
    @printf("Smooth:  [%.4f, %.4f, %.4f]  b1=%.4f  bL=%.4f\n",
            par_sml.a_init..., par_sml.b1_init, par_sml.bL_init)

    println("\n==== ε QUANTILES ====")
    @printf("True:    [%.4f, %.4f, %.4f]  b1=%.4f  bL=%.4f\n",
            par_true.a_eps..., par_true.b1_eps, par_true.bL_eps)
    @printf("QR:      [%.4f, %.4f, %.4f]  b1=%.4f  bL=%.4f\n",
            par0.a_eps..., par0.b1_eps, par0.bL_eps)
    @printf("Smooth:  [%.4f, %.4f, %.4f]  b1=%.4f  bL=%.4f\n",
            par_sml.a_eps..., par_sml.b1_eps, par_sml.bL_eps)

    @printf("\nTime: %.1f s\n", t)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end

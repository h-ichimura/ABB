#=
cspline_abb_gpu.jl — GPU-accelerated forward filter + FFBS for the ABB
piecewise-uniform model.

Bottleneck on CPU is the forward filter recursion
   p_t[g2] = ∑_{g1} T[g1, g2] · p_{t-1}[g1] · h_grid · f_eps(y_t - grid[g2])
which costs O(N · T · G²) per likelihood (or per FFBS) call.  We stack all N
panels into a single G×N matrix and replace the per-panel loop by a single
GEMM at each time step:
   P_pred = T' · (P · h_grid)         # G×N  =  G×G * G×N

Two functions exposed:

  • abb_uniform_loglik_gpu(...)     — total log-likelihood for the *original
                                       ABB MLE* parameterisation (with tunable
                                       b1_eps, bL_eps, b1_Q, bL_Q tail rates).
                                       Uses pw_logdens(...) so the result is
                                       identical to the CPU `loglik_all`.

  • abb_uniform_ffbs_gpu!(...)      — FFBS draw used inside the QR EM loop.
                                       Forward filter on GPU; backward
                                       sampling on CPU (sequential, O(N·T·G)
                                       and not a bottleneck).

Both require CUDA.jl to be loaded into Main BEFORE this file is included.

Style note: this file mirrors the `cspline_neg_loglik_gpu` GPU pattern
already in cspline_abb.jl (line 1834).  We do NOT modify the CPU
functions; new code lives here so the existing tests stay intact.
=#

using LinearAlgebra, Random

# Sanity check at include time: bail loudly if CUDA isn't loaded.
if !(isdefined(Main, :CUDA) && Main.CUDA.functional())
    error("cspline_abb_gpu.jl: CUDA.jl is not loaded or not functional. " *
          "Add `using CUDA` and ensure CUDA.functional() is true before " *
          "include(\"cspline_abb_gpu.jl\").")
end

# Bring CUDA names into local scope
const _CUDA = Main.CUDA
const _CuArray = Main.CuArray

# ================================================================
#  GPU FORWARD FILTER FOR THE EXACT ABB-PW MLE
# ================================================================

"""
    abb_uniform_loglik_gpu(par, y, cfg, grid, dgrid; G=201)

Total log-likelihood ∑_i log L_i for the ABB piecewise-uniform model with
tunable tail rates.  Drop-in replacement for `loglik_all` from exact_ml.jl.

Returns -Inf if any normaliser drops below 1e-300 (parameter region invalid).
"""
function abb_uniform_loglik_gpu(y::Matrix{Float64}, par, cfg,
                                 grid::Vector{Float64}, dgrid::Float64,
                                 T_mat::Matrix{Float64})
    N, T_obs = size(y)
    G = length(grid)

    # Initial density on CPU (pw) — NO normalisation, matches loglik_all
    # in exact_ml.jl which uses raw pw_dens values.  The pw density integrates
    # to 1 analytically; the discrete grid has Riemann error which is
    # absorbed identically by CPU and GPU paths.
    f_init = zeros(G)
    @inbounds for g in 1:G
        f_init[g] = exp(pw_logdens(grid[g], par.a_init, cfg.tau,
                                    par.b1_init, par.bL_init))
    end

    # f_eps[g, i, t] = f_eps(y[i,t] - grid[g]) on CPU (G·N·T floats)
    eps_dens = Array{Float64}(undef, G, N, T_obs)
    @inbounds for t_step in 1:T_obs, i in 1:N, g in 1:G
        eps_dens[g, i, t_step] = exp(pw_logdens(y[i, t_step] - grid[g],
                                                par.a_eps, cfg.tau,
                                                par.b1_eps, par.bL_eps))
    end

    # Transfer to GPU (single transfer)
    T_d   = _CuArray(T_mat)                          # G × G
    fI_d  = _CuArray(f_init)                          # G
    eps_d = _CuArray(eps_dens)                        # G × N × T

    # ---- t = 1 ------------------------------------------------------
    P_d = fI_d .* view(eps_d, :, :, 1)                # G × N
    L_d = sum(P_d; dims = 1) .* dgrid                 # 1 × N
    L_cpu = Array(L_d)
    any(L_cpu .< 1e-300) && return -Inf
    total_ll = sum(log.(L_cpu))
    P_d ./= L_d                                       # normalise columns

    # ---- t ≥ 2 ------------------------------------------------------
    for t_step in 2:T_obs
        P_d .*= dgrid                                 # weight for quadrature
        P_pred_d = transpose(T_d) * P_d               # G × N — main GEMM
        P_d = P_pred_d .* view(eps_d, :, :, t_step)
        L_d = sum(P_d; dims = 1) .* dgrid
        L_cpu = Array(L_d)
        any(L_cpu .< 1e-300) && return -Inf
        total_ll += sum(log.(L_cpu))
        P_d ./= L_d
    end

    total_ll
end

"""
    exact_neg_loglik_gpu(par, y, cfg; grid_min, grid_max, G)

GPU drop-in replacement for `exact_neg_loglik` from exact_ml.jl. Returns
-loglik / N (negative average log-likelihood).  Builds the transition matrix
on CPU (G Newton-free pw_logdens evals — fast) and does the forward filter
on GPU.
"""
function exact_neg_loglik_gpu(par, y::Matrix{Float64}, cfg;
                               grid_min::Float64 = -8.0,
                               grid_max::Float64 = 8.0,
                               G::Int = 201)
    N = cfg.N
    grid  = collect(range(grid_min, grid_max, length = G))
    dgrid = (grid_max - grid_min) / (G - 1)

    T_mat = zeros(G, G)
    compute_transition_matrix!(T_mat, grid, par, cfg)

    ll = abb_uniform_loglik_gpu(y, par, cfg, grid, dgrid, T_mat)
    -ll / N
end

# ================================================================
#  GPU FFBS FOR THE QR E-STEP
# ================================================================

"""
    abb_uniform_ffbs_gpu!(eta_draw, a_Q, a_init, a_eps1, a_eps3, y, K, σy, rng;
                          G=201)

GPU forward filter + CPU backward sampling, semantically identical to
`abb_uniform_ffbs!` in cspline_abb.jl line 3693.  Stores filtered marginals
on GPU during the forward pass, transfers G×N×T_obs back to CPU once for
the sequential backward sample.

`eta_draw` is updated in place and returned.
"""
function abb_uniform_ffbs_gpu!(eta_draw::Matrix{Float64},
                                a_Q::Matrix{Float64},
                                a_init::Vector{Float64},
                                a_eps1::Float64, a_eps3::Float64,
                                y::Matrix{Float64}, K::Int, σy::Float64,
                                rng::AbstractRNG; G::Int = 201)
    N, T_obs = size(y)
    G = isodd(G) ? G : G + 1
    grid = collect(range(-8.0, 8.0, length = G))
    h_grid = grid[2] - grid[1]

    # Build transition matrix (CPU, G·G evals — fast)
    T_mat = zeros(G, G)
    abb_uniform_transition_matrix!(T_mat, grid, G, a_Q, K, σy)

    # Initial density (CPU)
    f_init = zeros(G)
    if a_init[2] > a_init[1] && a_init[3] > a_init[2]
        @inbounds for g in 1:G
            f_init[g] = exp(abb_uniform_logf(grid[g], a_init))
        end
        Cinit = sum(f_init) * h_grid
        Cinit > 1e-300 && (f_init ./= Cinit)
    else
        fill!(f_init, 1.0 / G)
    end

    # Eps support (clip if non-monotone)
    a_eps = [a_eps1, 0.0, a_eps3]
    if !(a_eps[2] > a_eps[1] && a_eps[3] > a_eps[2])
        a_eps = [-0.5, 0.0, 0.5]
    end

    # eps_dens[g, i, t] (CPU)
    eps_dens = Array{Float64}(undef, G, N, T_obs)
    @inbounds for t_step in 1:T_obs, i in 1:N, g in 1:G
        eps_dens[g, i, t_step] = exp(abb_uniform_logf(y[i, t_step] - grid[g],
                                                       a_eps))
    end

    # ---- GPU forward pass --------------------------------------------------
    T_d     = _CuArray(T_mat)                        # G × G
    fI_d    = _CuArray(f_init)                       # G
    eps_d   = _CuArray(eps_dens)                     # G × N × T

    # filter_p_d[:, :, t] holds p(η_t | y_{1..t})  (column-normalised)
    filter_p_d = _CUDA.zeros(Float64, G, N, T_obs)

    # t = 1
    P_d = fI_d .* view(eps_d, :, :, 1)
    L_d = sum(P_d; dims = 1) .* h_grid
    P_d ./= max.(L_d, 1e-300)
    @views filter_p_d[:, :, 1] .= P_d

    for t_step in 2:T_obs
        P_d .*= h_grid
        P_pred_d = transpose(T_d) * P_d
        P_d = P_pred_d .* view(eps_d, :, :, t_step)
        L_d = sum(P_d; dims = 1) .* h_grid
        P_d ./= max.(L_d, 1e-300)
        @views filter_p_d[:, :, t_step] .= P_d
    end

    # Single bulk transfer back to CPU for the sequential backward sample
    filter_p = Array(filter_p_d)

    # ---- CPU backward sample -----------------------------------------------
    cdf = zeros(G); p = zeros(G)
    @inbounds for i in 1:N
        # η_T
        for g in 1:G; cdf[g] = filter_p[g, i, T_obs] * h_grid; end
        cumsum!(cdf, cdf); cdf ./= cdf[end]
        idx = searchsortedfirst(cdf, rand(rng))
        eta_draw[i, T_obs] = grid[clamp(idx, 1, G)]

        # η_{T-1} ... η_1
        for t_step in (T_obs - 1):-1:1
            g_next = clamp(round(Int,
                       (eta_draw[i, t_step + 1] - grid[1]) / h_grid) + 1,
                       1, G)
            for g in 1:G
                p[g] = T_mat[g, g_next] * filter_p[g, i, t_step] * h_grid
            end
            cumsum!(cdf, p); cdf ./= cdf[end]
            idx = searchsortedfirst(cdf, rand(rng))
            eta_draw[i, t_step] = grid[clamp(idx, 1, G)]
        end
    end
    eta_draw
end

# ================================================================
#  GPU QR ESTIMATOR (drop-in for estimate_abb_uniform_qr)
# ================================================================

"""
    estimate_abb_uniform_qr_gpu(y, K, σy, a_Q0, a_init0, a_eps10, a_eps30, τ;
                                 G=201, S_em=30, M_draws=10, verbose=false, seed=1)

QR estimator with GPU FFBS in the E-step.  Identical interface and identical
output structure to `estimate_abb_uniform_qr` (cspline_abb.jl line 3791).
"""
function estimate_abb_uniform_qr_gpu(y::Matrix{Float64}, K::Int, σy::Float64,
                                      a_Q0::Matrix{Float64},
                                      a_init0::Vector{Float64},
                                      a_eps10::Float64, a_eps30::Float64,
                                      τ::Vector{Float64};
                                      G::Int = 201, S_em::Int = 30,
                                      M_draws::Int = 10,
                                      verbose::Bool = false, seed::Int = 1)
    N, T_obs = size(y)
    rng = MersenneTwister(seed)

    a_Q     = copy(a_Q0)
    a_init  = copy(a_init0)
    a_eps1  = a_eps10
    a_eps3  = a_eps30

    eta_draw = zeros(N, T_obs)

    for iter in 1:S_em
        a_Q_sum    = zeros(K + 1, length(τ))
        a_init_sum = zeros(length(τ))
        ae1_sum    = 0.0
        ae3_sum    = 0.0

        for m in 1:M_draws
            abb_uniform_ffbs_gpu!(eta_draw, a_Q, a_init, a_eps1, a_eps3,
                                   y, K, σy, rng; G = G)
            qr_est       = cspline_qr_mstep(eta_draw, y, K, σy, τ)
            a_Q_sum    .+= qr_est.a_Q
            a_init_sum .+= qr_est.a_init
            ae1_sum     += qr_est.a_eps1
            ae3_sum     += qr_est.a_eps3
        end

        a_Q    .= a_Q_sum    ./ M_draws
        a_init .= a_init_sum ./ M_draws
        a_eps1  = ae1_sum    /  M_draws
        a_eps3  = ae3_sum    /  M_draws

        if verbose && (iter <= 5 || iter % 10 == 0)
            @printf("  ABB-QR-GPU iter %3d: ρ=%.4f  a_init=[%.3f,%.3f,%.3f]  a_eps=[%.3f,%.3f]\n",
                    iter, a_Q[2,2], a_init..., a_eps1, a_eps3); flush(stdout)
        end
    end

    M_Q    = _M_from_iqr(a_Q[1, 3] - a_Q[1, 1])
    M_init = _M_from_iqr(a_init[3] - a_init[1])
    M_eps  = _M_from_iqr(a_eps3 - a_eps1)

    (a_Q = a_Q, a_init = a_init, a_eps1 = a_eps1, a_eps3 = a_eps3,
     M_Q = M_Q, M_init = M_init, M_eps = M_eps)
end

# ================================================================
#  GPU EXACT MLE (drop-in for estimate_exact_ml)
# ================================================================

"""
    estimate_exact_ml_gpu(y, cfg, par0; G=201, grid_min=-8, grid_max=8,
                           maxiter=500, verbose=false)

Exact ML on the ABB piecewise-uniform likelihood, with the marginal forward
filter evaluated on GPU.  Same interface as `estimate_exact_ml` from
exact_ml.jl.
"""
function estimate_exact_ml_gpu(y::Matrix{Float64}, cfg, par0;
                                G::Int = 201,
                                grid_min::Float64 = -8.0,
                                grid_max::Float64 = 8.0,
                                maxiter::Int = 500,
                                verbose::Bool = false)
    K, L = cfg.K, cfg.L
    par_work = copy_params(par0)
    v = params_to_vec(par0)

    function obj(v)
        vec_to_params!(par_work, v, K, L)
        exact_neg_loglik_gpu(par_work, y, cfg;
                              grid_min = grid_min, grid_max = grid_max,
                              G = G)
    end

    verbose && (@printf("  Exact ML (GPU): initial obj = %.6f\n", obj(v));
                flush(stdout))

    res = optimize(obj, v, LBFGS(),
                   Optim.Options(iterations = maxiter, g_tol = 1e-5,
                                 show_trace = verbose, show_every = 5))

    vec_to_params!(par_work, Optim.minimizer(res), K, L)
    par_work
end

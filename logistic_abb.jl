#=
logistic_abb.jl — Logistic density with ABB-style quantile knot parameterization

Same quantile knots as ABB:
  q_ℓ(η_{t-1}) = Σ_k a_{kℓ} He_k(η_{t-1}/σ_y),  ℓ = 1,2,3

Density: asymmetric logistic matching q₁, q₂, q₃ at τ = (0.25, 0.50, 0.75).
  Left of median (q₂): logistic with scale α_L = log(3)/(q₂ - q₁)
  Right of median (q₂): logistic with scale α_R = log(3)/(q₃ - q₂)

This is the "split logistic" or "asymmetric logistic":
  f(x) = 2 α_L α_R / (α_L + α_R) × logistic_pdf(x; q₂, α_L)  for x ≤ q₂
  f(x) = 2 α_L α_R / (α_L + α_R) × logistic_pdf(x; q₂, α_R)  for x > q₂

where logistic_pdf(x; μ, α) = α exp(-α(x-μ)) / (1 + exp(-α(x-μ)))²

Free parameters: ONLY a_Q (9 params for K=2, L=3), same as ABB.
No extra spline coefficients. The density shape is fully determined by the knots.

Smooth density → exact ML via forward filter with LBFGS.
=#

include("ABB_three_period.jl")
using Optim, Printf, LinearAlgebra, ForwardDiff

"""Generic pw_logdens for ForwardDiff compatibility."""
function pw_logdens_generic(x, q, tau, b1, bL)
    L = length(q)
    if x <= q[1]
        return log(max(tau[1] * b1, 1e-300)) + b1 * (x - q[1])
    end
    if x > q[L]
        return log(max((1 - tau[L]) * bL, 1e-300)) - bL * (x - q[L])
    end
    for l in 1:L-1
        if x <= q[l+1]
            dq = q[l+1] - q[l]
            return dq > 1e-12 ? log(tau[l+1] - tau[l]) - log(dq) : -700.0
        end
    end
    return -700.0
end

# ================================================================
#  ASYMMETRIC LOGISTIC DENSITY
# ================================================================

"""Standard logistic PDF: f(x; μ, α) = α exp(-α(x-μ)) / (1+exp(-α(x-μ)))²"""
function logistic_pdf(x, μ, α)
    z = α * (x - μ)
    ez = exp(-z)
    α * ez / (1 + ez)^2
end

function logistic_logpdf(x, μ, α)
    z = α * (x - μ)
    log(α) - z - 2 * log1p(exp(-z))
end

"""
Asymmetric logistic PDF matching quantile knots q₁ < q₂ < q₃ at τ=(0.25,0.5,0.75).
"""
function asym_logistic_logpdf(x, q1, q2, q3)
    gap_L = q2 - q1; gap_R = q3 - q2
    if gap_L <= 0 || gap_R <= 0
        return -1e10  # infeasible: knots not ordered
    end
    log3 = log(3.0)
    α_L = log3 / gap_L
    α_R = log3 / gap_R
    log_w = log(2.0) + log(α_L) + log(α_R) - log(α_L + α_R)
    if x <= q2
        return log_w + logistic_logpdf(x, q2, α_L)
    else
        return log_w + logistic_logpdf(x, q2, α_R)
    end
end

function asym_logistic_pdf(x, q1, q2, q3)
    exp(asym_logistic_logpdf(x, q1, q2, q3))
end

# ================================================================
#  TRANSITION DENSITY ON GRID (precomputed)
# ================================================================

"""
Compute conditional quantile knots using GAP REPARAMETERIZATION.
  a_Q[:,1] = Hermite coefficients for median q₂(η)
  a_Q[:,2] = Hermite coefficients for log(gap_L) where gap_L = q₂ - q₁
  a_Q[:,3] = Hermite coefficients for log(gap_R) where gap_R = q₃ - q₂
Non-crossing guaranteed: q₁ < q₂ < q₃ for all η.
"""
function gap_quantiles(eta_lag::Float64, a_Q::Matrix{Float64},
                       K::Int, sigma_y::Float64)
    z = eta_lag / sigma_y
    hv = zeros(K + 1)
    hv[1] = 1.0
    K >= 1 && (hv[2] = z)
    for k in 2:K; hv[k+1] = z * hv[k] - (k - 1) * hv[k-1]; end
    q2 = dot(view(a_Q, :, 1), hv)
    log_gapL = dot(view(a_Q, :, 2), hv)
    log_gapR = dot(view(a_Q, :, 3), hv)
    q1 = q2 - exp(log_gapL)
    q3 = q2 + exp(log_gapR)
    (q1, q2, q3)
end

"""
Precompute G×G transition matrix using asymmetric logistic with gap reparam.
Non-crossing guaranteed. Analytically normalized.
"""
function logistic_transition_matrix!(T_mat::Matrix{Float64},
                                     grid::Vector{Float64},
                                     par::Params, cfg::Config)
    G = length(grid)
    log3 = log(3.0)
    @inbounds for g1 in 1:G
        q1, q2, q3 = gap_quantiles(grid[g1], par.a_Q, cfg.K, cfg.sigma_y)
        gap_L = q2 - q1; gap_R = q3 - q2  # always positive by construction
        α_L = log3 / gap_L; α_R = log3 / gap_R
        C = 2.0 * α_L * α_R / (α_L + α_R)
        for g2 in 1:G
            T_mat[g1, g2] = asym_logistic_pdf(grid[g2], q1, q2, q3) / C
        end
    end
end

# ================================================================
#  EXACT LIKELIHOOD VIA FORWARD FILTER
# ================================================================

function logistic_neg_loglik(par::Params, y::Matrix{Float64}, cfg::Config;
                             grid_min::Float64=-8.0, grid_max::Float64=8.0,
                             G::Int=201)  # odd for Simpson's rule
    N, T = cfg.N, cfg.T
    G = isodd(G) ? G : G + 1  # Simpson needs odd number of points
    grid = collect(range(grid_min, grid_max, length=G))
    h = (grid_max - grid_min) / (G - 1)
    # Simpson's weights: [1, 4, 2, 4, 2, ..., 4, 1] × h/3
    sw = zeros(G)
    sw[1] = 1.0; sw[G] = 1.0
    for i in 2:G-1; sw[i] = iseven(i) ? 4.0 : 2.0; end
    sw .*= h / 3.0

    T_mat = zeros(G, G)
    logistic_transition_matrix!(T_mat, grid, par, cfg)

    # f_init and f_eps as NORMALIZED asymmetric logistic
    # asym_logistic_pdf returns unnorm density × w where w = 2αLαR/(αL+αR) = C
    # So asym_logistic_pdf/C = unnorm density (NOT normalized!)
    # The normalized density = unnorm / C_true where C_true = ∫unnorm dx
    # Since asym_logistic_pdf = unnorm × w and ∫(asym_logistic_pdf)dx = w × C_true,
    # and C_true = analytical = w, we have ∫asym_logistic_pdf dx = w².
    # Wait — let me just verify: asym_logistic_pdf = w × logistic_pdf (left or right)
    # ∫ w × logistic_pdf dx = w × 1 = w (for each half, ×1/2, sum = w)
    # Actually ∫_{-∞}^{q2} w × logistic_pdf(x,q2,αL) dx = w × 1/2
    # and ∫_{q2}^{∞} w × logistic_pdf(x,q2,αR) dx = w × 1/2
    # Total = w. So ∫ asym_logistic_pdf dx = w = 2αLαR/(αL+αR).
    # For normalized density: f_norm = asym_logistic_pdf / w.
    # That's just logistic_pdf on each half.

    a_init_s = par.a_init  # must be ordered
    log3 = log(3.0)
    gap_L_init = a_init_s[2] - a_init_s[1]; gap_R_init = a_init_s[3] - a_init_s[2]
    αL_init = log3 / gap_L_init; αR_init = log3 / gap_R_init
    C_init = 2.0 * αL_init * αR_init / (αL_init + αR_init)
    f_init = [asym_logistic_pdf(grid[g], a_init_s[1], a_init_s[2], a_init_s[3]) / C_init
              for g in 1:G]

    a_eps_s = sort(par.a_eps)
    gap_L_eps = a_eps_s[2] - a_eps_s[1]; gap_R_eps = a_eps_s[3] - a_eps_s[2]
    αL_eps = log3 / gap_L_eps; αR_eps = log3 / gap_R_eps
    C_eps = 2.0 * αL_eps * αR_eps / (αL_eps + αR_eps)

    p = zeros(G); p_new = zeros(G); eps_vec = zeros(G)
    total_ll = 0.0

    pw = zeros(G)
    for i in 1:N
        for g in 1:G
            eps_vec[g] = asym_logistic_pdf(y[i,1] - grid[g],
                                            a_eps_s[1], a_eps_s[2], a_eps_s[3]) / C_eps
            p[g] = f_init[g] * eps_vec[g]
        end
        L1 = dot(p, sw)
        L1 < 1e-300 && return Inf
        total_ll += log(L1); p ./= L1

        pw .= p .* sw
        mul!(p_new, transpose(T_mat), pw)
        for g in 1:G
            p_new[g] *= asym_logistic_pdf(y[i,2] - grid[g],
                                           a_eps_s[1], a_eps_s[2], a_eps_s[3]) / C_eps
        end
        L2 = dot(p_new, sw)
        L2 < 1e-300 && return Inf
        total_ll += log(L2); p_new ./= L2
        p, p_new = p_new, p

        if T >= 3
            pw .= p .* sw
            mul!(p_new, transpose(T_mat), pw)
            for g in 1:G
                p_new[g] *= asym_logistic_pdf(y[i,3] - grid[g],
                                               a_eps_s[1], a_eps_s[2], a_eps_s[3]) / C_eps
            end
            L3 = dot(p_new, sw)
            L3 < 1e-300 && return Inf
            total_ll += log(L3)
        end
    end
    -total_ll / N
end

# ================================================================
#  DATA GENERATION FROM ASYMMETRIC LOGISTIC MODEL
# ================================================================

"""
True parameters in GAP REPARAMETERIZATION for logistic AR(1).
  a_Q[:,1] = median coefficients [0, ρ, 0]
  a_Q[:,2] = log(gap_L) coefficients [log(gap), 0, 0]  (constant gap)
  a_Q[:,3] = log(gap_R) coefficients [log(gap), 0, 0]  (constant gap)
  gap = σ_v√3/π × log(3)  (logistic scale × log(3))
"""
function make_true_params_gap(; rho=0.8, sigma_v=0.5, sigma_eps=0.3,
                                sigma_eta1=1.0, K=2)
    # Logistic scale s = σ√3/π, gap = s × log(3)
    s_v = sigma_v * sqrt(3) / π
    s_eps = sigma_eps * sqrt(3) / π
    s_eta1 = sigma_eta1 * sqrt(3) / π
    log3 = log(3.0)

    a_Q = zeros(K + 1, 3)
    a_Q[1, 1] = 0.0      # median intercept
    a_Q[2, 1] = rho       # median slope
    a_Q[1, 2] = log(s_v * log3)  # log(gap_L), constant
    a_Q[1, 3] = log(s_v * log3)  # log(gap_R), constant

    # Marginal quantiles (ordered): q₁ = -s×log(3), q₂ = 0, q₃ = s×log(3)
    a_init = [-s_eta1 * log3, 0.0, s_eta1 * log3]
    a_eps = [-s_eps * log3, 0.0, s_eps * log3]

    Params(a_Q, 0.0, 0.0,   # tail rates unused
           a_init, 0.0, 0.0,
           a_eps, 0.0, 0.0)
end

function asym_logistic_draw(rng::AbstractRNG, q1::Float64, q2::Float64, q3::Float64)
    gap_L = q2 - q1; gap_R = q3 - q2
    log3 = log(3.0)
    α_L = log3 / gap_L; α_R = log3 / gap_R
    u = rand(rng)
    # CDF of asymmetric logistic:
    # For x ≤ q₂: F(x) = α_R/(α_L+α_R) × 2/(1+exp(-α_L(x-q₂)))
    # For x > q₂: F(x) = 1 - α_L/(α_L+α_R) × 2/(1+exp(α_R(x-q₂)))
    # At q₂: F(q₂) = α_R/(α_L+α_R) = 0.5 only if α_L=α_R
    # For asymmetric: F(q₂) = α_R/(α_L+α_R)
    F_med = α_R / (α_L + α_R)
    if u <= F_med
        # x ≤ q₂: solve α_R/(α_L+α_R) × 2σ(α_L(x-q₂)) = u
        # σ(z) = u(α_L+α_R)/(2α_R)
        s = u * (α_L + α_R) / (2α_R)
        # σ(z) = s → z = log(s/(1-s)) → x = q₂ + log(s/(1-s))/α_L
        s = clamp(s, 1e-15, 1-1e-15)
        return q2 + log(s / (1 - s)) / α_L
    else
        # x > q₂: solve 1 - α_L/(α_L+α_R) × 2(1-σ(α_R(x-q₂))) = u
        # 1 - u = α_L/(α_L+α_R) × 2(1-σ(α_R(x-q₂)))
        # 1 - σ(z) = (1-u)(α_L+α_R)/(2α_L)
        s = (1 - u) * (α_L + α_R) / (2α_L)
        s = clamp(s, 1e-15, 1-1e-15)
        # 1 - σ(z) = s → σ(z) = 1-s → z = log((1-s)/s) → x = q₂ + log((1-s)/s)/α_R
        return q2 + log((1 - s) / s) / α_R
    end
end

"""
Generate data using gap reparameterization for transition.
a_init and a_eps: [q₁, q₂, q₃] directly (ordered, no gap reparam needed for marginals).
a_Q: gap reparam — col 1 = median, col 2 = log gap_L, col 3 = log gap_R.
"""
function generate_data_logistic_abb(N::Int, par::Params,
                                    sigma_y::Float64, K::Int; seed::Int=42)
    rng = MersenneTwister(seed); T = 3
    eta = zeros(N, T); y = zeros(N, T)
    for i in 1:N
        eta[i, 1] = asym_logistic_draw(rng, par.a_init[1], par.a_init[2], par.a_init[3])
    end
    for t in 2:T, i in 1:N
        q1, q2, q3 = gap_quantiles(eta[i, t-1], par.a_Q, K, sigma_y)
        eta[i, t] = asym_logistic_draw(rng, q1, q2, q3)
    end
    for t in 1:T, i in 1:N
        y[i, t] = eta[i, t] + asym_logistic_draw(rng, par.a_eps[1], par.a_eps[2], par.a_eps[3])
    end
    y, eta
end

# ================================================================
#  MLE ESTIMATION (LBFGS)
# ================================================================

"""
Neg-loglik as a pure function of parameter vector v (for ForwardDiff).
All data and config captured in closure; v is the only argument.
"""
function make_logistic_obj(y::Matrix{Float64}, cfg::Config;
                            G::Int=120, grid_min::Float64=-6.0, grid_max::Float64=6.0)
    N, T, K, L = cfg.N, cfg.T, cfg.K, cfg.L
    grid = collect(range(grid_min, grid_max, length=G))
    dgrid = (grid_max - grid_min) / (G - 1)

    function obj(v::AbstractVector{TT}) where TT
        np = (K + 1) * L
        a_Q = reshape(v[1:np], K + 1, L)
        i = np
        a_init = v[i+1:i+L]; i += L
        a_eps_fd = [v[i+1], zero(TT), v[i+2]]  # a_eps[2] = 0 (median fixed)

        tau = TT.(cfg.tau)

        # Transition matrix with normalization
        T_mat = zeros(TT, G, G)
        q_buf = zeros(TT, L)
        for g1 in 1:G
            # Compute quantile knots at this grid point
            z = grid[g1] / cfg.sigma_y
            hv = zeros(TT, K + 1)
            hv[1] = one(TT)
            K >= 1 && (hv[2] = TT(z))
            for k in 2:K; hv[k+1] = TT(z) * hv[k] - TT(k - 1) * hv[k-1]; end
            for l in 1:L
                q_buf[l] = sum(a_Q[k, l] * hv[k] for k in 1:K+1)
            end
            # Gap reparameterization: col1=median, col2=log_gap_L, col3=log_gap_R
            q2_fd = q_buf[1]
            q1_fd = q2_fd - exp(q_buf[2])
            q3_fd = q2_fd + exp(q_buf[3])

            for g2 in 1:G
                T_mat[g1, g2] = asym_logistic_pdf(TT(grid[g2]), q1_fd, q2_fd, q3_fd)
            end
            # Normalize row
            C = sum(T_mat[g1, :]) * TT(dgrid)
            if C > TT(1e-300)
                T_mat[g1, :] ./= C
            end
        end

        # f_init on grid (asymmetric logistic, NOT piecewise-uniform)
        f_init = zeros(TT, G)
        ai_s = sort(collect(a_init))
        C_init_fd = let gL = ai_s[2]-ai_s[1], gR = ai_s[3]-ai_s[2]
            aL = log(TT(3))/gL; aR = log(TT(3))/gR; TT(2)*aL*aR/(aL+aR)
        end
        for g in 1:G
            f_init[g] = asym_logistic_pdf(TT(grid[g]), ai_s[1], ai_s[2], ai_s[3]) / C_init_fd
        end

        # Forward filter
        p = zeros(TT, G)
        p_new = zeros(TT, G)
        total_ll = zero(TT)

        for i_obs in 1:N
            for g in 1:G
                eps_val = TT(y[i_obs, 1]) - TT(grid[g])
                p[g] = f_init[g] * let ae_s = sort(collect(a_eps_fd))
                    Ce = let gL=ae_s[2]-ae_s[1], gR=ae_s[3]-ae_s[2]
                        aL=log(TT(3))/gL; aR=log(TT(3))/gR; TT(2)*aL*aR/(aL+aR)
                    end
                    asym_logistic_pdf(eps_val, ae_s[1], ae_s[2], ae_s[3]) / Ce
                end
            end
            L1 = sum(p) * TT(dgrid)
            L1 < TT(1e-300) && return TT(1e10)
            total_ll += log(L1)
            p ./= L1

            # Step 2
            for g2 in 1:G
                p_new[g2] = sum(T_mat[g1, g2] * p[g1] for g1 in 1:G) * TT(dgrid)
            end
            for g in 1:G
                eps_val = TT(y[i_obs, 2]) - TT(grid[g])
                p_new[g] *= let ae_s = sort(collect(a_eps_fd))
                    Ce = let gL=ae_s[2]-ae_s[1], gR=ae_s[3]-ae_s[2]
                        aL=log(TT(3))/gL; aR=log(TT(3))/gR; TT(2)*aL*aR/(aL+aR)
                    end
                    asym_logistic_pdf(eps_val, ae_s[1], ae_s[2], ae_s[3]) / Ce
                end
            end
            L2 = sum(p_new) * TT(dgrid)
            L2 < TT(1e-300) && return TT(1e10)
            total_ll += log(L2)
            p_new ./= L2
            p, p_new = p_new, p

            # Step 3
            if T >= 3
                for g2 in 1:G
                    p_new[g2] = sum(T_mat[g1, g2] * p[g1] for g1 in 1:G) * TT(dgrid)
                end
                for g in 1:G
                    eps_val = TT(y[i_obs, 3]) - TT(grid[g])
                    p_new[g] *= let ae_s = sort(collect(a_eps_fd))
                    Ce = let gL=ae_s[2]-ae_s[1], gR=ae_s[3]-ae_s[2]
                        aL=log(TT(3))/gL; aR=log(TT(3))/gR; TT(2)*aL*aR/(aL+aR)
                    end
                    asym_logistic_pdf(eps_val, ae_s[1], ae_s[2], ae_s[3]) / Ce
                end
                end
                L3 = sum(p_new) * TT(dgrid)
                L3 < TT(1e-300) && return TT(1e10)
                total_ll += log(L3)
            end
        end
        -total_ll / TT(N)
    end
    obj
end

function estimate_logistic_abb_ml(y::Matrix{Float64}, cfg::Config, par0::Params;
                                  G::Int=201, maxiter::Int=100, verbose::Bool=true)
    K, L = cfg.K, cfg.L
    par = copy_params(par0)

    function obj(v)
        vec_to_params_la!(par, v, K, L)
        logistic_neg_loglik(par, y, cfg; G=G)
    end

    v0 = params_to_vec_la(par0)
    verbose && @printf("  Logistic-ABB ML initial obj: %.6f\n", obj(v0))

    # LBFGS with Optim.jl's built-in finite-difference gradient
    res = optimize(obj, v0, LBFGS(),
                   Optim.Options(iterations=maxiter, g_tol=1e-5,
                                 show_trace=verbose, show_every=10))
    v_opt = Optim.minimizer(res)
    vec_to_params_la!(par, v_opt, K, L)
    @printf("  Logistic-ABB ML final obj: %.6f (iters=%d)\n",
            Optim.minimum(res), Optim.iterations(res))
    par
end

# Parameter packing — only params used by logistic likelihood
# a_Q (9), a_init (3), a_eps[1] and a_eps[3] (2) = 14 total
# a_eps[2] fixed at 0 for identification (ε median = 0)
# Tail rates NOT needed — determined by knot gaps
function params_to_vec_la(par::Params)
    v = Float64[]
    append!(v, vec(par.a_Q))  # 9
    append!(v, par.a_init)    # 3
    push!(v, par.a_eps[1])    # 1
    push!(v, par.a_eps[3])    # 1
    v                         # total 14
end

function vec_to_params_la!(par::Params, v::Vector{Float64}, K::Int, L::Int)
    np = (K + 1) * L
    par.a_Q .= reshape(view(v, 1:np), K + 1, L)
    par.a_init .= view(v, np+1:np+L)
    par.a_eps[1] = v[np+L+1]
    par.a_eps[2] = 0.0        # fixed: ε median = 0
    par.a_eps[3] = v[np+L+2]
end

# ================================================================
#  TEST
# ================================================================

function test_logistic_abb()
    K = 2; L = 3; sigma_y = 1.0; T = 3; N = 500
    tau = [0.25, 0.50, 0.75]
    par_true = make_true_params_linear(tau=tau, sigma_y=sigma_y, K=K)
    cfg = Config(N, T, K, L, tau, sigma_y, 1, 1, 1, fill(0.05, T))

    println("="^60)
    println("  LOGISTIC-ABB TEST")
    println("="^60)

    # Verify density integrates to 1
    q = [-0.337, 0.0, 0.337]
    dx = 0.001; xgrid = collect(-5.0:dx:5.0)
    total = sum(asym_logistic_pdf(x, q[1], q[2], q[3]) for x in xgrid) * dx
    @printf("Density integral: %.6f (should be ~1)\n", total)

    # Verify quantiles
    samples = [asym_logistic_draw(MersenneTwister(i), q[1], q[2], q[3]) for i in 1:100000]
    emp_q = quantile(samples, tau)
    @printf("Empirical quantiles: [%.4f, %.4f, %.4f] (true [%.4f, %.4f, %.4f])\n",
            emp_q..., q...)

    # Generate data from logistic-ABB model
    println("\nGenerating data (N=$N)...")
    y, eta = generate_data_logistic_abb(N, par_true, sigma_y, K; seed=42)
    @printf("η range: [%.3f, %.3f]\n", extrema(eta)...)

    nll_truth = logistic_neg_loglik(par_true, y, cfg; G=120)
    @printf("neg-ll at truth: %.6f\n", nll_truth)

    # QR warm start
    println("\nQR warm start...")
    par0 = init_params(y, cfg)
    eta_all = zeros(N, T, 20)
    for m in 1:20; eta_all[:,:,m] .= 0.6 .* y; end
    cfg_em = Config(N, T, K, L, tau, sigma_y, 10, 100, 20, fill(0.05, T))
    for _ in 1:10
        e_step!(eta_all, y, par0, cfg_em)
        m_step_qr!(par0, eta_all, y, cfg_em)
    end
    @printf("QR slopes: [%.4f, %.4f, %.4f]\n", par0.a_Q[2,:]...)

    # MLE
    println("\nLogistic-ABB MLE...")
    t = @elapsed par_ml = estimate_logistic_abb_ml(y, cfg, par0; G=120,
                                                     maxiter=100, verbose=true)

    println("\n==== RESULTS ====")
    @printf("True slopes:  [%.4f, %.4f, %.4f]\n", par_true.a_Q[2,:]...)
    @printf("QR slopes:    [%.4f, %.4f, %.4f]\n", par0.a_Q[2,:]...)
    @printf("ML slopes:    [%.4f, %.4f, %.4f]\n", par_ml.a_Q[2,:]...)
    @printf("True intcpts: [%.4f, %.4f, %.4f]\n", par_true.a_Q[1,:]...)
    @printf("QR intcpts:   [%.4f, %.4f, %.4f]\n", par0.a_Q[1,:]...)
    @printf("ML intcpts:   [%.4f, %.4f, %.4f]\n", par_ml.a_Q[1,:]...)
    @printf("Time: %.1fs\n", t)
end

if abspath(PROGRAM_FILE) == @__FILE__
    test_logistic_abb()
end

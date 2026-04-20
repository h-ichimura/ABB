#=
misspecified_ar2.jl — Compare MLE vs QR density approximation
under AR(2) + χ²(3) DGP (both models completely misspecified).

DGP:
  η_t = ρ₁ η_{t-1} + ρ₂ η_{t-2} + v_t
  v_t ~ σ_v × (χ²(3) - 3) / √6   (zero mean, asymmetric, heavy-tailed)
  ε_t ~ σ_ε × (χ²(3) - 3) / √6   (same distribution for measurement error)
  y_t = η_t + ε_t

Estimation model: AR(1) cubic spline (misspecified: wrong dynamics + wrong density shape)
Both QR and MLE assume AR(1) — neither can capture the AR(2) dependence.

True f(y_t | y_{t-1}) computed by Monte Carlo (no shortcuts).
=#

include("cspline_abb.jl")
using Printf, Statistics, Random, Distributions

# ================================================================
#  AR(2) + χ² DGP
# ================================================================

function draw_chi2_centered(rng::AbstractRNG, df::Int, σ::Float64)
    raw = rand(rng, Chisq(df))
    σ * (raw - df) / sqrt(2.0 * df)
end

function generate_ar2_data(N::Int, T::Int;
                            ρ1::Float64=0.5, ρ2::Float64=0.3,
                            σ_v::Float64=0.5, σ_ε::Float64=0.3,
                            df_v::Int=3, df_ε::Int=3,
                            seed::Int=42, burnin::Int=200)
    rng = MersenneTwister(seed)
    # Need T+1 η values to have T periods of y with AR(2)
    eta = zeros(N, T); y = zeros(N, T)

    for i in 1:N
        # Burn-in
        η_prev2 = 0.0; η_prev1 = 0.0
        for b in 1:burnin
            v = draw_chi2_centered(rng, df_v, σ_v)
            η_new = ρ1 * η_prev1 + ρ2 * η_prev2 + v
            η_prev2 = η_prev1; η_prev1 = η_new
        end
        eta[i,1] = η_prev1

        # For t=2, need η_{t-2} — store previous
        η_lag2 = η_prev2
        for t in 2:T
            v = draw_chi2_centered(rng, df_v, σ_v)
            eta[i,t] = ρ1 * eta[i,t-1] + ρ2 * η_lag2 + v
            η_lag2 = eta[i,t-1]
        end
    end

    for t in 1:T, i in 1:N
        y[i,t] = eta[i,t] + draw_chi2_centered(rng, df_ε, σ_ε)
    end

    y, eta
end

# ================================================================
#  TRUE f(y_t | y_{t-1}) BY MONTE CARLO
#
#  Generate large sample from DGP, bin by y_{t-1}, build empirical
#  conditional density of y_t. No model assumptions.
# ================================================================

function true_conditional_density_empirical(y_grid::Vector{Float64},
                                             y_lag_target::Float64,
                                             y_data::Matrix{Float64};
                                             bandwidth_y_lag::Float64=0.3,
                                             kernel_bw::Float64=0.0)
    N, T = size(y_data)
    ny = length(y_grid)
    dy = y_grid[2] - y_grid[1]

    # Collect y_t values where y_{t-1} is close to y_lag_target
    y_t_samples = Float64[]
    for t in 2:T, i in 1:N
        if abs(y_data[i, t-1] - y_lag_target) < bandwidth_y_lag
            push!(y_t_samples, y_data[i, t])
        end
    end

    n_samples = length(y_t_samples)
    n_samples < 50 && (@printf("  WARNING: only %d samples near y_lag=%.2f\n", n_samples, y_lag_target))

    # Kernel density estimate
    h = kernel_bw > 0 ? kernel_bw : 1.06 * std(y_t_samples) * n_samples^(-0.2)

    f = zeros(ny)
    for (iy, yv) in enumerate(y_grid)
        for ys in y_t_samples
            z = (yv - ys) / h
            f[iy] += exp(-0.5 * z^2) / (h * sqrt(2π))
        end
        f[iy] /= n_samples
    end

    # Normalize
    total = sum(f) * dy
    total > 0 && (f ./= total)
    f, n_samples
end

# ================================================================
#  ESTIMATED f(y_t | y_{t-1}) FROM MODEL
# ================================================================

function estimated_cond_density(y_grid::Vector{Float64}, y_lag::Float64,
                                 a_Q::Matrix{Float64}, M_Q::Float64,
                                 a_eps1::Float64, a_eps3::Float64, M_eps::Float64,
                                 K::Int, σy::Float64, τ::Vector{Float64}; G::Int=201)
    ws = CSplineWorkspace(G, K)
    nk = K + 1
    grid = ws.grid[1:G]; sw = ws.sw[1:G]

    # Transition matrix
    cspline_transition_matrix!(ws.T_mat, ws.grid, G, a_Q, M_Q, K, σy, τ,
                               ws.hv_buf, ws.t_buf, ws.s_buf, ws.masses_buf, ws.c1buf)

    # Eps density
    a_eps_s = [a_eps1, 0.0, a_eps3]
    s_eps = zeros(3); βLe=Ref(0.0); βRe=Ref(0.0); κ1e=Ref(0.0); κ3e=Ref(0.0)
    solve_cspline_c2!(s_eps, βLe, βRe, κ1e, κ3e, a_eps_s, τ, M_eps)
    lr_eps = max(s_eps[1], s_eps[2], s_eps[3])
    m_eps = zeros(4)
    cspline_masses!(m_eps, a_eps_s, s_eps, βLe[], βRe[], κ1e[], κ3e[], lr_eps)
    C_eps = sum(m_eps)

    # Stationary distribution by power iteration
    T_mat = view(ws.T_mat, 1:G, 1:G)
    p_stat = ones(G) / G; p_new = zeros(G)
    for iter in 1:50
        pw = p_stat .* sw
        mul!(view(p_new, 1:G), transpose(T_mat), pw)
        L = dot(view(p_new, 1:G), sw); L > 0 && (p_stat .= p_new ./ L)
    end

    # p(η_{t-1} | y_{t-1}) ∝ p_stat(η) × f_ε(y_lag - η)
    p_cond = zeros(G)
    for g in 1:G
        log_fe = cspline_eval(y_lag - grid[g], a_eps_s, s_eps, βLe[], βRe[], κ1e[], κ3e[]) - lr_eps
        p_cond[g] = p_stat[g] * exp(log_fe) / C_eps
    end
    Z = dot(p_cond, sw); Z > 0 && (p_cond ./= Z)

    # f(y_t | y_{t-1}) = Σ_{g1} p(g1|y_lag) Σ_{g2} T(g1,g2) f_ε(y-g2) sw(g2) × sw(g1)
    ny = length(y_grid); dy = y_grid[2] - y_grid[1]
    f = zeros(ny)
    for (iy, yv) in enumerate(y_grid)
        val = 0.0
        for g1 in 1:G
            p_cond[g1] < 1e-300 && continue
            inner = 0.0
            for g2 in 1:G
                log_fe = cspline_eval(yv - grid[g2], a_eps_s, s_eps, βLe[], βRe[], κ1e[], κ3e[]) - lr_eps
                inner += ws.T_mat[g1, g2] * exp(log_fe) / C_eps * sw[g2]
            end
            val += p_cond[g1] * inner * sw[g1]
        end
        f[iy] = val
    end
    total = sum(f) * dy; total > 0 && (f ./= total)
    f
end

# ================================================================
#  MAIN
# ================================================================

function run_ar2_comparison(; ρ1::Float64=0.5, ρ2::Float64=0.3,
                              N::Int=50000, T::Int=5, seed::Int=42)
    K = 2; σy = 1.0; τ = [0.25, 0.50, 0.75]
    σ_v = 0.5; σ_ε = 0.3

    println("=" ^ 70)
    @printf("AR(2) + χ²(3) DGP: ρ₁=%.2f, ρ₂=%.2f, N=%d, T=%d\n", ρ1, ρ2, N, T)
    @printf("  v ~ σ_v×(χ²(3)-3)/√6, ε ~ σ_ε×(χ²(3)-3)/√6\n")
    @printf("  Both MLE and QR assume AR(1) — completely misspecified dynamics\n")
    println("=" ^ 70)
    println()

    # Generate data (large N for clean empirical conditional density)
    y, eta = generate_ar2_data(N, T; ρ1=ρ1, ρ2=ρ2, σ_v=σ_v, σ_ε=σ_ε, seed=seed)
    @printf("Data: y range [%.2f, %.2f], η range [%.2f, %.2f]\n",
            extrema(y)..., extrema(eta)...)
    @printf("  η std=%.3f, ε std=%.3f, y std=%.3f\n",
            std(eta), std(y .- eta), std(y))
    @printf("  Skewness of v: %.3f (χ² is right-skewed)\n",
            mean(((eta[:,2] .- ρ1.*eta[:,1]) ./ σ_v).^3))
    println()

    # Use subset for estimation (first N_est individuals)
    N_est = min(N, 5000)
    y_est = y[1:N_est, :]

    # ---- Estimate with profiled MLE ----
    # Initial values from approximate ρ
    ρ_approx = ρ1 + ρ2  # AR(2) → approximate AR(1) persistence
    tp_init = make_true_cspline(rho=ρ_approx, sigma_v=σ_v, sigma_eps=σ_ε)
    v_prof = pack_profiled(tp_init.a_Q, tp_init.a_init, tp_init.a_eps1, tp_init.a_eps3)

    @printf("Estimating profiled MLE (N=%d)...\n", N_est); flush(stdout)
    v_ml, nll_ml = estimate_profiled_ml(y_est, K, σy, v_prof, τ; G=201, maxiter=500)
    aQ_ml, MQ_ml, ai_ml, Mi_ml, ae1_ml, ae3_ml, Me_ml = unpack_profiled(v_ml, K)
    @printf("  MLE: ρ=%.4f  ae3=%.4f  M_Q=%.2f  M_eps=%.2f\n",
            aQ_ml[2,2], ae3_ml, MQ_ml, Me_ml)

    # ---- Estimate with QR ----
    @printf("Estimating QR (N=%d)...\n", N_est); flush(stdout)
    qr = estimate_cspline_qr(y_est, K, σy, tp_init.a_Q, tp_init.a_init,
                               tp_init.a_eps1, tp_init.a_eps3, τ;
                               G=201, S_em=30, M_draws=10, verbose=false, seed=seed)
    MQ_qr = _M_from_iqr(qr.a_Q[1,3] - qr.a_Q[1,1])
    Me_qr = qr.M_eps
    @printf("  QR:  ρ=%.4f  ae3=%.4f  M_Q=%.2f  M_eps=%.2f\n\n",
            qr.a_Q[2,2], qr.a_eps3, MQ_qr, Me_qr)

    # ---- Compare conditional densities ----
    # Use FULL dataset (N=50000) for empirical true density
    y_lags = quantile(y[:,1], [0.1, 0.25, 0.5, 0.75, 0.9])
    y_eval = collect(range(-4.0, 4.0, length=500))
    dy = y_eval[2] - y_eval[1]

    @printf("Conditional density f(y_t | y_{t-1}) comparison:\n")
    @printf("%-12s  %6s  %10s  %10s  %10s  %10s\n",
            "y_{t-1}", "n_obs", "KS_ML", "KS_QR", "L1_ML", "L1_QR")
    println("-" ^ 66)

    for y_lag in y_lags
        # True: empirical from full dataset
        f_true, n_obs = true_conditional_density_empirical(y_eval, y_lag, y;
                                                            bandwidth_y_lag=0.2)

        # MLE estimated
        f_ml = estimated_cond_density(y_eval, y_lag, aQ_ml, MQ_ml,
                                       ae1_ml, ae3_ml, Me_ml, K, σy, τ)

        # QR estimated
        f_qr = estimated_cond_density(y_eval, y_lag, qr.a_Q, MQ_qr,
                                       qr.a_eps1, qr.a_eps3, Me_qr, K, σy, τ)

        # KS
        cdf_true = cumsum(f_true) * dy
        cdf_ml = cumsum(f_ml) * dy
        cdf_qr = cumsum(f_qr) * dy
        ks_ml = maximum(abs.(cdf_ml .- cdf_true))
        ks_qr = maximum(abs.(cdf_qr .- cdf_true))

        # L1
        l1_ml = sum(abs.(f_ml .- f_true)) * dy
        l1_qr = sum(abs.(f_qr .- f_true)) * dy

        winner = ks_ml < ks_qr ? "MLE" : "QR"
        @printf("y=%+7.3f  %6d  %10.4f  %10.4f  %10.4f  %10.4f  %s\n",
                y_lag, n_obs, ks_ml, ks_qr, l1_ml, l1_qr, winner)
    end
    println()
    flush(stdout)
end

# Run with GP-style calibration: unit root with mean reversion
for T in [3, 5]
    run_ar2_comparison(ρ1=1.0, ρ2=-0.2, N=50000, T=T, seed=42)
    println()
end

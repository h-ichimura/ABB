#=
misspecified_comparison_v2.jl — Compare MLE vs QR density approximation
under genuinely non-Gaussian, nonlinear DGP.

DGP (switching persistence, ABB supplement eq S7):
  η_t = ρ(η_{t-1}, v_t) × η_{t-1} + v_t
  ρ(η, v) = 1 - δ × 1{|η| > d} × 1{|v| > b}
  v_t ~ t(df=5) × scale  (fat-tailed, not Gaussian)
  ε_t ~ σ_ε × (χ²(3) - 3) / √6  (skewed, zero mean)
  y_t = η_t + ε_t

Both QR and MLE are misspecified:
  - QR: polynomial quantile function can't capture switching ρ
  - MLE: cubic spline can't capture non-smooth transition or skewed ε

Comparison: f(y_t | y_{t-1}) by MC (true) vs forward filter (estimated)
Metrics: KS distance, L1 distance
=#

include("cspline_abb.jl")
using Printf, Statistics, Random, Distributions

# ================================================================
#  SWITCHING-PERSISTENCE DGP
# ================================================================

struct SwitchingDGP
    ρ_base::Float64    # base persistence (e.g., 1.0 for random walk)
    δ::Float64         # persistence drop for unusual shocks (e.g., 0.2)
    d::Float64         # threshold for η (switching occurs when |η| > d)
    b::Float64         # threshold for v (switching occurs when |v| > b)
    df_v::Int          # degrees of freedom for t-distributed v
    σ_v::Float64       # scale of v shocks
    df_eps::Int        # degrees of freedom for chi-squared ε
    σ_ε::Float64       # scale of ε
end

function default_dgp(; ρ_base=0.8)
    # Calibrate d, b so switching probability ≈ 15%
    # P(|η| > d) × P(|v| > b) ≈ 0.15
    # With stationary σ_η ≈ σ_v/√(1-ρ²) ≈ 0.5/0.6 ≈ 0.83
    # P(|η/0.83| > d/0.83) ≈ 0.5 → d ≈ 0.56
    # P(|v| > b) ≈ 0.30 → b ≈ 0.53 for t(5) with scale 0.5
    SwitchingDGP(ρ_base, 0.2, 0.56, 0.35, 5, 0.5, 3, 0.3)
end

function ρ_switch(η_lag::Float64, v::Float64, dgp::SwitchingDGP)
    if abs(η_lag) > dgp.d && abs(v) > dgp.b
        return dgp.ρ_base - dgp.δ
    end
    dgp.ρ_base
end

function draw_v(rng::AbstractRNG, dgp::SwitchingDGP)
    # t-distributed: scale so that std ≈ σ_v
    # Var(t(df)) = df/(df-2), so scale = σ_v × √((df-2)/df)
    scale = dgp.σ_v * sqrt((dgp.df_v - 2) / dgp.df_v)
    scale * rand(rng, TDist(dgp.df_v))
end

function draw_ε(rng::AbstractRNG, dgp::SwitchingDGP)
    # Chi-squared(df) shifted to zero mean, scaled to σ_ε
    # χ²(df) has mean df, variance 2df
    # (χ² - df) / √(2df) has mean 0, variance 1
    raw = rand(rng, Chisq(dgp.df_eps))
    dgp.σ_ε * (raw - dgp.df_eps) / sqrt(2 * dgp.df_eps)
end

function generate_switching_data(N::Int, T::Int, dgp::SwitchingDGP;
                                  seed::Int=42, burnin::Int=200)
    rng = MersenneTwister(seed)
    eta = zeros(N, T); y = zeros(N, T)

    for i in 1:N
        # Burn-in for stationary distribution
        η = 0.0
        for b in 1:burnin
            v = draw_v(rng, dgp)
            ρ = ρ_switch(η, v, dgp)
            η = ρ * η + v
        end
        eta[i,1] = η

        for t in 2:T
            v = draw_v(rng, dgp)
            ρ = ρ_switch(eta[i,t-1], v, dgp)
            eta[i,t] = ρ * eta[i,t-1] + v
        end
    end

    for t in 1:T, i in 1:N
        y[i,t] = eta[i,t] + draw_ε(rng, dgp)
    end

    y, eta
end

# ================================================================
#  TRUE f(y_t | y_{t-1}) BY MONTE CARLO
#
#  For each conditioning value y_{t-1}, simulate R paths:
#    1. Draw η_{t-1} from p(η_{t-1} | y_{t-1}) by importance sampling
#    2. Draw v_t from t(df), compute η_t = ρ(η_{t-1},v_t)η_{t-1} + v_t
#    3. Draw ε_t, compute y_t = η_t + ε_t
#    4. Build histogram/density of y_t
#
#  p(η_{t-1} | y_{t-1}) ∝ f_η(η) × f_ε(y_{t-1} - η)
#  Use importance sampling with Gaussian proposal centered at y_{t-1}
# ================================================================

function true_conditional_density_mc(y_grid::Vector{Float64}, y_lag::Float64,
                                      dgp::SwitchingDGP;
                                      R::Int=100000, seed::Int=12345)
    rng = MersenneTwister(seed)
    ny = length(y_grid)
    dy = y_grid[2] - y_grid[1]

    # Step 1: Sample η_{t-1} from p(η | y_lag) by rejection/importance sampling
    # p(η | y) ∝ f_stationary(η) × f_ε(y - η)
    # Proposal: N(y_lag, σ²_proposal) where σ² ≈ σ²_η (stationary variance)
    σ_η_approx = dgp.σ_v / sqrt(max(1 - dgp.ρ_base^2, 0.01))
    σ_prop = σ_η_approx * 0.8  # slightly narrower than stationary

    # Generate weighted samples of y_t
    y_samples = zeros(R)
    weights = ones(R)

    for r in 1:R
        # Draw η_{t-1} from proposal N(y_lag × K, σ_prop) where K is Kalman gain
        K = σ_η_approx^2 / (σ_η_approx^2 + dgp.σ_ε^2)
        η_mean = K * y_lag
        η_lag = η_mean + σ_prop * randn(rng)

        # Importance weight: f_stationary(η) × f_ε(y_lag - η) / proposal(η)
        # f_ε is chi-squared based
        eps_val = y_lag - η_lag
        # f_ε(eps): density of σ_ε × (χ²(df) - df) / √(2df)
        # Transform: x = eps/σ_ε × √(2df) + df → χ²(df) density at x
        x_chi = eps_val / dgp.σ_ε * sqrt(2 * dgp.df_eps) + dgp.df_eps
        if x_chi > 0
            log_f_eps = logpdf(Chisq(dgp.df_eps), x_chi) +
                        log(sqrt(2 * dgp.df_eps) / dgp.σ_ε)
        else
            log_f_eps = -1e10
        end

        # f_stationary(η): approximate as N(0, σ²_η)
        log_f_eta = -0.5 * (η_lag / σ_η_approx)^2 - log(σ_η_approx * sqrt(2π))

        # proposal: N(η_mean, σ_prop²)
        log_prop = -0.5 * ((η_lag - η_mean) / σ_prop)^2 - log(σ_prop * sqrt(2π))

        log_w = log_f_eta + log_f_eps - log_prop
        weights[r] = exp(log_w)

        # Step 2: Draw v_t and compute η_t
        v = draw_v(rng, dgp)
        ρ = ρ_switch(η_lag, v, dgp)
        η_t = ρ * η_lag + v

        # Step 3: Draw ε_t and compute y_t
        y_samples[r] = η_t + draw_ε(rng, dgp)
    end

    # Normalize weights
    weights ./= sum(weights)

    # Step 4: Build weighted kernel density estimate
    # Use Gaussian kernel with bandwidth h
    h = 1.06 * std(y_samples) * R^(-0.2)  # Silverman's rule
    f = zeros(ny)
    for (iy, yv) in enumerate(y_grid)
        for r in 1:R
            z = (yv - y_samples[r]) / h
            f[iy] += weights[r] * exp(-0.5 * z^2) / (h * sqrt(2π))
        end
    end

    # Normalize
    total = sum(f) * dy
    total > 0 && (f ./= total)
    f
end

# ================================================================
#  ESTIMATED f(y_t | y_{t-1}) FROM FORWARD FILTER
# ================================================================

function estimated_conditional_density(y_grid::Vector{Float64}, y_lag::Float64,
                                        a_Q::Matrix{Float64}, M_Q::Float64,
                                        a_eps1::Float64, a_eps3::Float64, M_eps::Float64,
                                        K::Int, σy::Float64, τ::Vector{Float64};
                                        G::Int=201)
    # Use the forward filter approach:
    # f(y_t | y_{t-1}) = ∫∫ f_trans(η_t|η_{t-1}) f_ε(y_t-η_t) p(η_{t-1}|y_{t-1}) dη_t dη_{t-1}
    #
    # Approximate by discretizing on η-grid:
    # 1. Compute p(η_{t-1} | y_{t-1}) ∝ f_init(η) × f_ε(y_{t-1} - η) on grid
    # 2. For each y in y_grid: f(y|y_lag) = Σ_{g1,g2} T(g1,g2) f_ε(y-g2) p(g1|y_lag) sw(g1) sw(g2)

    ws = CSplineWorkspace(G, K)
    nk = K + 1
    grid = ws.grid[1:G]
    sw = ws.sw[1:G]

    # Build transition matrix
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

    # Init/marginal density on grid (use stationary from transition)
    # Power iteration for stationary distribution
    p_stat = ones(G) / G
    p_new = zeros(G)
    T_mat = view(ws.T_mat, 1:G, 1:G)
    for iter in 1:50
        pw = p_stat .* sw
        mul!(view(p_new, 1:G), transpose(T_mat), pw)
        L = dot(view(p_new, 1:G), sw)
        L > 0 && (p_stat .= p_new ./ L)
    end

    # p(η_{t-1} | y_{t-1}) ∝ p_stat(η) × f_ε(y_lag - η)
    p_cond = zeros(G)
    for g in 1:G
        log_fe = cspline_eval(y_lag - grid[g], a_eps_s, s_eps, βLe[], βRe[], κ1e[], κ3e[]) - lr_eps
        p_cond[g] = p_stat[g] * exp(log_fe) / C_eps
    end
    Z = dot(p_cond, sw)
    Z > 0 && (p_cond ./= Z)

    # f(y_t | y_{t-1}) = Σ_{g1} p_cond(g1) × Σ_{g2} T(g1,g2) f_ε(y - g2) sw(g2)  × sw(g1)
    ny = length(y_grid)
    dy = y_grid[2] - y_grid[1]
    f = zeros(ny)

    for (iy, yv) in enumerate(y_grid)
        val = 0.0
        for g1 in 1:G
            if p_cond[g1] < 1e-300; continue; end
            inner = 0.0
            for g2 in 1:G
                log_fe = cspline_eval(yv - grid[g2], a_eps_s, s_eps, βLe[], βRe[], κ1e[], κ3e[]) - lr_eps
                inner += ws.T_mat[g1, g2] * exp(log_fe) / C_eps * sw[g2]
            end
            val += p_cond[g1] * inner * sw[g1]
        end
        f[iy] = val
    end

    total = sum(f) * dy
    total > 0 && (f ./= total)
    f
end

# ================================================================
#  MAIN COMPARISON
# ================================================================

function run_comparison_v2(; ρ_base::Float64=0.8, N::Int=5000, T::Int=3, seed::Int=42)
    K = 2; σy = 1.0; τ = [0.25, 0.50, 0.75]
    dgp = default_dgp(ρ_base=ρ_base)

    println("=" ^ 70)
    @printf("Switching DGP: ρ_base=%.1f, t(df=%d) shocks, χ²(%d) eps, N=%d, T=%d\n",
            ρ_base, dgp.df_v, dgp.df_eps, N, T)
    @printf("  δ=%.2f, d=%.2f, b=%.2f (switching prob ≈ 15%%)\n", dgp.δ, dgp.d, dgp.b)
    println("=" ^ 70)
    println()

    # Generate data
    y, eta = generate_switching_data(N, T, dgp; seed=seed)
    @printf("Data: y range [%.2f, %.2f], η range [%.2f, %.2f]\n",
            extrema(y)..., extrema(eta)...)
    @printf("  η std=%.3f, ε std=%.3f, y std=%.3f\n",
            std(eta), std(y .- eta), std(y))

    # Check switching frequency
    n_switch = 0; n_total = 0
    rng_check = MersenneTwister(seed)
    for i in 1:min(N, 1000)
        η = 0.0
        for b in 1:200; v=draw_v(rng_check,dgp); η=ρ_switch(η,v,dgp)*η+v; end
        for t in 2:T
            v = draw_v(rng_check, dgp)
            ρ_actual = ρ_switch(η, v, dgp)
            ρ_actual < dgp.ρ_base && (n_switch += 1)
            n_total += 1
            η = ρ_actual * η + v
        end
    end
    @printf("  Actual switching frequency: %.1f%%\n\n", 100*n_switch/n_total)

    # ---- Estimate with profiled MLE ----
    tp_init = make_true_cspline(rho=ρ_base, sigma_v=dgp.σ_v, sigma_eps=dgp.σ_ε)
    v_prof = pack_profiled(tp_init.a_Q, tp_init.a_init, tp_init.a_eps1, tp_init.a_eps3)

    @printf("Estimating profiled MLE...\n"); flush(stdout)
    v_ml, nll_ml = estimate_profiled_ml(y, K, σy, v_prof, τ; G=201, maxiter=500)
    aQ_ml, MQ_ml, ai_ml, Mi_ml, ae1_ml, ae3_ml, Me_ml = unpack_profiled(v_ml, K)
    @printf("  MLE: ρ=%.4f  ae3=%.4f  M_Q=%.2f  M_eps=%.2f\n",
            aQ_ml[2,2], ae3_ml, MQ_ml, Me_ml)

    # ---- Estimate with QR ----
    @printf("Estimating QR...\n"); flush(stdout)
    qr = estimate_cspline_qr(y, K, σy, tp_init.a_Q, tp_init.a_init,
                               tp_init.a_eps1, tp_init.a_eps3, τ;
                               G=201, S_em=30, M_draws=10, verbose=false, seed=seed)
    MQ_qr = _M_from_iqr(qr.a_Q[1,3] - qr.a_Q[1,1])
    Me_qr = qr.M_eps
    @printf("  QR:  ρ=%.4f  ae3=%.4f  M_Q=%.2f  M_eps=%.2f\n\n",
            qr.a_Q[2,2], qr.a_eps3, MQ_qr, Me_qr)

    # ---- Compare conditional densities f(y_t | y_{t-1}) ----
    y_lags = quantile(y[:,1], [0.1, 0.5, 0.9])
    y_eval = collect(range(-5.0, 5.0, length=500))
    dy = y_eval[2] - y_eval[1]

    @printf("Conditional density comparison f(y_t | y_{t-1}):\n")
    @printf("%-12s  %10s  %10s  %10s  %10s\n", "y_{t-1}", "KS_ML", "KS_QR", "L1_ML", "L1_QR")
    println("-" ^ 60)

    for y_lag in y_lags
        # True density by Monte Carlo
        f_true = true_conditional_density_mc(y_eval, y_lag, dgp; R=200000, seed=abs(Int(hash(y_lag)%10000))+seed)

        # MLE estimated density
        f_ml = estimated_conditional_density(y_eval, y_lag, aQ_ml, MQ_ml,
                                              ae1_ml, ae3_ml, Me_ml, K, σy, τ)

        # QR estimated density
        f_qr = estimated_conditional_density(y_eval, y_lag, qr.a_Q, MQ_qr,
                                              qr.a_eps1, qr.a_eps3, Me_qr, K, σy, τ)

        # KS distance
        cdf_true = cumsum(f_true) * dy
        cdf_ml = cumsum(f_ml) * dy
        cdf_qr = cumsum(f_qr) * dy
        ks_ml = maximum(abs.(cdf_ml .- cdf_true))
        ks_qr = maximum(abs.(cdf_qr .- cdf_true))

        # L1 distance
        l1_ml = sum(abs.(f_ml .- f_true)) * dy
        l1_qr = sum(abs.(f_qr .- f_true)) * dy

        winner = ks_ml < ks_qr ? "MLE<QR" : "QR<MLE"
        @printf("y=%+8.3f  %10.4f  %10.4f  %10.4f  %10.4f  %s\n",
                y_lag, ks_ml, ks_qr, l1_ml, l1_qr, winner)
    end
    println()
    flush(stdout)
end

# Run for both ρ values and T values
for ρ in [0.8, 1.0]
    for T in [3, 5]
        run_comparison_v2(ρ_base=ρ, N=5000, T=T, seed=42)
        println()
    end
end

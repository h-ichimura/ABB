#=
misspecified_comparison.jl — Compare MLE vs QR density approximation
under misspecified DGP (mixture shocks, non-Gaussian transition).

DGP: η_t = ρ η_{t-1} + v_t,  y_t = η_t + ε_t
  v_t ~ 0.85 N(0, σ²_v) + 0.15 N(0, 4σ²_v)  (fat-tailed mixture)
  ε_t ~ N(0, σ²_ε)
  σ_v = 0.5, σ_ε = 0.3 (close to ABB/GP estimates)

Estimation model: C² cubic spline (misspecified)
Compare: KS distance, L1 distance, graphical densities
=#

include("cspline_abb.jl")
using Printf, Statistics, Random, LinearAlgebra

# ================================================================
#  MISSPECIFIED DGP: Mixture AR(1) + Gaussian ε
# ================================================================

function generate_misspecified(N::Int, T::Int, ρ::Float64;
                                σ_v::Float64=0.5, σ_ε::Float64=0.3,
                                p_mix::Float64=0.15, scale_mix::Float64=2.0,
                                seed::Int=42, burnin::Int=100)
    rng = MersenneTwister(seed)
    eta = zeros(N, T); y = zeros(N, T)

    # Stationary variance of η: Var(η) = σ²_v_eff / (1 - ρ²)
    # where σ²_v_eff = (1-p)σ²_v + p(scale²σ²_v) = σ²_v(1-p+p×scale²)
    σ_v_eff = σ_v * sqrt((1-p_mix) + p_mix*scale_mix^2)

    for i in 1:N
        # Burn-in for stationary init (only if ρ < 1)
        η = 0.0
        if ρ < 1.0
            for b in 1:burnin
                if rand(rng) < p_mix
                    η = ρ * η + scale_mix * σ_v * randn(rng)
                else
                    η = ρ * η + σ_v * randn(rng)
                end
            end
        else
            # Random walk: start from N(0, σ²_v × burnin) is not stationary
            # Just start from 0
            η = 0.0
        end
        eta[i,1] = η

        for t in 2:T
            if rand(rng) < p_mix
                eta[i,t] = ρ * eta[i,t-1] + scale_mix * σ_v * randn(rng)
            else
                eta[i,t] = ρ * eta[i,t-1] + σ_v * randn(rng)
            end
        end
    end

    for t in 1:T, i in 1:N
        y[i,t] = eta[i,t] + σ_ε * randn(rng)
    end

    y, eta
end

# ================================================================
#  TRUE CONDITIONAL DENSITY (via convolution on fine grid)
# ================================================================

function true_conditional_density_y(y_grid::Vector{Float64}, y_lag::Float64,
                                      ρ::Float64, σ_v::Float64, σ_ε::Float64;
                                      p_mix::Float64=0.15, scale_mix::Float64=2.0,
                                      n_eta::Int=2000)
    # f(y_t | y_{t-1}) = ∫ f_trans(η_t | η_{t-1}) f_ε(y_t - η_t) f(η_{t-1} | y_{t-1}) dη_t dη_{t-1}
    # For simplicity, condition on η_{t-1} = E[η | y_{t-1}] ≈ signal extraction
    # Signal extraction: η_{t-1} ≈ (σ²_η / σ²_y) × y_{t-1} where σ²_y = σ²_η + σ²_ε
    # For the mixture, σ²_v_eff = σ²_v × ((1-p) + p×scale²)
    σ²_v_eff = σ_v^2 * ((1-p_mix) + p_mix*scale_mix^2)
    if ρ < 1.0
        σ²_η = σ²_v_eff / (1 - ρ^2)
    else
        # Non-stationary: use large variance
        σ²_η = σ²_v_eff * 100  # approximate
    end
    σ²_y = σ²_η + σ_ε^2
    # Kalman gain
    K = σ²_η / σ²_y
    η_hat = K * y_lag  # E[η_{t-1} | y_{t-1}]

    # f(y_t | y_{t-1}) ≈ ∫ f_v(η_t - ρ η_hat) × f_ε(y_t - η_t) dη_t
    # = convolution of mixture-Gaussian(ρ η_hat, σ²_v_mix) with N(0, σ²_ε)
    # evaluated at y_t
    # Result: mixture of Gaussians in y_t
    μ = ρ * η_hat
    # Component 1: N(μ, σ²_v + σ²_ε)
    σ1 = sqrt(σ_v^2 + σ_ε^2)
    # Component 2: N(μ, scale²σ²_v + σ²_ε)
    σ2 = sqrt(scale_mix^2 * σ_v^2 + σ_ε^2)

    f = zeros(length(y_grid))
    for (i, y) in enumerate(y_grid)
        z1 = (y - μ) / σ1
        z2 = (y - μ) / σ2
        f[i] = (1-p_mix) / σ1 * exp(-0.5*z1^2) / sqrt(2π) +
               p_mix / σ2 * exp(-0.5*z2^2) / sqrt(2π)
    end
    f
end

# ================================================================
#  ESTIMATED CONDITIONAL DENSITY FROM MLE/QR
# ================================================================

function estimated_density_y(y_grid::Vector{Float64}, y_lag::Float64,
                              a_Q::Matrix{Float64}, M_Q::Float64,
                              a_eps1::Float64, a_eps3::Float64, M_eps::Float64,
                              K::Int, σy::Float64, τ::Vector{Float64};
                              n_eta::Int=500)
    nk = K+1
    # f(y_t | y_{t-1}) = ∫ f_trans(η_t | η_{t-1}) f_ε(y_t - η_t) dη_t
    # Approximate: condition on η_{t-1} via signal extraction (same as true)
    # Actually, compute on a grid of η_t and integrate

    # Solve transition density at η_{t-1} = y_lag (approximate)
    z = y_lag / σy
    hv = zeros(nk); hv[1]=1.0; K>=1 && (hv[2]=z)
    for k in 2:K; hv[k+1] = z*hv[k] - (k-1)*hv[k-1]; end
    t_loc = [dot(view(a_Q,:,l), hv) for l in 1:3]

    s = zeros(3); βL=Ref(0.0); βR=Ref(0.0); κ1=Ref(0.0); κ3=Ref(0.0)
    (t_loc[2] <= t_loc[1] || t_loc[3] <= t_loc[2]) && return zeros(length(y_grid))
    solve_cspline_c2!(s, βL, βR, κ1, κ3, t_loc, τ, M_Q)
    lr_t = max(s[1], s[2], s[3])
    m_t = zeros(4)
    cspline_masses!(m_t, t_loc, s, βL[], βR[], κ1[], κ3[], lr_t)
    C_t = sum(m_t)
    C_t < 1e-300 && return zeros(length(y_grid))

    # Eps density
    a_eps_s = [a_eps1, 0.0, a_eps3]
    s_eps = zeros(3); βLe=Ref(0.0); βRe=Ref(0.0); κ1e=Ref(0.0); κ3e=Ref(0.0)
    solve_cspline_c2!(s_eps, βLe, βRe, κ1e, κ3e, a_eps_s, τ, M_eps)
    lr_e = max(s_eps[1], s_eps[2], s_eps[3])
    m_e = zeros(4)
    cspline_masses!(m_e, a_eps_s, s_eps, βLe[], βRe[], κ1e[], κ3e[], lr_e)
    C_e = sum(m_e)

    # Convolution on fine η grid
    η_lo = t_loc[1] - 5.0/sqrt(max(-M_Q, 0.1))
    η_hi = t_loc[3] + 5.0/sqrt(max(-M_Q, 0.1))
    dη = (η_hi - η_lo) / (n_eta - 1)
    η_grid = collect(range(η_lo, η_hi, length=n_eta))

    f = zeros(length(y_grid))
    for (iy, yv) in enumerate(y_grid)
        val = 0.0
        for (ig, ηv) in enumerate(η_grid)
            f_trans = exp(cspline_eval(ηv, t_loc, s, βL[], βR[], κ1[], κ3[]) - lr_t) / C_t
            f_eps = exp(cspline_eval(yv - ηv, a_eps_s, s_eps, βLe[], βRe[], κ1e[], κ3e[]) - lr_e) / C_e
            val += f_trans * f_eps * dη
        end
        f[iy] = val
    end
    f
end

# ================================================================
#  MAIN COMPARISON
# ================================================================

function run_comparison(; ρ::Float64=0.8, N::Int=5000, T::Int=3, seed::Int=42)
    K = 2; σy = 1.0; τ = [0.25, 0.50, 0.75]
    σ_v = 0.5; σ_ε = 0.3

    println("=" ^ 70)
    @printf("Misspecified DGP: rho=%.1f, mixture shocks, N=%d, T=%d\n", ρ, N, T)
    println("=" ^ 70)
    println()

    # Generate misspecified data
    y, eta = generate_misspecified(N, T, ρ; σ_v=σ_v, σ_ε=σ_ε, seed=seed)
    @printf("Data: y range [%.2f, %.2f], η range [%.2f, %.2f]\n",
            extrema(y)..., extrema(eta)...)

    # ---- Estimate with profiled MLE ----
    # Need initial values — use OLS-based quantile estimates
    tp_init = make_true_cspline(rho=ρ, sigma_v=σ_v, sigma_eps=σ_ε)
    v_prof = pack_profiled(tp_init.a_Q, tp_init.a_init, tp_init.a_eps1, tp_init.a_eps3)

    @printf("\nEstimating profiled MLE...\n"); flush(stdout)
    v_ml, nll_ml = estimate_profiled_ml(y, K, σy, v_prof, τ; G=201, maxiter=500)
    aQ_ml, MQ_ml, ai_ml, Mi_ml, ae1_ml, ae3_ml, Me_ml = unpack_profiled(v_ml, K)
    @printf("  MLE: ρ=%.4f  ae3=%.4f  M_Q=%.2f\n", aQ_ml[2,2], ae3_ml, MQ_ml)

    # ---- Estimate with QR ----
    @printf("Estimating QR...\n"); flush(stdout)
    qr = estimate_cspline_qr(y, K, σy, tp_init.a_Q, tp_init.a_init,
                               tp_init.a_eps1, tp_init.a_eps3, τ;
                               G=201, S_em=30, M_draws=10, verbose=false, seed=seed)
    MQ_qr = _M_from_iqr(qr.a_Q[1,3] - qr.a_Q[1,1])
    Me_qr = qr.M_eps
    @printf("  QR:  ρ=%.4f  ae3=%.4f  M_Q=%.2f\n", qr.a_Q[2,2], qr.a_eps3, MQ_qr)

    # ---- Compare conditional densities f(y_t | y_{t-1}) ----
    # At several conditioning values
    y_lags = quantile(y[:,1], [0.1, 0.5, 0.9])
    y_eval = collect(range(-4.0, 4.0, length=500))
    dy = y_eval[2] - y_eval[1]

    @printf("\nConditional density comparison f(y_t | y_{t-1}):\n")
    @printf("%-12s  %10s  %10s  %10s  %10s\n", "y_{t-1}", "KS_ML", "KS_QR", "L1_ML", "L1_QR")
    println("-"^56)

    for y_lag in y_lags
        f_true = true_conditional_density_y(y_eval, y_lag, ρ, σ_v, σ_ε)
        f_ml = estimated_density_y(y_eval, y_lag, aQ_ml, MQ_ml, ae1_ml, ae3_ml, Me_ml,
                                    K, σy, τ)
        f_qr = estimated_density_y(y_eval, y_lag, qr.a_Q, MQ_qr, qr.a_eps1, qr.a_eps3, Me_qr,
                                    K, σy, τ)

        # Normalize
        f_true ./= sum(f_true)*dy
        f_ml ./= max(sum(f_ml)*dy, 1e-300)
        f_qr ./= max(sum(f_qr)*dy, 1e-300)

        # KS distance
        cdf_true = cumsum(f_true)*dy
        cdf_ml = cumsum(f_ml)*dy
        cdf_qr = cumsum(f_qr)*dy
        ks_ml = maximum(abs.(cdf_ml - cdf_true))
        ks_qr = maximum(abs.(cdf_qr - cdf_true))

        # L1 distance
        l1_ml = sum(abs.(f_ml - f_true)) * dy
        l1_qr = sum(abs.(f_qr - f_true)) * dy

        @printf("y=%-+8.3f  %10.4f  %10.4f  %10.4f  %10.4f  %s\n",
                y_lag, ks_ml, ks_qr, l1_ml, l1_qr,
                ks_ml < ks_qr ? "MLE<QR" : "QR<MLE")
    end

    # ---- Marginal density of y_t ----
    @printf("\nMarginal density of y_t:\n")
    # Kernel density estimate from data
    # True: mixture of Gaussians (convolution of stationary η + ε)
    # Compare MLE and QR implied marginal

    println()
    flush(stdout)
end

# Run for both ρ values
for ρ in [0.8, 1.0]
    run_comparison(ρ=ρ, N=5000, T=3, seed=42)
    println()
    run_comparison(ρ=ρ, N=5000, T=5, seed=42)
    println("\n\n")
end

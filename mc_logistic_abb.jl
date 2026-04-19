#=
mc_logistic_abb.jl — Monte Carlo: QR vs MLE(cold) vs MLE(QR warm start)

DGP: all-smooth asymmetric logistic (init, transition, ε)
ABB-style quantile knot parameterization.

Three estimators:
  1. QR: ABB stochastic EM
  2. MLE cold: LBFGS from init_params (slope=0.5, away from truth)
  3. MLE warm: LBFGS from QR solution
=#

include("logistic_abb.jl")
using Printf, Statistics, Serialization

"""
Logistic partial log-likelihood for MH E-step.
Uses asymmetric logistic for transition and ε (not piecewise-uniform).
"""
function logistic_partial_loglik(y, eta, t::Int, par::Params,
                                 cfg::Config, sigma_y::Float64)
    a_eps_s = sort(par.a_eps)
    log3 = log(3.0)
    αL_e = log3 / (a_eps_s[2] - a_eps_s[1])
    αR_e = log3 / (a_eps_s[3] - a_eps_s[2])
    C_e = 2.0 * αL_e * αR_e / (αL_e + αR_e)
    ll = asym_logistic_logpdf(y[t] - eta[t], a_eps_s[1], a_eps_s[2], a_eps_s[3]) - log(C_e)

    if t == 1
        a_init_s = sort(par.a_init)
        αL_i = log3 / (a_init_s[2] - a_init_s[1])
        αR_i = log3 / (a_init_s[3] - a_init_s[2])
        C_i = 2.0 * αL_i * αR_i / (αL_i + αR_i)
        ll += asym_logistic_logpdf(eta[1], a_init_s[1], a_init_s[2], a_init_s[3]) - log(C_i)
    end
    if t >= 2
        q1, q2, q3 = gap_quantiles(eta[t-1], par.a_Q, cfg.K, sigma_y)
        gap_L = q2 - q1; gap_R = q3 - q2
        αL = log3 / gap_L; αR = log3 / gap_R
        C_t = 2.0 * αL * αR / (αL + αR)
        ll += asym_logistic_logpdf(eta[t], q1, q2, q3) - log(C_t)
    end
    if t < cfg.T
        q1, q2, q3 = gap_quantiles(eta[t], par.a_Q, cfg.K, sigma_y)
        gap_L = q2 - q1; gap_R = q3 - q2
        αL = log3 / gap_L; αR = log3 / gap_R
        C_t = 2.0 * αL * αR / (αL + αR)
        ll += asym_logistic_logpdf(eta[t+1], q1, q2, q3) - log(C_t)
    end
    ll
end

"""E-step using logistic density for the posterior."""
function logistic_e_step!(eta_all::Array{Float64,3}, y::Matrix{Float64},
                          par::Params, cfg::Config, sigma_y::Float64)
    N, T, M = cfg.N, cfg.T, cfg.M
    n_draws = cfg.n_draws
    save_start = n_draws - M + 1
    eta_cur = eta_all[:, :, M]
    acc_count = zeros(T)
    eta_buf = zeros(T)

    pll = zeros(N, T)
    for i in 1:N, t in 1:T
        pll[i,t] = logistic_partial_loglik(view(y,i,:), view(eta_cur,i,:),
                                            t, par, cfg, sigma_y)
    end

    save_idx = 0
    for d in 1:n_draws
        for t in 1:T
            vp = cfg.var_prop[t]
            for i in 1:N
                @inbounds for s in 1:T; eta_buf[s] = eta_cur[i,s]; end
                eta_buf[t] = eta_cur[i,t] + sqrt(vp) * randn()
                prop = logistic_partial_loglik(view(y,i,:), eta_buf,
                                               t, par, cfg, sigma_y)
                if log(rand()) < prop - pll[i,t]
                    eta_cur[i,t] = eta_buf[t]
                    pll[i,t] = prop
                    t > 1 && (pll[i,t-1] = logistic_partial_loglik(
                        view(y,i,:), view(eta_cur,i,:), t-1, par, cfg, sigma_y))
                    t < T && (pll[i,t+1] = logistic_partial_loglik(
                        view(y,i,:), view(eta_cur,i,:), t+1, par, cfg, sigma_y))
                    acc_count[t] += 1
                end
            end
        end
        if d >= save_start
            save_idx += 1
            eta_all[:, :, save_idx] .= eta_cur
        end
    end
    acc_count ./ (N * n_draws)
end

"""
Convert QR's direct-quantile a_Q (col1=τ₁, col2=τ₂, col3=τ₃) to
gap parameterization (col1=median, col2=log_gap_L, col3=log_gap_R).

Since gap = Q₂(η) - Q₁(η) = h(η)'(a_QR[:,2] - a_QR[:,1]) is linear in h(η),
log(gap) is NOT linear in h(η). We approximate by regression on the
stacked η_{t-1} values from the E-step draws.
"""
function direct_to_gap!(par::Params, eta_all::Array{Float64,3}, cfg::Config)
    K, L = cfg.K, cfg.L
    # Save direct-quantile columns
    a_direct = copy(par.a_Q)  # col1 = τ₁ quantile, col2 = τ₂, col3 = τ₃

    # Set col1 = median (τ₂)
    par.a_Q[:, 1] .= a_direct[:, 2]

    # Compute gaps at observed η_{t-1} values and regress log(gap) on Hermite
    eta_t, eta_lag = stack_transition(eta_all, cfg)
    H = hermite_basis(eta_lag, K, cfg.sigma_y)
    n_obs = length(eta_t)

    # Q values at observed η_lag
    Q1 = H * a_direct[:, 1]  # τ₁ quantile values
    Q2 = H * a_direct[:, 2]  # τ₂ quantile values
    Q3 = H * a_direct[:, 3]  # τ₃ quantile values

    gap_L = Q2 .- Q1  # should be positive if no crossing
    gap_R = Q3 .- Q2

    # Clip to prevent log of non-positive
    gap_L = max.(gap_L, 1e-6)
    gap_R = max.(gap_R, 1e-6)

    # Regress log(gap) on Hermite basis
    par.a_Q[:, 2] .= H \ log.(gap_L)
    par.a_Q[:, 3] .= H \ log.(gap_R)
end

"""
Convert gap-parameterized a_Q to direct-quantile a_Q for use by m_step_qr!.
Evaluates gap_quantiles at observed η_lag values and fits linear Hermite to each Q.
"""
function gap_to_direct!(par::Params, eta_all::Array{Float64,3}, cfg::Config)
    K, L = cfg.K, cfg.L
    eta_t, eta_lag = stack_transition(eta_all, cfg)
    H = hermite_basis(eta_lag, K, cfg.sigma_y)

    # Evaluate Q₁, Q₂, Q₃ at each observed η_lag via gap_quantiles
    n_obs = length(eta_t)
    Q1 = zeros(n_obs); Q2 = zeros(n_obs); Q3 = zeros(n_obs)
    for j in 1:n_obs
        q1, q2, q3 = gap_quantiles(eta_lag[j], par.a_Q, K, cfg.sigma_y)
        Q1[j] = q1; Q2[j] = q2; Q3[j] = q3
    end

    # Fit linear Hermite to each quantile column
    par.a_Q[:, 1] .= H \ Q1  # τ₁ quantile coefficients
    par.a_Q[:, 2] .= H \ Q2  # τ₂
    par.a_Q[:, 3] .= H \ Q3  # τ₃
end

function run_one(seed::Int, N::Int, par_true::Params,
                 K::Int, L::Int, sigma_y::Float64, tau::Vector{Float64})
    y, eta = generate_data_logistic_abb(N, par_true, sigma_y, K; seed=seed)
    cfg = Config(N, 3, K, L, tau, sigma_y, 1, 1, 1, fill(0.05, 3))

    # 1. QR via EM with LOGISTIC E-step (correct posterior)
    par_qr = copy_params(par_true)  # start from truth for E-step
    par_qr.a_Q .*= 0.9  # perturb slightly (in gap space)
    eta_all = zeros(N, 3, 20)
    for m in 1:20; eta_all[:,:,m] .= 0.6 .* y; end
    cfg_em = Config(N, 3, K, L, tau, sigma_y, 30, 100, 20, fill(0.05, 3))
    t_qr = @elapsed for iter in 1:30
        logistic_e_step!(eta_all, y, par_qr, cfg_em, sigma_y)
        # Convert gap → direct before QR
        gap_to_direct!(par_qr, eta_all, cfg_em)
        # Run standard QR (operates on direct-quantile columns)
        m_step_qr!(par_qr, eta_all, y, cfg_em)
        # Convert direct → gap after QR
        direct_to_gap!(par_qr, eta_all, cfg_em)
        # Fix ε median at 0: shift all ε quantiles so a_eps[2] = 0
        shift = par_qr.a_eps[2]
        par_qr.a_eps .-= shift
    end

    # 2. MLE cold: LBFGS from truth (10% perturbed)
    par_cold = copy_params(par_true)
    par_cold.a_Q .*= 0.9
    t_cold = @elapsed par_ml_cold = estimate_logistic_abb_ml(y, cfg, par_cold;
                                                              G=201, maxiter=30, verbose=false)

    # 3. MLE warm: LBFGS from QR solution
    par_warm = copy_params(par_qr)
    t_warm = @elapsed par_ml_warm = estimate_logistic_abb_ml(y, cfg, par_warm;
                                                              G=201, maxiter=30, verbose=false)

    sl_qr = par_qr.a_Q[2, :] ./ sigma_y
    sl_cold = par_ml_cold.a_Q[2, :] ./ sigma_y
    sl_warm = par_ml_warm.a_Q[2, :] ./ sigma_y
    in_qr = par_qr.a_Q[1, :]
    in_cold = par_ml_cold.a_Q[1, :]
    in_warm = par_ml_warm.a_Q[1, :]

    nll_qr = logistic_neg_loglik(par_qr, y, cfg; G=201)
    nll_cold = logistic_neg_loglik(par_ml_cold, y, cfg; G=201)
    nll_warm = logistic_neg_loglik(par_ml_warm, y, cfg; G=201)

    (seed=seed,
     sl_qr=sl_qr, sl_cold=sl_cold, sl_warm=sl_warm,
     in_qr=in_qr, in_cold=in_cold, in_warm=in_warm,
     nll_qr=nll_qr, nll_cold=nll_cold, nll_warm=nll_warm,
     t_qr=t_qr, t_cold=t_cold, t_warm=t_warm)
end

function mc_run(; N=300, R=20, K=2, sigma_y=1.0)
    L = 3; tau = [0.25, 0.50, 0.75]
    par_true = make_true_params_gap(K=K)
    true_sl = par_true.a_Q[2, :] ./ sigma_y
    true_in = par_true.a_Q[1, :]

    @printf("MC: QR vs MLE(cold) vs MLE(warm), N=%d, R=%d\n", N, R)
    @printf("True slopes:  [%.4f, %.4f, %.4f]\n", true_sl...)
    @printf("True intcpts: [%.4f, %.4f, %.4f]\n\n", true_in...)
    flush(stdout)

    results = []
    for r in 1:R
        res = run_one(r, N, par_true, K, L, sigma_y, tau)
        push!(results, res)
        @printf("r=%2d | QR=[%.3f,%.3f,%.3f] cold=[%.3f,%.3f,%.3f] warm=[%.3f,%.3f,%.3f] nll=[%.3f,%.3f,%.3f] (%.0f/%.0f/%.0fs)\n",
                r, res.sl_qr..., res.sl_cold..., res.sl_warm...,
                res.nll_qr, res.nll_cold, res.nll_warm,
                res.t_qr, res.t_cold, res.t_warm)
        flush(stdout)
    end

    function summarize(name, ests, true_vals)
        m = mean(ests, dims=1)[:]; b = m .- true_vals; s = std(ests, dims=1)[:]
        rmse = sqrt.(mean((ests .- true_vals').^2, dims=1)[:])
        @printf("  %-10s mean=[%.4f,%.4f,%.4f] bias=[%+.4f,%+.4f,%+.4f] std=[%.4f,%.4f,%.4f] RMSE=[%.4f,%.4f,%.4f]\n",
                name, m..., b..., s..., rmse...)
    end

    sl_qr   = hcat([r.sl_qr   for r in results]...)' |> collect
    sl_cold = hcat([r.sl_cold  for r in results]...)' |> collect
    sl_warm = hcat([r.sl_warm  for r in results]...)' |> collect
    in_qr   = hcat([r.in_qr   for r in results]...)' |> collect
    in_cold = hcat([r.in_cold  for r in results]...)' |> collect
    in_warm = hcat([r.in_warm  for r in results]...)' |> collect

    println("\n","="^70)
    println("SLOPES (true = [$(join([@sprintf("%.4f",s) for s in true_sl], ", "))])")
    summarize("QR", sl_qr, true_sl)
    summarize("MLE cold", sl_cold, true_sl)
    summarize("MLE warm", sl_warm, true_sl)
    rmse_qr = sqrt.(mean((sl_qr .- true_sl').^2, dims=1)[:])
    rmse_cold = sqrt.(mean((sl_cold .- true_sl').^2, dims=1)[:])
    rmse_warm = sqrt.(mean((sl_warm .- true_sl').^2, dims=1)[:])
    @printf("  Eff QR/cold: [%.2f, %.2f, %.2f]\n", (rmse_qr ./ rmse_cold)...)
    @printf("  Eff QR/warm: [%.2f, %.2f, %.2f]\n", (rmse_qr ./ rmse_warm)...)

    println("\nINTERCEPTS (true = [$(join([@sprintf("%.4f",s) for s in true_in], ", "))])")
    summarize("QR", in_qr, true_in)
    summarize("MLE cold", in_cold, true_in)
    summarize("MLE warm", in_warm, true_in)

    println("\nNEG-LL (lower is better)")
    @printf("  QR:       mean=%.4f\n", mean(r.nll_qr for r in results))
    @printf("  MLE cold: mean=%.4f\n", mean(r.nll_cold for r in results))
    @printf("  MLE warm: mean=%.4f\n", mean(r.nll_warm for r in results))

    @printf("\nAvg time: QR=%.1fs, MLE cold=%.1fs, MLE warm=%.1fs\n",
            mean(r.t_qr for r in results),
            mean(r.t_cold for r in results),
            mean(r.t_warm for r in results))

    serialize("mc_logistic_abb_N$(N)_R$(R).jls", results)
    @printf("Saved mc_logistic_abb_N%d_R%d.jls\n", N, R)
end

for N in [500, 1000]
    mc_run(N=N, R=20)
    println("\n\n")
end

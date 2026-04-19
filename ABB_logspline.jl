#=
ABB_logspline.jl — Compare QR vs logspline MLE on data from ABB's DGP

Same DGP: piecewise-uniform (ABB model)
Same E-step draws
Different M-steps:
  QR: quantile regression (ABB's approach)
  LS-MLE: logspline density, optimized by LBFGS (smooth)

The logspline is a smooth approximation to the true piecewise-uniform
transition density. It enables smooth MLE optimization.
=#

include("ABB_three_period.jl")
include("logspline.jl")

using Optim, Serialization

# ================================================================
#  LOGSPLINE TRANSITION M-STEP
# ================================================================

"""
M-step: fit logspline transition density to the posterior draws by MLE.
Uses LBFGS on the smooth logspline neg-loglik.
Marginals (init, eps) estimated by sample quantiles (same as QR).
"""
function m_step_logspline!(a_trans::Matrix{Float64},
                           knots_sp::Vector{Float64},
                           par::Params,  # for marginals
                           eta_all::Array{Float64,3},
                           y::Matrix{Float64}, cfg::Config)
    K_sp = length(knots_sp)
    J = cfg.K; L = cfg.L

    # Marginals: same as QR
    eta1_all = stack_initial(eta_all, cfg)
    for l in 1:L; par.a_init[l] = quantile(eta1_all, cfg.tau[l]); end
    par.b1_init, par.bL_init = update_tails(eta1_all, par.a_init[1], par.a_init[L])

    eps_all = stack_eps(eta_all, y, cfg)
    for l in 1:L; par.a_eps[l] = quantile(eps_all, cfg.tau[l]); end
    par.a_eps .-= mean(par.a_eps)
    par.b1_eps, par.bL_eps = update_tails(eps_all, par.a_eps[1], par.a_eps[L])

    # Transition: LBFGS on smooth logspline
    eta_t, eta_lag = stack_transition(eta_all, cfg)
    H_lag = hermite_basis(eta_lag, J, cfg.sigma_y)

    function neg_ll(theta)
        logspline_neg_loglik(theta, eta_t, H_lag, knots_sp)
    end

    function neg_ll_grad!(g, theta)
        g .= logspline_neg_loglik_grad(theta, eta_t, H_lag, knots_sp)
    end

    theta0 = vec(copy(a_trans))
    od = Optim.OnceDifferentiable(neg_ll, neg_ll_grad!, theta0)
    res = optimize(od, theta0, LBFGS(),
                   Optim.Options(iterations=100, g_tol=1e-6, show_trace=false))

    a_trans .= reshape(Optim.minimizer(res), K_sp + 2, J + 1)
    nothing
end

# ================================================================
#  E-STEP WITH LOGSPLINE POSTERIOR
# ================================================================

"""Cache-based E-step using logspline transition for the posterior."""
function ls_e_step!(eta_all::Array{Float64,3}, y::Matrix{Float64},
                    par::Params, ls::LogsplineTransition, cfg::Config)
    N, T, M = cfg.N, cfg.T, cfg.M
    n_draws = cfg.n_draws
    save_start = n_draws - M + 1

    eta_cur = eta_all[:, :, M]
    acc_count = zeros(T)
    eta_buf = zeros(T)

    # Cache log-normalizers: logC[i,t] = log C(η_{i,t}) for transition from t
    logC = zeros(N, T)
    for i in 1:N, t in 1:T
        beta0, beta1, gamma = logspline_coeffs(ls, eta_cur[i,t])
        logC[i,t] = log(max(logspline_normalize(beta0, beta1, gamma, ls.knots), 1e-300))
    end

    # Partial loglik with cached normalizers
    function pll_cached(y_i, eta_i, t, logC_i)
        ll = pw_logdens(y_i[t] - eta_i[t], par.a_eps, cfg.tau,
                        par.b1_eps, par.bL_eps)
        if t == 1
            ll += pw_logdens(eta_i[1], par.a_init, cfg.tau,
                             par.b1_init, par.bL_init)
        end
        if t >= 2
            beta0, beta1, gamma = logspline_coeffs(ls, eta_i[t-1])
            ll += logspline_s(eta_i[t], beta0, beta1, gamma, ls.knots) - logC_i[t-1]
        end
        if t < T
            beta0, beta1, gamma = logspline_coeffs(ls, eta_i[t])
            ll += logspline_s(eta_i[t+1], beta0, beta1, gamma, ls.knots) - logC_i[t]
        end
        ll
    end

    pll = zeros(N, T)
    for i in 1:N, t in 1:T
        pll[i,t] = pll_cached(view(y,i,:), view(eta_cur,i,:), t, view(logC,i,:))
    end

    save_idx = 0
    for d in 1:n_draws
        for t in 1:T
            vp = cfg.var_prop[t]
            for i in 1:N
                @inbounds for s in 1:T; eta_buf[s] = eta_cur[i,s]; end
                eta_buf[t] = eta_cur[i,t] + sqrt(vp) * randn()

                # Recompute logC only if η_t changed as conditioning variable
                logC_prop = copy(view(logC, i, :))
                beta0, beta1, gamma = logspline_coeffs(ls, eta_buf[t])
                logC_prop[t] = log(max(logspline_normalize(
                    beta0, beta1, gamma, ls.knots), 1e-300))

                prop = pll_cached(view(y,i,:), eta_buf, t, logC_prop)
                if log(rand()) < prop - pll[i,t]
                    eta_cur[i,t] = eta_buf[t]
                    logC[i,t] = logC_prop[t]
                    pll[i,t] = prop
                    t > 1 && (pll[i,t-1] = pll_cached(
                        view(y,i,:), view(eta_cur,i,:), t-1, view(logC,i,:)))
                    t < T && (pll[i,t+1] = pll_cached(
                        view(y,i,:), view(eta_cur,i,:), t+1, view(logC,i,:)))
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

# ================================================================
#  ESTIMATION LOOP
# ================================================================

function estimate_ls(y::Matrix{Float64}, cfg::Config;
                     knots_sp::Vector{Float64}=[-1.0, -0.3, 0.0, 0.3, 1.0],
                     verbose::Bool=true)
    N, T, M = cfg.N, cfg.T, cfg.M
    S = cfg.maxiter; J = cfg.K; L = cfg.L
    K_sp = length(knots_sp)

    # Initialize: use ABB params for marginals, logspline for transition
    par = init_params(y, cfg)

    # Initialize logspline transition coefficients
    a_trans = zeros(K_sp + 2, J + 1)
    # Start with small negative cubics (tail decay) and moderate linear term
    a_trans[2, 2] = 0.5 * cfg.sigma_y  # slope ~ 0.5 (away from true 0.8)
    for k in 1:K_sp
        a_trans[k + 2, 1] = -0.3
    end

    eta_all = zeros(N, T, M)
    for m in 1:M; eta_all[:,:,m] .= 0.6 .* y; end

    ll_hist = zeros(S)

    for iter in 1:S
        ls = LogsplineTransition(a_trans, knots_sp, cfg.sigma_y, J)

        # E-step with logspline posterior
        acc = ls_e_step!(eta_all, y, par, ls, cfg)

        # M-step: logspline MLE for transition, sample quantiles for marginals
        m_step_logspline!(a_trans, knots_sp, par, eta_all, y, cfg)

        # Monitor: logspline transition loglik (not piecewise-uniform)
        ls_new = LogsplineTransition(a_trans, knots_sp, cfg.sigma_y, J)
        ll = 0.0
        for m in 1:M, i in 1:N
            eta_i = view(eta_all, i, :, m)
            ll += pw_logdens(eta_i[1], par.a_init, cfg.tau,
                             par.b1_init, par.bL_init)
            for t in 1:T
                ll += pw_logdens(y[i,t] - eta_i[t], par.a_eps, cfg.tau,
                                 par.b1_eps, par.bL_eps)
            end
            for t in 2:T
                ll += logspline_transition_logdens(eta_i[t], eta_i[t-1], ls_new)
            end
        end
        ll_hist[iter] = ll / (N * M)

        if verbose && (iter % 10 == 0 || iter <= 3)
            @printf("  [LS ] %3d/%d | ll %8.4f | acc %s\n",
                    iter, S, ll_hist[iter],
                    join([@sprintf("%.2f",a) for a in acc], "/"))
        end
    end

    @printf("  Logspline: S=%d done\n", S)
    a_trans, par, eta_all, ll_hist
end

# ================================================================
#  COMPARISON: QR vs LOGSPLINE MLE on ABB DGP
# ================================================================

function run_ls_comparison(; N=300, K=2, L=3, maxiter=50, n_draws=200, M=50,
                            var_prop=0.05, seed=42,
                            knots_sp=[-1.0, -0.3, 0.0, 0.3, 1.0])
    T = 3; sigma_y = 1.0
    tau = collect(range(1/(L+1), stop=L/(L+1), length=L))
    par_true = make_true_params_linear(tau=tau, sigma_y=sigma_y, K=K)
    y, eta_true = generate_data_abb(N, par_true, tau, sigma_y, K; seed=seed)
    vp = fill(var_prop, T)
    cfg = Config(N, T, K, L, tau, sigma_y, maxiter, n_draws, M, vp)

    println("="^60)
    println("  QR vs Logspline MLE on ABB DGP")
    println("="^60)
    @printf("N=%d  S=%d  M=%d  n_draws=%d\n", N, maxiter, M, n_draws)
    println("True persistence: 0.8")
    println()

    # QR
    println("--- QR ---")
    t_qr = @elapsed par_qr, _, eta_qr, ll_qr, hist_qr =
        estimate(y, cfg; method=:qr)
    @printf("  %.1f s\n", t_qr)

    # Logspline MLE
    println("--- Logspline MLE ---")
    t_ls = @elapsed a_trans_ls, par_ls, eta_ls, ll_ls =
        estimate_ls(y, cfg; knots_sp=knots_sp)
    @printf("  %.1f s\n", t_ls)

    # Results
    println("\n--- Results ---")
    S2 = div(maxiter, 2)
    @printf("Avg ll (last %d):  QR=%.4f  LS=%.4f\n",
            S2, mean(ll_qr[end-S2+1:end]), mean(ll_ls[end-S2+1:end]))

    println("\nQR persistence: ", round.(par_qr.a_Q[2,:] ./ sigma_y, digits=4))
    println("LS β₁ coeffs:   ", round.(a_trans_ls[2,:], digits=4))
    println("True:            0.8")

    println("\nη recovery (corr):")
    for t in 1:T
        @printf("  t=%d: QR=%.4f  LS=%.4f\n", t,
                cor(eta_qr[:,t,M], eta_true[:,t]),
                cor(eta_ls[:,t,M], eta_true[:,t]))
    end

    (par_true=par_true, par_qr=par_qr, a_trans_ls=a_trans_ls,
     ll_qr=ll_qr, ll_ls=ll_ls, eta_true=eta_true)
end

# ================================================================
if abspath(PROGRAM_FILE) == @__FILE__
    run_ls_comparison(N=200, maxiter=30, n_draws=100, M=20)
end

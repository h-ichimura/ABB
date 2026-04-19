#=
test_mstep.jl — Test the M-step with true η provided directly
If MLE M-step can't recover truth here, the bug is in the M-step code.
If it can, the bug is in E-step / stacking / interaction.
=#

# Load the main code (will run the comparison at the end, so we redirect)
# Instead, just include the functions we need

include("ABB_three_period.jl")

println("\n","="^60)
println("TEST: M-step with true η (no latent variable problem)")
println("="^60)

# Setup
K=2; L=3; sigma_y=1.0; T=3; N=2000
tau = [0.25, 0.50, 0.75]
par_true = make_true_params_linear(tau=tau, sigma_y=sigma_y, K=K)

println("\nTrue transition parameters:")
@printf("  slopes:     [%.4f, %.4f, %.4f]\n", par_true.a_Q[2,:]...)
@printf("  intercepts: [%.4f, %.4f, %.4f]\n", par_true.a_Q[1,:]...)
@printf("  quadratic:  [%.4f, %.4f, %.4f]\n", par_true.a_Q[3,:]...)
@printf("  b1_Q=%.4f  bL_Q=%.4f\n", par_true.b1_Q, par_true.bL_Q)

# Generate data with no epsilon (y = eta exactly)
y, eta_true = generate_data_abb(N, par_true, tau, sigma_y, K; seed=42)

# Set epsilon quantiles to zero (no transitory shock in this test)
# Actually generate_data_abb adds epsilon. Let's use eta as y instead.
y_clean = copy(eta_true)  # y = eta, no noise

# Create cfg with M=1 (we provide the true eta as the single draw)
M = 1
cfg = Config(N, T, K, L, tau, sigma_y, 1, 1, M, fill(0.05, T))

# Create eta_all as a 3D array with the true eta
eta_all = zeros(N, T, M)
eta_all[:, :, 1] .= eta_true

# ---- Test QR M-step ----
println("\n--- QR M-step with true η ---")
par_qr = copy_params(par_true)
# Perturb starting values
par_qr.a_Q .= par_true.a_Q .* 0.5
par_qr.b1_Q = 3.0; par_qr.bL_Q = 3.0
m_step_qr!(par_qr, eta_all, y_clean, cfg)
@printf("  slopes:     [%.4f, %.4f, %.4f]  (true: 0.8)\n", par_qr.a_Q[2,:]...)
@printf("  intercepts: [%.4f, %.4f, %.4f]  (true: [-0.337, 0, 0.337])\n", par_qr.a_Q[1,:]...)
@printf("  b1_Q=%.4f  bL_Q=%.4f  (true: 2.0, 2.0)\n", par_qr.b1_Q, par_qr.bL_Q)

# Check eps quantiles (should be ~0 since y = eta)
@printf("  eps quantiles: [%.4f, %.4f, %.4f]  (should be ~0)\n", par_qr.a_eps...)

# ---- Test MLE M-step ----
println("\n--- MLE M-step with true η (coordinate descent, no QR warm start) ---")

# Save and replace m_step_mle! with a version that does NOT call QR
function m_step_mle_no_qr!(par::Params, eta_all::Array{Float64,3},
                             y::Matrix{Float64}, cfg::Config)
    K, L = cfg.K, cfg.L
    N, T, M = cfg.N, cfg.T, cfg.M

    # Marginals: sample quantiles (same as QR, this is just the MLE for marginals)
    eta1_all = stack_initial(eta_all, cfg)
    for l in 1:L; par.a_init[l] = quantile(eta1_all, cfg.tau[l]); end
    par.b1_init, par.bL_init = update_tails(eta1_all, par.a_init[1], par.a_init[L])

    eps_all = stack_eps(eta_all, y, cfg)
    for l in 1:L; par.a_eps[l] = quantile(eps_all, cfg.tau[l]); end
    par.a_eps .-= mean(par.a_eps)
    par.b1_eps, par.bL_eps = update_tails(eps_all, par.a_eps[1], par.a_eps[L])

    # Transition: Nelder-Mead on true loglik, NO QR warm start
    eta_t, eta_lag = stack_transition(eta_all, cfg)
    H = hermite_basis(eta_lag, K, cfg.sigma_y)
    n_obs = length(eta_t)
    Q_mat = Matrix{Float64}(undef, n_obs, L)
    q_sorted = Vector{Float64}(undef, L)

    function neg_loglik(theta)
        a = reshape(theta, K+1, L)
        mul!(Q_mat, H, a)

        # Crossing penalty
        cp = 0.0
        @inbounds for j in 1:n_obs
            for l in 1:L-1
                gap = Q_mat[j, l+1] - Q_mat[j, l]
                gap < 0.0 && (cp += gap * gap)
            end
        end

        # Tail rates from correct columns
        r1 = eta_t .- view(Q_mat, :, 1)
        rL = eta_t .- view(Q_mat, :, L)
        ml = r1 .<= 0; mh = rL .>= 0
        s1 = sum(r1[ml]); sL = sum(rL[mh])
        b1v = s1 < -1e-10 ? -count(ml)/s1 : par.b1_Q
        bLv = sL >  1e-10 ?  count(mh)/sL : par.bL_Q
        ll = 0.0
        @inbounds for j in 1:n_obs
            ll += pw_logdens(eta_t[j], view(Q_mat, j, :), cfg.tau, b1v, bLv)
        end
        -ll / n_obs + 100.0 * cp / n_obs
    end

    # Start from current par (perturbed), NOT from QR
    theta0 = vec(copy(par.a_Q))
    @printf("    NM start neg-ll = %.6f\n", neg_loglik(theta0))

    res = optimize(neg_loglik, theta0, NelderMead(),
                   Optim.Options(iterations=10000, f_reltol=1e-10,
                                 show_trace=false))

    @printf("    NM final neg-ll = %.6f  (iters=%d)\n",
            Optim.minimum(res), Optim.iterations(res))

    par.a_Q .= reshape(Optim.minimizer(res), K+1, L)

    # Final tail update
    mul!(Q_mat, H, par.a_Q)
    r1 = eta_t .- view(Q_mat, :, 1); rL = eta_t .- view(Q_mat, :, L)
    ml = r1 .<= 0; mh = rL .>= 0
    s1 = sum(r1[ml]); sL = sum(rL[mh])
    s1 < -1e-10 && (par.b1_Q = -count(ml)/s1)
    sL >  1e-10 && (par.bL_Q =  count(mh)/sL)
    nothing
end

par_mle = copy_params(par_true)
par_mle.a_Q .= par_true.a_Q .* 0.5
par_mle.b1_Q = 3.0; par_mle.bL_Q = 3.0
m_step_mle_no_qr!(par_mle, eta_all, y_clean, cfg)
@printf("  slopes:     [%.4f, %.4f, %.4f]  (true: 0.8)\n", par_mle.a_Q[2,:]...)
@printf("  intercepts: [%.4f, %.4f, %.4f]  (true: [-0.337, 0, 0.337])\n", par_mle.a_Q[1,:]...)
@printf("  quadratic:  [%.4f, %.4f, %.4f]  (true: 0)\n", par_mle.a_Q[3,:]...)
@printf("  b1_Q=%.4f  bL_Q=%.4f  (true: 2.0, 2.0)\n", par_mle.b1_Q, par_mle.bL_Q)
@printf("  eps quantiles: [%.4f, %.4f, %.4f]  (should be ~0)\n", par_mle.a_eps...)

# ---- Compare log-likelihoods ----
println("\n--- Log-likelihood comparison ---")
q_buf = zeros(L)
function avg_loglik(par, y, eta, cfg)
    N = cfg.N
    ll = 0.0
    for i in 1:N
        ll += full_loglik(view(y,i,:), view(eta,i,:), par, cfg, q_buf)
    end
    ll / N
end

ll_true = avg_loglik(par_true, y_clean, eta_true, cfg)
ll_qr   = avg_loglik(par_qr,  y_clean, eta_true, cfg)
ll_mle  = avg_loglik(par_mle, y_clean, eta_true, cfg)
@printf("  loglik at truth: %.6f\n", ll_true)
@printf("  loglik at QR:    %.6f\n", ll_qr)
@printf("  loglik at MLE:   %.6f\n", ll_mle)
@printf("  QR  vs truth: %+.6f\n", ll_qr - ll_true)
@printf("  MLE vs truth: %+.6f\n", ll_mle - ll_true)


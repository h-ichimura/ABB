include("ABB_three_period.jl")

K=2; L=3; sigma_y=1.0; T=3; N=500; M=50
tau = [0.25, 0.50, 0.75]
par_true = make_true_params_linear(tau=tau, sigma_y=sigma_y, K=K)
y, eta_true = generate_data_abb(N, par_true, tau, sigma_y, K; seed=42)
vp = fill(0.05, T)
cfg = Config(N, T, K, L, tau, sigma_y, 10, 200, M, vp)

eta_all = zeros(N, T, M)
for m in 1:M; eta_all[:,:,m] .= 0.6 .* y; end

# Run a few EM iterations and check for crossing at each M-step
global par = init_params(y, cfg)
for iter in 1:10
    global par
    acc = e_step!(eta_all, y, par, cfg)

    # Check crossing before M-step
    eta_t, eta_lag = stack_transition(eta_all, cfg)
    H = hermite_basis(eta_lag, K, sigma_y)
    Q_pre = H * par.a_Q
    n_cross_pre = count(Q_pre[:,1] .>= Q_pre[:,2] .|| Q_pre[:,2] .>= Q_pre[:,3])

    # Run QR M-step
    par_qr = copy_params(par)
    m_step_qr!(par_qr, eta_all, y, cfg)
    Q_qr = H * par_qr.a_Q
    n_cross_qr = count(Q_qr[:,1] .>= Q_qr[:,2] .|| Q_qr[:,2] .>= Q_qr[:,3])

    # Run MLE M-step
    par_mle = copy_params(par)
    m_step_mle!(par_mle, eta_all, y, cfg)
    Q_mle = H * par_mle.a_Q
    n_cross_mle = count(Q_mle[:,1] .>= Q_mle[:,2] .|| Q_mle[:,2] .>= Q_mle[:,3])

    n_obs = size(Q_pre, 1)
    @printf("iter %2d: crossing pre=%d/%d  QR=%d/%d  MLE=%d/%d\n",
            iter, n_cross_pre, n_obs, n_cross_qr, n_obs, n_cross_mle, n_obs)

    # Use MLE params for next iteration
    par = par_mle
end

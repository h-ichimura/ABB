include("ABB_three_period.jl")

K=2; L=3; sigma_y=1.0; T=3; N=500; M=50
tau = [0.25, 0.50, 0.75]
par_true = make_true_params_linear(tau=tau, sigma_y=sigma_y, K=K)
y, eta_true = generate_data_abb(N, par_true, tau, sigma_y, K; seed=42)
vp = fill(0.05, T)
cfg = Config(N, T, K, L, tau, sigma_y, 5, 200, M, vp)

eta_all = zeros(N, T, M)
for m in 1:M; eta_all[:,:,m] .= 0.6 .* y; end

global par = init_params(y, cfg)
q_buf = zeros(L)

for iter in 1:5
    global par
    acc = e_step!(eta_all, y, par, cfg)

    # Compute ll before M-step
    ll_before = sum(full_loglik(view(y,i,:), view(eta_all,i,:,m), par, cfg, q_buf)
                    for m in 1:M, i in 1:N) / (N*M)

    # QR M-step
    par_qr = copy_params(par)
    m_step_qr!(par_qr, eta_all, y, cfg)
    ll_qr = sum(full_loglik(view(y,i,:), view(eta_all,i,:,m), par_qr, cfg, q_buf)
                for m in 1:M, i in 1:N) / (N*M)

    # MLE M-step (starts from QR internally)
    par_mle = copy_params(par)
    m_step_mle!(par_mle, eta_all, y, cfg)
    ll_mle = sum(full_loglik(view(y,i,:), view(eta_all,i,:,m), par_mle, cfg, q_buf)
                 for m in 1:M, i in 1:N) / (N*M)

    @printf("iter %d: ll_before=%.4f  ll_qr=%.4f  ll_mle=%.4f  Δ(QR)=%+.4f  Δ(MLE)=%+.4f  Δ(MLE-QR)=%+.4f\n",
            iter, ll_before, ll_qr, ll_mle,
            ll_qr - ll_before, ll_mle - ll_before, ll_mle - ll_qr)
    @printf("       QR slopes: [%.4f, %.4f, %.4f]  MLE slopes: [%.4f, %.4f, %.4f]\n",
            par_qr.a_Q[2,:]..., par_mle.a_Q[2,:]...)

    par = par_mle
end

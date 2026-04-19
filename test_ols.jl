#=
test_ols.jl — Verify OLS M-step with observed η
=#
include("ABB_three_period.jl")

K=2; L=3; sigma_y=1.0; T=3; N=2000; M=1
tau = [0.25, 0.50, 0.75]
par_true = make_true_params_linear(tau=tau, sigma_y=sigma_y, K=K)

println("True parameters:")
@printf("  slopes:     [%.4f, %.4f, %.4f]\n", par_true.a_Q[2,:]...)
@printf("  intercepts: [%.4f, %.4f, %.4f]\n", par_true.a_Q[1,:]...)
@printf("  quadratic:  [%.4f, %.4f, %.4f]\n", par_true.a_Q[3,:]...)
@printf("  b1_Q=%.4f  bL_Q=%.4f\n", par_true.b1_Q, par_true.bL_Q)

y, eta_true = generate_data_abb(N, par_true, tau, sigma_y, K; seed=42)
y_clean = copy(eta_true)  # no eps

cfg = Config(N, T, K, L, tau, sigma_y, 1, 1, M, fill(0.05, T))
eta_all = zeros(N, T, M)
eta_all[:,:,1] .= eta_true

# --- QR ---
println("\n--- QR M-step ---")
par_qr = copy_params(par_true)
par_qr.a_Q .= par_true.a_Q .* 0.5; par_qr.b1_Q=3.0; par_qr.bL_Q=3.0
m_step_qr!(par_qr, eta_all, y_clean, cfg)
@printf("  slopes:     [%.4f, %.4f, %.4f]\n", par_qr.a_Q[2,:]...)
@printf("  intercepts: [%.4f, %.4f, %.4f]\n", par_qr.a_Q[1,:]...)
@printf("  b1=%.4f  bL=%.4f\n", par_qr.b1_Q, par_qr.bL_Q)

# --- OLS (starting from 0.5*truth -- wrong segments) ---
println("\n--- OLS M-step (start at 0.5*truth) ---")
par_ols = copy_params(par_true)
par_ols.a_Q .= par_true.a_Q .* 0.5; par_ols.b1_Q=3.0; par_ols.bL_Q=3.0
m_step_ols!(par_ols, eta_all, y_clean, cfg)
@printf("  slopes:     [%.4f, %.4f, %.4f]\n", par_ols.a_Q[2,:]...)
@printf("  intercepts: [%.4f, %.4f, %.4f]\n", par_ols.a_Q[1,:]...)
@printf("  quadratic:  [%.4f, %.4f, %.4f]\n", par_ols.a_Q[3,:]...)
@printf("  b1=%.4f  bL=%.4f\n", par_ols.b1_Q, par_ols.bL_Q)

# --- OLS (starting from truth -- correct segment assignments) ---
println("\n--- OLS M-step (par = truth, correct segments) ---")
par_ols2 = copy_params(par_true)  # segments computed from par_true.a_Q = truth
m_step_ols!(par_ols2, eta_all, y_clean, cfg)
@printf("  slopes:     [%.4f, %.4f, %.4f]\n", par_ols2.a_Q[2,:]...)
@printf("  intercepts: [%.4f, %.4f, %.4f]\n", par_ols2.a_Q[1,:]...)
@printf("  quadratic:  [%.4f, %.4f, %.4f]\n", par_ols2.a_Q[3,:]...)
@printf("  b1=%.4f  bL=%.4f\n", par_ols2.b1_Q, par_ols2.bL_Q)

# --- OLS with QR-estimated segments ---
# This mimics the EM: QR ran first (giving par_qr), then OLS uses par_qr for segments
println("\n--- OLS M-step (par = QR estimate, QR-based segments) ---")
par_ols3 = copy_params(par_qr)  # segments computed from QR solution
m_step_ols!(par_ols3, eta_all, y_clean, cfg)
@printf("  slopes:     [%.4f, %.4f, %.4f]\n", par_ols3.a_Q[2,:]...)
@printf("  intercepts: [%.4f, %.4f, %.4f]\n", par_ols3.a_Q[1,:]...)
@printf("  quadratic:  [%.4f, %.4f, %.4f]\n", par_ols3.a_Q[3,:]...)
@printf("  b1=%.4f  bL=%.4f\n", par_ols3.b1_Q, par_ols3.bL_Q)

# --- Log-likelihoods ---
println("\n--- Log-likelihood comparison ---")
q_buf = zeros(L)
function avg_ll(par)
    ll = 0.0
    for i in 1:N
        ll += full_loglik(view(y_clean,i,:), view(eta_true,i,:), par, cfg, q_buf)
    end
    ll / N
end
ll_true = avg_ll(par_true)
ll_qr   = avg_ll(par_qr)
ll_ols  = avg_ll(par_ols)
ll_ols2 = avg_ll(par_ols2)
ll_ols3 = avg_ll(par_ols3)
@printf("  truth:       %.6f\n", ll_true)
@printf("  QR:          %.6f  (vs truth: %+.6f)\n", ll_qr,   ll_qr   - ll_true)
@printf("  OLS(0.5x):   %.6f  (vs truth: %+.6f)\n", ll_ols,  ll_ols  - ll_true)
@printf("  OLS(truth):  %.6f  (vs truth: %+.6f)\n", ll_ols2, ll_ols2 - ll_true)
@printf("  OLS(QR):     %.6f  (vs truth: %+.6f)\n", ll_ols3, ll_ols3 - ll_true)

# --- Segment counts ---
println("\n--- Segment counts ---")
eta_t, eta_lag = stack_transition(eta_all, cfg)
H = hermite_basis(eta_lag, K, sigma_y)
Q_mat = H * par_true.a_Q
n_obs = length(eta_t)
seg = zeros(Int, n_obs)
for j in 1:n_obs
    x = eta_t[j]
    if x <= Q_mat[j,1]; seg[j]=0
    elseif x > Q_mat[j,3]; seg[j]=3
    elseif x <= Q_mat[j,2]; seg[j]=1
    else; seg[j]=2; end
end
for s in 0:3
    @printf("  segment %d: %d obs (%.1f%%)\n", s, count(seg.==s),
            100*count(seg.==s)/n_obs)
end

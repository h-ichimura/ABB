#=
test_coord_descent.jl — Test new coordinate descent MLE M-step
=#
include("ABB_three_period.jl")

K=2; L=3; sigma_y=1.0; T=3; N=2000; M=1
tau = [0.25, 0.50, 0.75]
par_true = make_true_params_linear(tau=tau, sigma_y=sigma_y, K=K)

println("True parameters:")
@printf("  slopes:     [%.4f, %.4f, %.4f]\n", par_true.a_Q[2,:]...)
@printf("  intercepts: [%.4f, %.4f, %.4f]\n", par_true.a_Q[1,:]...)
@printf("  b1_Q=%.4f  bL_Q=%.4f\n", par_true.b1_Q, par_true.bL_Q)

y, eta_true = generate_data_abb(N, par_true, tau, sigma_y, K; seed=42)
y_clean = copy(eta_true)

cfg = Config(N, T, K, L, tau, sigma_y, 1, 1, M, fill(0.05, T))
eta_all = zeros(N, T, M)
eta_all[:,:,1] .= eta_true

q_buf = zeros(L)
function avg_ll(par)
    ll = 0.0
    for i in 1:N
        ll += full_loglik(view(y_clean,i,:), view(eta_true,i,:), par, cfg, q_buf)
    end
    ll / N
end

# --- QR ---
println("\n--- QR ---")
par_qr = copy_params(par_true)
par_qr.a_Q .= par_true.a_Q .* 0.5; par_qr.b1_Q=3.0; par_qr.bL_Q=3.0
m_step_qr!(par_qr, eta_all, y_clean, cfg)
@printf("  slopes:     [%.4f, %.4f, %.4f]\n", par_qr.a_Q[2,:]...)
@printf("  intercepts: [%.4f, %.4f, %.4f]\n", par_qr.a_Q[1,:]...)
@printf("  b1=%.4f  bL=%.4f\n", par_qr.b1_Q, par_qr.bL_Q)
@printf("  loglik: %.6f\n", avg_ll(par_qr))

# --- MLE (coordinate descent) from 0.5x truth ---
println("\n--- MLE (coord descent) from 0.5x truth ---")
par_mle = copy_params(par_true)
par_mle.a_Q .= par_true.a_Q .* 0.5; par_mle.b1_Q=3.0; par_mle.bL_Q=3.0
t = @elapsed m_step_mle!(par_mle, eta_all, y_clean, cfg)
@printf("  slopes:     [%.4f, %.4f, %.4f]\n", par_mle.a_Q[2,:]...)
@printf("  intercepts: [%.4f, %.4f, %.4f]\n", par_mle.a_Q[1,:]...)
@printf("  quadratic:  [%.4f, %.4f, %.4f]\n", par_mle.a_Q[3,:]...)
@printf("  b1=%.4f  bL=%.4f\n", par_mle.b1_Q, par_mle.bL_Q)
@printf("  loglik: %.6f  (%.1f s)\n", avg_ll(par_mle), t)

# --- MLE from truth ---
println("\n--- MLE (coord descent) from truth ---")
par_mle2 = copy_params(par_true)
t2 = @elapsed m_step_mle!(par_mle2, eta_all, y_clean, cfg)
@printf("  slopes:     [%.4f, %.4f, %.4f]\n", par_mle2.a_Q[2,:]...)
@printf("  intercepts: [%.4f, %.4f, %.4f]\n", par_mle2.a_Q[1,:]...)
@printf("  b1=%.4f  bL=%.4f\n", par_mle2.b1_Q, par_mle2.bL_Q)
@printf("  loglik: %.6f  (%.1f s)\n", avg_ll(par_mle2), t2)

# --- Summary ---
println("\n--- Summary ---")
@printf("  truth loglik: %.6f\n", avg_ll(par_true))
@printf("  QR loglik:    %.6f\n", avg_ll(par_qr))
@printf("  MLE(0.5x):    %.6f\n", avg_ll(par_mle))
@printf("  MLE(truth):   %.6f\n", avg_ll(par_mle2))

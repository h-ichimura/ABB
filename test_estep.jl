#=
test_estep.jl — Verify E-step and stacking functions
=#
include("ABB_three_period.jl")

passed = 0; failed = 0
function check(name, cond)
    global passed, failed
    if cond; println("  PASS: $name"); passed+=1
    else;    println("  FAIL: $name"); failed+=1; end
end

K=2; L=3; sigma_y=1.0; T=3; N=100; M=3
tau = [0.25, 0.50, 0.75]
par_true = make_true_params_linear(tau=tau, sigma_y=sigma_y, K=K)
y, eta_true = generate_data_abb(N, par_true, tau, sigma_y, K; seed=42)

# ================================================================
println("="^60)
println("TEST A: stack_transition")
println("="^60)

cfg = Config(N, T, K, L, tau, sigma_y, 1, 1, M, fill(0.05, T))
eta_all = zeros(N, T, M)
for m in 1:M, t in 1:T, i in 1:N
    eta_all[i, t, m] = 100.0*m + 10.0*t + 0.01*i  # unique tag
end

et, el = stack_transition(eta_all, cfg)
n_expected = N*(T-1)*M
check("length = N*(T-1)*M = $(n_expected)", length(et) == n_expected)

# Check first block: m=1, t=2, i=1..N
# eta_t should be eta_all[i, 2, 1] = 100*1 + 10*2 + 0.01*i = 120.01..120.xx
# eta_lag should be eta_all[i, 1, 1] = 100*1 + 10*1 + 0.01*i = 110.01..110.xx
check("et[1] = eta_all[1,2,1]", abs(et[1] - eta_all[1,2,1]) < 1e-10)
check("el[1] = eta_all[1,1,1]", abs(el[1] - eta_all[1,1,1]) < 1e-10)
check("et[N] = eta_all[N,2,1]", abs(et[N] - eta_all[N,2,1]) < 1e-10)

# Second block: m=1, t=3, i=1..N
check("et[N+1] = eta_all[1,3,1]", abs(et[N+1] - eta_all[1,3,1]) < 1e-10)
check("el[N+1] = eta_all[1,2,1]", abs(el[N+1] - eta_all[1,2,1]) < 1e-10)

# Third block: m=2, t=2, i=1..N
idx_m2_t2 = N*(T-1)*1 + 1  # first entry of m=2
check("et[m2_t2] = eta_all[1,2,2]", abs(et[idx_m2_t2] - eta_all[1,2,2]) < 1e-10)
check("el[m2_t2] = eta_all[1,1,2]", abs(el[idx_m2_t2] - eta_all[1,1,2]) < 1e-10)

# ================================================================
println("\n","="^60)
println("TEST B: stack_initial")
println("="^60)

ei = stack_initial(eta_all, cfg)
check("length = N*M = $(N*M)", length(ei) == N*M)
check("ei[1] = eta_all[1,1,1]", abs(ei[1] - eta_all[1,1,1]) < 1e-10)
check("ei[N] = eta_all[N,1,1]", abs(ei[N] - eta_all[N,1,1]) < 1e-10)
check("ei[N+1] = eta_all[1,1,2]", abs(ei[N+1] - eta_all[1,1,2]) < 1e-10)

# ================================================================
println("\n","="^60)
println("TEST C: stack_eps")
println("="^60)

y_test = zeros(N, T)
for t in 1:T, i in 1:N
    y_test[i, t] = 1000.0*t + i  # unique
end

ee = stack_eps(eta_all, y_test, cfg)
check("length = N*T*M = $(N*T*M)", length(ee) == N*T*M)
# First entry: m=1, t=1, i=1 => y[1,1] - eta_all[1,1,1]
expected_e1 = y_test[1,1] - eta_all[1,1,1]
check("ee[1] = y[1,1]-eta_all[1,1,1]", abs(ee[1] - expected_e1) < 1e-10)

# ================================================================
println("\n","="^60)
println("TEST D: E-step produces correct posterior draws")
println("="^60)
println("  With true params and y = eta + eps, run MH.")
println("  Check that E[eta|y] is close to posterior mean.")
println("  For large n_draws, the MH chain should mix well.")

N2 = 200; M2 = 1
cfg2 = Config(N2, T, K, L, tau, sigma_y, 1, 500, M2, fill(0.08, T))
y2, eta2_true = generate_data_abb(N2, par_true, tau, sigma_y, K; seed=99)

# Run many E-steps and collect draws to estimate posterior mean
n_reps = 50
eta_draws = zeros(N2, T, n_reps)
eta_all2 = zeros(N2, T, M2)
eta_all2[:,:,1] .= 0.5 .* y2  # initial guess

for rep in 1:n_reps
    e_step!(eta_all2, y2, par_true, cfg2)
    eta_draws[:,:,rep] .= eta_all2[:,:,1]
end

# Posterior mean estimate
eta_post_mean = mean(eta_draws, dims=3)[:,:,1]

# Check correlation with true eta
for t in 1:T
    c = cor(eta_post_mean[:,t], eta2_true[:,t])
    @printf("  t=%d: corr(E[eta|y], eta_true) = %.4f\n", t, c)
    check("E-step posterior corr > 0.85 at t=$t", c > 0.85)
end

# Check that posterior mean is between y and 0
# (since eta = y - eps, and eps has mean ~0, posterior mean should be close to y
#  but shrunk toward unconditional mean)
mean_ratio = mean(abs.(eta_post_mean)) / mean(abs.(y2))
@printf("  |E[eta|y]| / |y| = %.4f (should be < 1, shrinkage)\n", mean_ratio)
check("posterior shrinkage", mean_ratio < 1.0)

# ================================================================
println("\n","="^60)
println("TEST E: One full EM iteration with true params as start")
println("="^60)
println("  Start at truth, run 1 E-step + M-step.")
println("  Parameters should stay near truth.")

N3 = 500; M3 = 50
cfg3 = Config(N3, T, K, L, tau, sigma_y, 1, 300, M3, fill(0.08, T))
y3, eta3_true = generate_data_abb(N3, par_true, tau, sigma_y, K; seed=77)

eta_all3 = zeros(N3, T, M3)
for m in 1:M3; eta_all3[:,:,m] .= 0.6 .* y3; end

par_test = copy_params(par_true)

# Run E-step
println("  Running E-step...")
acc = e_step!(eta_all3, y3, par_test, cfg3)
@printf("  Acceptance rates: %s\n", join([@sprintf("%.2f",a) for a in acc], "/"))

# Run QR M-step
par_qr_test = copy_params(par_test)
m_step_qr!(par_qr_test, eta_all3, y3, cfg3)
@printf("  QR slopes:  [%.4f, %.4f, %.4f]  (true: 0.8)\n", par_qr_test.a_Q[2,:]...)
@printf("  QR intcpts: [%.4f, %.4f, %.4f]\n", par_qr_test.a_Q[1,:]...)
@printf("  QR b1=%.4f bL=%.4f  (true: 2.0)\n", par_qr_test.b1_Q, par_qr_test.bL_Q)
@printf("  QR eps_q:   [%.4f, %.4f, %.4f]  (true: [-0.202, 0, 0.202])\n", par_qr_test.a_eps...)

for l in 1:L
    check("QR slope $l near 0.8", abs(par_qr_test.a_Q[2,l]/sigma_y - 0.8) < 0.15)
end

# Run MLE M-step (uses QR warm start internally)
par_mle_test = copy_params(par_test)
m_step_mle!(par_mle_test, eta_all3, y3, cfg3)
@printf("  MLE slopes: [%.4f, %.4f, %.4f]  (true: 0.8)\n", par_mle_test.a_Q[2,:]...)
@printf("  MLE intcpts:[%.4f, %.4f, %.4f]\n", par_mle_test.a_Q[1,:]...)
@printf("  MLE b1=%.4f bL=%.4f\n", par_mle_test.b1_Q, par_mle_test.bL_Q)
@printf("  MLE eps_q:  [%.4f, %.4f, %.4f]\n", par_mle_test.a_eps...)

# Compare log-likelihoods
q_buf = zeros(L)
function avg_ll(par, y, eta_all, cfg)
    ll = 0.0
    for m in 1:cfg.M, i in 1:cfg.N
        ll += full_loglik(view(y,i,:), view(eta_all,i,:,m), par, cfg, q_buf)
    end
    ll / (cfg.N * cfg.M)
end

ll_true = avg_ll(par_true, y3, eta_all3, cfg3)
ll_qr   = avg_ll(par_qr_test, y3, eta_all3, cfg3)
ll_mle  = avg_ll(par_mle_test, y3, eta_all3, cfg3)
@printf("\n  Avg loglik:  truth=%.4f  QR=%.4f  MLE=%.4f\n", ll_true, ll_qr, ll_mle)
@printf("  QR  vs truth: %+.4f\n", ll_qr - ll_true)
@printf("  MLE vs truth: %+.4f\n", ll_mle - ll_true)
@printf("  MLE vs QR:    %+.4f\n", ll_mle - ll_qr)

check("MLE loglik >= QR loglik", ll_mle >= ll_qr - 0.001)

# ================================================================
println("\n","="^60)
@printf("SUMMARY: %d passed, %d failed\n", passed, failed)
println("="^60)

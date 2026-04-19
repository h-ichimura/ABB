#=
test_mle_variants.jl — Compare three MLE M-step implementations

Approach 1a: IPNewton (Optim.jl) — full 9-param problem with linear constraints
Approach 1b: Coordinate descent with analytical feasible intervals
Approach 2:  JuMP.jl + Ipopt
=#

include("mle_variants.jl")

using Printf

# ================================================================
#  Setup: generate data with known truth
# ================================================================
K = 2; L = 3; sigma_y = 1.0; T = 3; N = 2000
tau = [0.25, 0.50, 0.75]
par_true = make_true_params_linear(tau=tau, sigma_y=sigma_y, K=K)

println("="^70)
println("  COMPARING MLE M-STEP VARIANTS (observed η, no E-step)")
println("="^70)
println("\nTrue transition parameters:")
@printf("  slopes:     [%.4f, %.4f, %.4f]\n", par_true.a_Q[2,:]...)
@printf("  intercepts: [%.4f, %.4f, %.4f]\n", par_true.a_Q[1,:]...)
@printf("  quadratic:  [%.4f, %.4f, %.4f]\n", par_true.a_Q[3,:]...)
@printf("  b1_Q=%.4f  bL_Q=%.4f\n", par_true.b1_Q, par_true.bL_Q)

y, eta_true = generate_data_abb(N, par_true, tau, sigma_y, K; seed=42)

# Use η itself as y (no eps, no latent variable problem)
y_clean = copy(eta_true)
M = 1
cfg = Config(N, T, K, L, tau, sigma_y, 1, 1, M, fill(0.05, T))
eta_all = zeros(N, T, M)
eta_all[:, :, 1] .= eta_true

# ================================================================
#  Helper: full log-likelihood at given parameters
# ================================================================
function eval_loglik(par, y, eta, cfg)
    N = cfg.N
    q_buf = zeros(cfg.L)
    ll = 0.0
    for i in 1:N
        ll += full_loglik(view(y, i, :), view(eta, i, :), par, cfg, q_buf)
    end
    ll / N
end

# ================================================================
#  Baseline: QR
# ================================================================
println("\n", "-"^70)
println("  QR (baseline)")
println("-"^70)
par_qr = copy_params(par_true)
par_qr.a_Q .= par_true.a_Q .* 0.5  # perturb start
par_qr.b1_Q = 3.0; par_qr.bL_Q = 3.0
t_qr = @elapsed m_step_qr!(par_qr, eta_all, y_clean, cfg)
ll_qr = eval_loglik(par_qr, y_clean, eta_true, cfg)
@printf("  slopes:     [%.4f, %.4f, %.4f]\n", par_qr.a_Q[2,:]...)
@printf("  intercepts: [%.4f, %.4f, %.4f]\n", par_qr.a_Q[1,:]...)
@printf("  quadratic:  [%.4f, %.4f, %.4f]\n", par_qr.a_Q[3,:]...)
@printf("  b1_Q=%.4f  bL_Q=%.4f\n", par_qr.b1_Q, par_qr.bL_Q)
@printf("  loglik: %.6f  (truth: %.6f)\n", ll_qr, eval_loglik(par_true, y_clean, eta_true, cfg))
@printf("  time: %.2f s\n", t_qr)

# Check if QR satisfies non-crossing
eta_t_stk, eta_lag_stk = stack_transition(eta_all, cfg)
H_stk = hermite_basis(eta_lag_stk, K, sigma_y)
qr_ok = check_non_crossing(par_qr.a_Q, H_stk)
println("  non-crossing: ", qr_ok)

# ================================================================
#  Approach 1a: IPNewton
# ================================================================
println("\n", "-"^70)
println("  Approach 1a: IPNewton (Optim.jl)")
println("-"^70)
par_ipn = copy_params(par_true)
par_ipn.a_Q .= par_true.a_Q .* 0.5
par_ipn.b1_Q = 3.0; par_ipn.bL_Q = 3.0

result_1a = try
    t_1a = @elapsed obj_1a = m_step_mle_ipnewton!(par_ipn, eta_all, y_clean, cfg)
    ll_1a = eval_loglik(par_ipn, y_clean, eta_true, cfg)
    @printf("  slopes:     [%.4f, %.4f, %.4f]\n", par_ipn.a_Q[2,:]...)
    @printf("  intercepts: [%.4f, %.4f, %.4f]\n", par_ipn.a_Q[1,:]...)
    @printf("  quadratic:  [%.4f, %.4f, %.4f]\n", par_ipn.a_Q[3,:]...)
    @printf("  b1_Q=%.4f  bL_Q=%.4f\n", par_ipn.b1_Q, par_ipn.bL_Q)
    @printf("  loglik: %.6f  (obj at end: %.6f)\n", ll_1a, -obj_1a)
    @printf("  non-crossing: %s\n", check_non_crossing(par_ipn.a_Q, H_stk))
    @printf("  time: %.2f s\n", t_1a)
    (par=copy_params(par_ipn), loglik=ll_1a, time=t_1a, ok=true)
catch e
    println("  FAILED: ", e)
    (ok=false,)
end

# ================================================================
#  Approach 1b: Coordinate descent with feasible intervals
# ================================================================
println("\n", "-"^70)
println("  Approach 1b: Coordinate descent with feasible intervals")
println("-"^70)
par_cd = copy_params(par_true)
par_cd.a_Q .= par_true.a_Q .* 0.5
par_cd.b1_Q = 3.0; par_cd.bL_Q = 3.0

result_1b = try
    t_1b = @elapsed obj_1b = m_step_mle_cdfeas!(par_cd, eta_all, y_clean, cfg; verbose=true)
    ll_1b = eval_loglik(par_cd, y_clean, eta_true, cfg)
    @printf("  slopes:     [%.4f, %.4f, %.4f]\n", par_cd.a_Q[2,:]...)
    @printf("  intercepts: [%.4f, %.4f, %.4f]\n", par_cd.a_Q[1,:]...)
    @printf("  quadratic:  [%.4f, %.4f, %.4f]\n", par_cd.a_Q[3,:]...)
    @printf("  b1_Q=%.4f  bL_Q=%.4f\n", par_cd.b1_Q, par_cd.bL_Q)
    @printf("  loglik: %.6f  (obj at end: %.6f)\n", ll_1b, -obj_1b)
    @printf("  non-crossing: %s\n", check_non_crossing(par_cd.a_Q, H_stk))
    @printf("  time: %.2f s\n", t_1b)
    (par=copy_params(par_cd), loglik=ll_1b, time=t_1b, ok=true)
catch e
    println("  FAILED: ", e)
    (ok=false,)
end

# ================================================================
#  Approach 2: JuMP + Ipopt
# ================================================================
println("\n", "-"^70)
println("  Approach 2: JuMP + Ipopt")
println("-"^70)
par_jump = copy_params(par_true)
par_jump.a_Q .= par_true.a_Q .* 0.5
par_jump.b1_Q = 3.0; par_jump.bL_Q = 3.0

result_2 = try
    t_2 = @elapsed obj_2 = m_step_mle_jump!(par_jump, eta_all, y_clean, cfg)
    ll_2 = eval_loglik(par_jump, y_clean, eta_true, cfg)
    @printf("  slopes:     [%.4f, %.4f, %.4f]\n", par_jump.a_Q[2,:]...)
    @printf("  intercepts: [%.4f, %.4f, %.4f]\n", par_jump.a_Q[1,:]...)
    @printf("  quadratic:  [%.4f, %.4f, %.4f]\n", par_jump.a_Q[3,:]...)
    @printf("  b1_Q=%.4f  bL_Q=%.4f\n", par_jump.b1_Q, par_jump.bL_Q)
    @printf("  loglik: %.6f\n", ll_2)
    @printf("  non-crossing: %s\n", check_non_crossing(par_jump.a_Q, H_stk))
    @printf("  time: %.2f s\n", t_2)
    (par=copy_params(par_jump), loglik=ll_2, time=t_2, ok=true)
catch e
    println("  FAILED: ", e)
    (ok=false,)
end

# ================================================================
#  Summary
# ================================================================
println("\n", "="^70)
println("  SUMMARY")
println("="^70)
ll_true = eval_loglik(par_true, y_clean, eta_true, cfg)
@printf("True loglik: %.6f\n", ll_true)
@printf("QR loglik:   %.6f (diff: %+.6f, time %.2fs)\n",
        ll_qr, ll_qr - ll_true, t_qr)

for (name, r) in [("1a IPNewton", result_1a),
                   ("1b CD+feas",  result_1b),
                   ("2  JuMP+Ipopt", result_2)]
    if r.ok
        @printf("%s loglik: %.6f (diff: %+.6f, time %.2fs)\n",
                name, r.loglik, r.loglik - ll_true, r.time)
    else
        @printf("%s: FAILED\n", name)
    end
end

println("\nDone.")

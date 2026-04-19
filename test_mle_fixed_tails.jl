#=
test_mle_fixed_tails.jl — Does profiling tail rates cause the MLE slope bias?

Compare: (1) MLE with profiled tails (current)
         (2) MLE with tails FIXED at truth
         (3) MLE with tails fixed at QR estimates

If (2) fixes the bias, the profiled tail rates are the culprit.
If (2) still shows bias, the issue is fundamental to CDLL + MCEM.
=#

include("ABB_three_period.jl")
using Printf, Statistics

K = 2; L = 3; sigma_y = 1.0; T = 3
tau = [0.25, 0.50, 0.75]
par_true = make_true_params_linear(tau=tau, sigma_y=sigma_y, K=K)

N = 500; M = 50; S = 50
cfg = Config(N, T, K, L, tau, sigma_y, S, 200, M, fill(0.05, T))
y, eta_true = generate_data_abb(N, par_true, tau, sigma_y, K; seed=42)

# Do one E-step at truth to get draws
eta_all = zeros(N, T, M)
for m in 1:M; eta_all[:,:,m] .= 0.6 .* y; end
par = copy_params(par_true)
acc = e_step!(eta_all, y, par, cfg)

eta_t, eta_lag = stack_transition(eta_all, cfg)
H = hermite_basis(eta_lag, K, sigma_y)
n_obs = length(eta_t)

println("="^70)
println("  DIAGNOSTIC: Does tail profiling cause MLE slope bias?")
println("="^70)
@printf("  n_obs = %d, E-step from truth\n", n_obs)

# ── QR baseline ──────────────────────────────────────────────────
par_qr = copy_params(par_true)
par_qr.a_Q .= par_true.a_Q .* 0.5
par_qr.b1_Q = 3.0; par_qr.bL_Q = 3.0
m_step_qr!(par_qr, eta_all, y, cfg)
println("\nQR:")
@printf("  slopes: [%.4f, %.4f, %.4f]\n", par_qr.a_Q[2,:]...)
@printf("  b1=%.4f, bL=%.4f\n", par_qr.b1_Q, par_qr.bL_Q)

# ── MLE with profiled tails (current implementation) ─────────────
par_mle = copy_params(par_true)
par_mle.a_Q .= par_true.a_Q .* 0.5
par_mle.b1_Q = 3.0; par_mle.bL_Q = 3.0
m_step_mle!(par_mle, eta_all, y, cfg)
println("\nMLE (profiled tails):")
@printf("  slopes: [%.4f, %.4f, %.4f]\n", par_mle.a_Q[2,:]...)
@printf("  b1=%.4f, bL=%.4f\n", par_mle.b1_Q, par_mle.bL_Q)

# ── MLE with tails FIXED at truth ────────────────────────────────
function m_step_mle_fixed_tails!(par, eta_all, y, cfg, b1_fix, bL_fix)
    K, L = cfg.K, cfg.L
    eta1_all = stack_initial(eta_all, cfg)
    for l in 1:L; par.a_init[l] = quantile(eta1_all, cfg.tau[l]); end
    par.b1_init, par.bL_init = update_tails(eta1_all, par.a_init[1], par.a_init[L])
    eps_all = stack_eps(eta_all, y, cfg)
    for l in 1:L; par.a_eps[l] = quantile(eps_all, cfg.tau[l]); end
    par.a_eps .-= mean(par.a_eps)
    par.b1_eps, par.bL_eps = update_tails(eps_all, par.a_eps[1], par.a_eps[L])

    eta_t, eta_lag = stack_transition(eta_all, cfg)
    H = hermite_basis(eta_lag, K, cfg.sigma_y)
    n_obs = length(eta_t)

    m_step_qr!(par, eta_all, y, cfg)
    a_cur = copy(par.a_Q)

    # Neg-CDLL with FIXED tail rates (no profiling)
    function neg_cdll_fixed(a_Q_mat)
        Q_mat = H * a_Q_mat
        ll = 0.0
        @inbounds for j in 1:n_obs
            ll += pw_logdens(eta_t[j], view(Q_mat, j, :), cfg.tau, b1_fix, bL_fix)
        end
        -ll / n_obs
    end

    # Feasible interval (same as m_step_mle! but simplified)
    function feasible_interval(k, l_target)
        lo = -Inf; hi = Inf; cur = a_cur[k, l_target]
        if l_target < L
            for j in 1:n_obs
                g = 0.0
                for m in 1:K+1; g += H[j,m] * (a_cur[m, l_target+1] - a_cur[m, l_target]); end
                hjk = H[j, k]
                if hjk > 1e-14;     hi = min(hi, cur + g / hjk)
                elseif hjk < -1e-14; lo = max(lo, cur + g / hjk)
                end
            end
        end
        if l_target > 1
            for j in 1:n_obs
                g = 0.0
                for m in 1:K+1; g += H[j,m] * (a_cur[m, l_target] - a_cur[m, l_target-1]); end
                hjk = H[j, k]
                if hjk > 1e-14;     lo = max(lo, cur - g / hjk)
                elseif hjk < -1e-14; hi = min(hi, cur - g / hjk)
                end
            end
        end
        (lo, hi)
    end

    for cycle in 1:8
        max_step = 0.0
        for l in 1:L, k in 1:K+1
            cur = a_cur[k, l]
            lo_f, hi_f = feasible_interval(k, l)
            w = cycle <= 2 ? 0.5 : (cycle <= 5 ? 0.2 : 0.05)
            lo = max(lo_f + 1e-8, cur - w)
            hi = min(hi_f - 1e-8, cur + w)
            lo >= hi && continue
            f1d(v) = (a_cur[k,l] = v; neg_cdll_fixed(a_cur))
            res = optimize(f1d, lo, hi)
            nv = Optim.minimizer(res)
            max_step = max(max_step, abs(nv - cur))
            a_cur[k, l] = nv
        end
        max_step < 1e-6 && break
    end
    par.a_Q .= a_cur
    par.b1_Q = b1_fix; par.bL_Q = bL_fix
end

par_fix_true = copy_params(par_true)
par_fix_true.a_Q .= par_true.a_Q .* 0.5
par_fix_true.b1_Q = 3.0; par_fix_true.bL_Q = 3.0
m_step_mle_fixed_tails!(par_fix_true, eta_all, y, cfg,
                         par_true.b1_Q, par_true.bL_Q)
println("\nMLE (tails fixed at truth: b1=2.0, bL=2.0):")
@printf("  slopes: [%.4f, %.4f, %.4f]\n", par_fix_true.a_Q[2,:]...)

# ── MLE with tails fixed at QR estimates ─────────────────────────
par_fix_qr = copy_params(par_true)
par_fix_qr.a_Q .= par_true.a_Q .* 0.5
par_fix_qr.b1_Q = 3.0; par_fix_qr.bL_Q = 3.0
m_step_mle_fixed_tails!(par_fix_qr, eta_all, y, cfg,
                         par_qr.b1_Q, par_qr.bL_Q)
println("\nMLE (tails fixed at QR: b1=$(round(par_qr.b1_Q,digits=3)), bL=$(round(par_qr.bL_Q,digits=3))):")
@printf("  slopes: [%.4f, %.4f, %.4f]\n", par_fix_qr.a_Q[2,:]...)

println("\n", "="^70)
println("  SUMMARY")
println("="^70)
@printf("  True:              [0.8000, 0.8000, 0.8000]\n")
@printf("  QR:                [%.4f, %.4f, %.4f]\n", par_qr.a_Q[2,:]...)
@printf("  MLE profiled:      [%.4f, %.4f, %.4f]\n", par_mle.a_Q[2,:]...)
@printf("  MLE fixed@truth:   [%.4f, %.4f, %.4f]\n", par_fix_true.a_Q[2,:]...)
@printf("  MLE fixed@QR:      [%.4f, %.4f, %.4f]\n", par_fix_qr.a_Q[2,:]...)

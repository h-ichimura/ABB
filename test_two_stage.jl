#=
test_two_stage.jl — Two-stage QR → gap MLE estimator

Stage 1: Run full QR EM for S=50 iterations. Take ergodic avg as theta_QR.
Stage 2: Run full gap MLE EM starting from theta_QR for another S=50 iterations.
         Take ergodic avg as theta_MLE.

If the MLE slope bias in single-stage is due to early-iteration feedback from
cold starts, the two-stage version should recover close-to-truth slopes.
=#

include("mle_gap.jl")
using Printf, Statistics, Random, Serialization

K = 2; L = 3; sigma_y = 1.0; T = 3
tau = [0.25, 0.50, 0.75]
par_true = make_true_params_linear(tau=tau, sigma_y=sigma_y, K=K)
N = 500; M = 50; S = 50

cfg = Config(N, T, K, L, tau, sigma_y, S, 200, M, fill(0.05, T))
y, eta_true = generate_data_abb(N, par_true, tau, sigma_y, K; seed=42)

println("="^70)
println("  Two-stage estimator: QR EM → gap MLE EM (seed=42)")
println("="^70)

# ── Stage 1: QR EM ──────────────────────────────────────────────
println("\n--- Stage 1: QR EM ---")
t_qr = @elapsed (par_qr, _, _, ll_qr, hist_qr) = estimate(y, cfg; method=:qr, verbose=false)
@printf("  Ergodic avg slopes: [%.4f, %.4f, %.4f], t=%.1fs\n",
        par_qr.a_Q[2,:]..., t_qr)
@printf("  Ergodic avg intercepts: [%.4f, %.4f, %.4f]\n", par_qr.a_Q[1,:]...)
@printf("  b1_Q=%.4f, bL_Q=%.4f\n", par_qr.b1_Q, par_qr.bL_Q)

# ── Stage 2: gap MLE EM starting from Stage 1 ergodic avg ──────
println("\n--- Stage 2: gap MLE EM starting from QR ergodic avg ---")

par = copy_params(par_qr)  # WARM START FROM QR CONVERGED
eta_all = zeros(N, T, M)
for m in 1:M; eta_all[:,:,m] .= 0.6 .* y; end

ll_hist = zeros(S)
q_buf = zeros(L)
hist = ParamHistory(
    zeros(K+1,L,S), zeros(S), zeros(S),
    zeros(L,S), zeros(S), zeros(S),
    zeros(L,S), zeros(S), zeros(S))

t_stage2 = @elapsed begin
    for iter in 1:S
        acc = e_step!(eta_all, y, par, cfg)
        m_step_mle_gap!(par, eta_all, y, cfg)
        hist.a_Q[:,:,iter].=par.a_Q; hist.b1_Q[iter]=par.b1_Q; hist.bL_Q[iter]=par.bL_Q
        hist.a_init[:,iter].=par.a_init; hist.b1_init[iter]=par.b1_init; hist.bL_init[iter]=par.bL_init
        hist.a_eps[:,iter].=par.a_eps; hist.b1_eps[iter]=par.b1_eps; hist.bL_eps[iter]=par.bL_eps

        ll = 0.0
        for m in 1:M, i in 1:N
            ll += full_loglik(view(y,i,:), view(eta_all,i,:,m), par, cfg, q_buf)
        end
        ll_hist[iter] = ll / (N * M)
        if iter % 10 == 0 || iter <= 3
            @printf("  [MLE] %3d/%d | ll %8.4f | acc %s | slopes=[%.3f,%.3f,%.3f]\n",
                    iter, S, ll_hist[iter],
                    join([@sprintf("%.2f", a) for a in acc], "/"),
                    par.a_Q[2,:]...)
        end
    end
end

# Ergodic average over last S/2 iterations
S2 = div(S, 2); rng = (S - S2 + 1):S
slopes_avg = vec(mean(hist.a_Q[2, :, rng], dims=2))
intercepts_avg = vec(mean(hist.a_Q[1, :, rng], dims=2))

# ── Summary ─────────────────────────────────────────────────────
println("\n", "="^70)
println("  COMPARISON: Single-stage vs Two-stage")
println("="^70)
@printf("  True slopes:                [0.8000, 0.8000, 0.8000]\n")
@printf("  QR only (Stage 1):          [%.4f, %.4f, %.4f]  mean=%.4f\n",
        par_qr.a_Q[2,:]..., mean(par_qr.a_Q[2,:]))
@printf("  Gap MLE two-stage (Stage 2): [%.4f, %.4f, %.4f]  mean=%.4f\n",
        slopes_avg..., mean(slopes_avg))

println("\n  Intercepts:")
@printf("  True:                       [-0.3371, 0.0000, 0.3371]\n")
@printf("  QR only:                    [%.4f, %.4f, %.4f]\n", par_qr.a_Q[1,:]...)
@printf("  Two-stage MLE:              [%.4f, %.4f, %.4f]\n", intercepts_avg...)

@printf("\n  Times: Stage 1 QR = %.1fs, Stage 2 gap MLE = %.1fs, total = %.1fs\n",
        t_qr, t_stage2, t_qr + t_stage2)

# Percentile table for Stage 2
println("\nStage 2 slope percentiles across last 25 iterations:")
@printf("  %5s  %6s  %6s  %6s  %6s  %6s\n", "tau", "p10", "p25", "p50", "p75", "p90")
for l in 1:L
    v = vec(hist.a_Q[2, l, rng])
    pv = quantile(v, [0.1, 0.25, 0.5, 0.75, 0.9])
    @printf("  %5.2f  %6.4f  %6.4f  %6.4f  %6.4f  %6.4f\n",
            tau[l], pv...)
end

# Save results
serialize("results_two_stage.jls",
          (par_qr=par_qr, hist_qr=hist_qr, ll_qr=ll_qr,
           par_mle=par, hist_mle=hist, ll_mle=ll_hist,
           slopes_avg=slopes_avg, intercepts_avg=intercepts_avg))
println("\nSaved results_two_stage.jls")

#=
test_full_em_gap.jl — Run full EM with gap-reparameterized MLE M-step

Compare QR vs std MLE vs gap MLE on full EM at N=500, S=50.
See if gap MLE recovers persistence better than std MLE.
=#

include("mle_gap.jl")

# Replace m_step_mle! dispatch for testing
# We'll run estimate() with a custom method by modifying the estimate loop inline
# ... actually simpler: run estimate() for QR and std MLE, then manually run full EM with gap

using Printf, Statistics, Random

K = 2; L = 3; sigma_y = 1.0; T = 3
tau = [0.25, 0.50, 0.75]
par_true = make_true_params_linear(tau=tau, sigma_y=sigma_y, K=K)
N = 500; M = 50; S = 50

cfg = Config(N, T, K, L, tau, sigma_y, S, 200, M, fill(0.05, T))
y, eta_true = generate_data_abb(N, par_true, tau, sigma_y, K; seed=42)

println("="^70)
println("  FULL EM comparison: QR vs std MLE vs gap MLE (N=$N, S=$S, M=$M)")
println("="^70)

function run_em(y, cfg, m_step_fn!; label="method")
    par = init_params(y, cfg)
    N, T, M = cfg.N, cfg.T, cfg.M
    S = cfg.maxiter; K, L = cfg.K, cfg.L
    eta_all = zeros(N, T, M)
    for m in 1:M; eta_all[:,:,m] .= 0.6 .* y; end

    ll_hist = zeros(S)
    q_buf = zeros(L)
    hist = ParamHistory(
        zeros(K+1,L,S), zeros(S), zeros(S),
        zeros(L,S), zeros(S), zeros(S),
        zeros(L,S), zeros(S), zeros(S))

    for iter in 1:S
        acc = e_step!(eta_all, y, par, cfg)
        m_step_fn!(par, eta_all, y, cfg)

        # Save history
        hist.a_Q[:,:,iter].=par.a_Q; hist.b1_Q[iter]=par.b1_Q; hist.bL_Q[iter]=par.bL_Q
        hist.a_init[:,iter].=par.a_init; hist.b1_init[iter]=par.b1_init; hist.bL_init[iter]=par.bL_init
        hist.a_eps[:,iter].=par.a_eps; hist.b1_eps[iter]=par.b1_eps; hist.bL_eps[iter]=par.bL_eps

        ll = 0.0
        for m in 1:M, i in 1:N
            ll += full_loglik(view(y,i,:), view(eta_all,i,:,m), par, cfg, q_buf)
        end
        ll_hist[iter] = ll / (N * M)
        if iter % 10 == 0 || iter <= 3
            @printf("  [%s] %3d/%d | ll %8.4f | acc %s\n",
                    label, iter, cfg.maxiter, ll_hist[iter],
                    join([@sprintf("%.2f", a) for a in acc], "/"))
        end
    end

    # Ergodic average over last S/2 iterations (same as estimate())
    S2 = div(S, 2); rng = (S - S2 + 1):S
    par_avg = Params(
        dropdims(mean(hist.a_Q[:,:,rng], dims=3), dims=3),
        mean(hist.b1_Q[rng]), mean(hist.bL_Q[rng]),
        vec(mean(hist.a_init[:,rng], dims=2)),
        mean(hist.b1_init[rng]), mean(hist.bL_init[rng]),
        vec(mean(hist.a_eps[:,rng], dims=2)),
        mean(hist.b1_eps[rng]), mean(hist.bL_eps[rng]))

    par_avg, ll_hist, hist
end

# QR
println("\n--- QR ---")
t_qr = @elapsed (par_qr, _, _, ll_qr, _) = estimate(y, cfg; method=:qr, verbose=false)
@printf("  final ll=%.4f, slopes=[%.4f,%.4f,%.4f], t=%.1fs\n",
        mean(ll_qr[end-24:end]), par_qr.a_Q[2,:]..., t_qr)

# std MLE
println("\n--- std MLE ---")
t_std = @elapsed (par_std, _, _, ll_std, _) = estimate(y, cfg; method=:mle, verbose=false)
@printf("  final ll=%.4f, slopes=[%.4f,%.4f,%.4f], t=%.1fs\n",
        mean(ll_std[end-24:end]), par_std.a_Q[2,:]..., t_std)

# Gap MLE (using run_em helper with gap M-step)
println("\n--- gap MLE ---")
t_gap = @elapsed par_gap, ll_gap, hist_gap = run_em(y, cfg, m_step_mle_gap!; label="gap")
@printf("  final ll=%.4f, slopes=[%.4f,%.4f,%.4f] (ergodic avg), t=%.1fs\n",
        mean(ll_gap[end-24:end]), par_gap.a_Q[2,:]..., t_gap)

println("\n", "="^70)
println("  SUMMARY (true slope = 0.8000 for all quantile levels)")
println("="^70)
@printf("  QR:      slopes=[%.4f,%.4f,%.4f]  mean=%.4f  t=%.1fs\n",
        par_qr.a_Q[2,:]..., mean(par_qr.a_Q[2,:]), t_qr)
@printf("  std MLE: slopes=[%.4f,%.4f,%.4f]  mean=%.4f  t=%.1fs\n",
        par_std.a_Q[2,:]..., mean(par_std.a_Q[2,:]), t_std)
@printf("  gap MLE: slopes=[%.4f,%.4f,%.4f]  mean=%.4f  t=%.1fs\n",
        par_gap.a_Q[2,:]..., mean(par_gap.a_Q[2,:]), t_gap)

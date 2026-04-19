#=
test_gap_ergodic.jl — Run gap MLE full EM with proper ergodic averaging
=#

include("mle_gap.jl")
using Printf, Statistics

K = 2; L = 3; sigma_y = 1.0; T = 3
tau = [0.25, 0.50, 0.75]
par_true = make_true_params_linear(tau=tau, sigma_y=sigma_y, K=K)
N = 500; M = 50; S = 50

cfg = Config(N, T, K, L, tau, sigma_y, S, 200, M, fill(0.05, T))
y, eta_true = generate_data_abb(N, par_true, tau, sigma_y, K; seed=42)

println("="^70)
println("  Gap MLE full EM with ergodic averaging (seed=42, N=500, S=50, M=50)")
println("="^70)

par = init_params(y, cfg)
eta_all = zeros(N, T, M)
for m in 1:M; eta_all[:,:,m] .= 0.6 .* y; end

ll_hist = zeros(S)
q_buf = zeros(L)
hist = ParamHistory(
    zeros(K+1,L,S), zeros(S), zeros(S),
    zeros(L,S), zeros(S), zeros(S),
    zeros(L,S), zeros(S), zeros(S))

t_gap = @elapsed begin
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
            @printf("  [gap] %3d/%d | ll %8.4f | acc %s | slopes=[%.3f,%.3f,%.3f]\n",
                    iter, S, ll_hist[iter],
                    join([@sprintf("%.2f", a) for a in acc], "/"),
                    par.a_Q[2,:]...)
        end
    end
end

# Ergodic average
S2 = div(S, 2); rng = (S - S2 + 1):S
slopes_avg = vec(mean(hist.a_Q[2, :, rng], dims=2))
intercepts_avg = vec(mean(hist.a_Q[1, :, rng], dims=2))

# Last iteration (for comparison with previous buggy report)
slopes_last = par.a_Q[2, :]

println("\n", "="^70)
println("  RESULTS")
println("="^70)
@printf("  True slopes:       [0.8000, 0.8000, 0.8000]\n")
@printf("  Last-iter slopes:  [%.4f, %.4f, %.4f]\n", slopes_last...)
@printf("  Ergodic avg:       [%.4f, %.4f, %.4f]  <-- proper ABB estimator\n", slopes_avg...)
@printf("  Ergodic intercepts:[%.4f, %.4f, %.4f]  (true [-0.337, 0, 0.337])\n", intercepts_avg...)
@printf("  Time: %.1fs\n", t_gap)

# Percentiles of slope across iterations
println("\nSlope percentiles across last 25 iterations:")
@printf("  %5s  %6s  %6s  %6s  %6s  %6s\n", "tau", "p10", "p25", "p50", "p75", "p90")
for l in 1:L
    v = vec(hist.a_Q[2, l, rng])
    pv = quantile(v, [0.1, 0.25, 0.5, 0.75, 0.9])
    @printf("  %5.2f  %6.4f  %6.4f  %6.4f  %6.4f  %6.4f\n",
            tau[l], pv...)
end

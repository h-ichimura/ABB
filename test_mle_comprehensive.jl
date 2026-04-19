#=
test_mle_comprehensive.jl — Thorough tests before HPC deployment

Test 1: M-step on observed η across multiple seeds
Test 2: Non-crossing preserved even with adversarial starts
Test 3: Full EM likelihood trend (MCEM-noisy but should trend up)
Test 4: Full EM parameter recovery across seeds (N=500, limited budget)
Test 5: Timing: QR vs MLE across sample sizes
=#

include("ABB_three_period.jl")
using Printf, Statistics, Random, Dates

# ================================================================
#  Helpers
# ================================================================

function eval_loglik(par, y, eta, cfg)
    q_buf = zeros(cfg.L)
    ll = 0.0
    for i in 1:cfg.N
        ll += full_loglik(view(y, i, :), view(eta, i, :), par, cfg, q_buf)
    end
    ll / cfg.N
end

function check_non_crossing_full(a_Q, eta_lag_values, K, sigma_y)
    L = size(a_Q, 2)
    H = hermite_basis(eta_lag_values, K, sigma_y)
    Q_mat = H * a_Q
    n_obs = size(H, 1)
    n_cross = 0
    for j in 1:n_obs, l in 1:L-1
        Q_mat[j, l+1] < Q_mat[j, l] && (n_cross += 1)
    end
    n_cross
end

# Common setup
K = 2; L = 3; sigma_y = 1.0; T = 3
tau = [0.25, 0.50, 0.75]
par_true = make_true_params_linear(tau=tau, sigma_y=sigma_y, K=K)

# Open output file
io = open("results_mle_comprehensive.txt", "w")
log_out(args...) = (println(args...); println(io, args...))

log_out("="^70)
log_out("  COMPREHENSIVE MLE TESTS (before HPC)")
log_out("="^70)
log_out("Date: ", now())
log_out("True params: persistence=0.8, sigma_v=0.5, sigma_eps=0.3, K=$K, L=$L")

# ================================================================
#  TEST 1: M-step on observed η, multiple seeds
# ================================================================
log_out("\n", "="^70)
log_out("  TEST 1: M-step on observed η (10 seeds, N=2000)")
log_out("="^70)
log_out("Measures pure M-step quality — no E-step noise")

N_test = 2000
seeds_test1 = 1:10

results_t1 = []
for seed in seeds_test1
    y, eta_true = generate_data_abb(N_test, par_true, tau, sigma_y, K; seed=seed)
    y_clean = copy(eta_true)
    M = 1
    cfg = Config(N_test, T, K, L, tau, sigma_y, 1, 1, M, fill(0.05, T))
    eta_all = zeros(N_test, T, M)
    eta_all[:, :, 1] .= eta_true

    # QR
    par_qr = copy_params(par_true); par_qr.a_Q .= par_true.a_Q .* 0.5
    par_qr.b1_Q = 3.0; par_qr.bL_Q = 3.0
    t_qr = @elapsed m_step_qr!(par_qr, eta_all, y_clean, cfg)
    ll_qr = eval_loglik(par_qr, y_clean, eta_true, cfg)

    # MLE
    par_mle = copy_params(par_true); par_mle.a_Q .= par_true.a_Q .* 0.5
    par_mle.b1_Q = 3.0; par_mle.bL_Q = 3.0
    t_mle = @elapsed m_step_mle!(par_mle, eta_all, y_clean, cfg)
    ll_mle = eval_loglik(par_mle, y_clean, eta_true, cfg)

    # Check non-crossing on the training data
    eta_lag_stk = vec(eta_true[:, 1:T-1])
    nc_qr = check_non_crossing_full(par_qr.a_Q, eta_lag_stk, K, sigma_y)
    nc_mle = check_non_crossing_full(par_mle.a_Q, eta_lag_stk, K, sigma_y)

    push!(results_t1, (seed=seed, ll_qr=ll_qr, ll_mle=ll_mle,
                        slopes_qr=par_qr.a_Q[2,:], slopes_mle=par_mle.a_Q[2,:],
                        t_qr=t_qr, t_mle=t_mle, nc_qr=nc_qr, nc_mle=nc_mle))
end

ll_true = 0.0  # average across seeds
log_out(@sprintf("%6s  %10s  %10s  %10s  %8s  %8s  %s",
                 "seed", "ll_QR", "ll_MLE", "diff", "t_QR", "t_MLE", "cross(QR/MLE)"))
for r in results_t1
    log_out(@sprintf("%6d  %10.4f  %10.4f  %+10.4f  %8.2fs  %8.2fs  %d/%d",
            r.seed, r.ll_qr, r.ll_mle, (r.ll_mle - r.ll_qr),
            r.t_qr, r.t_mle, r.nc_qr, r.nc_mle))
end
mle_beats_qr = sum(r.ll_mle > r.ll_qr for r in results_t1)
log_out(@sprintf("\nMLE beat QR in %d/%d seeds. Avg gain: %.5f",
                 mle_beats_qr, length(results_t1),
                 mean(r.ll_mle - r.ll_qr for r in results_t1)))

avg_slope_qr = mean(mean(r.slopes_qr) for r in results_t1)
avg_slope_mle = mean(mean(r.slopes_mle) for r in results_t1)
log_out(@sprintf("Avg slope (true=0.8000): QR=%.4f, MLE=%.4f",
                 avg_slope_qr, avg_slope_mle))

# ================================================================
#  TEST 2: Non-crossing under adversarial starts
# ================================================================
log_out("\n", "="^70)
log_out("  TEST 2: Non-crossing with adversarial starts")
log_out("="^70)
log_out("Starts: small-slope, large-slope, crossing, zero")

seed = 42
N_test2 = 1000
y, eta_true = generate_data_abb(N_test2, par_true, tau, sigma_y, K; seed=seed)
y_clean = copy(eta_true)
M = 1
cfg = Config(N_test2, T, K, L, tau, sigma_y, 1, 1, M, fill(0.05, T))
eta_all = zeros(N_test2, T, M); eta_all[:, :, 1] .= eta_true

eta_lag_stk = vec(eta_true[:, 1:T-1])

# Define bad starts
bad_starts = Dict{String, Matrix{Float64}}()
bad_starts["small-slope"]  = [-0.3 0.0 0.3;  0.3 0.3 0.3;  0.0 0.0 0.0]
bad_starts["large-slope"]  = [-0.3 0.0 0.3;  1.2 1.2 1.2;  0.0 0.0 0.0]
bad_starts["crossing"]     = [0.3 0.0 -0.3;  0.8 0.8 0.8;  0.0 0.0 0.0]  # reversed intercepts
bad_starts["all-zero"]     = zeros(K+1, L)

log_out(@sprintf("%-15s  %12s  %12s  %s", "start", "ll_final", "cross_start", "cross_end"))
for (name, a0) in bad_starts
    par_mle = copy_params(par_true)
    par_mle.a_Q .= a0
    par_mle.b1_Q = 3.0; par_mle.bL_Q = 3.0
    nc_start = check_non_crossing_full(par_mle.a_Q, eta_lag_stk, K, sigma_y)

    try
        m_step_mle!(par_mle, eta_all, y_clean, cfg)
        ll = eval_loglik(par_mle, y_clean, eta_true, cfg)
        nc_end = check_non_crossing_full(par_mle.a_Q, eta_lag_stk, K, sigma_y)
        log_out(@sprintf("%-15s  %12.4f  %12d  %d", name, ll, nc_start, nc_end))
    catch e
        log_out(@sprintf("%-15s  FAILED: %s", name, e))
    end
end

# ================================================================
#  TEST 3: Full EM likelihood trend at small-medium N
# ================================================================
log_out("\n", "="^70)
log_out("  TEST 3: Full EM likelihood trend (N=500, M=50, S=50, seed=42)")
log_out("="^70)
log_out("MCEM-noisy, but likelihood should trend up overall")

cfg = Config(500, T, K, L, tau, sigma_y, 50, 200, 50, fill(0.05, T))
y, eta_true = generate_data_abb(500, par_true, tau, sigma_y, K; seed=42)

for meth in [:qr, :mle]
    log_out("\n  --- Method: $(uppercase(string(meth))) ---")
    par_avg, _, _, ll_hist, hist = estimate(y, cfg; method=meth, verbose=false)
    dll = diff(ll_hist)
    n_viol = count(dll .< -0.01)

    log_out(@sprintf("  Final ll: %.4f (avg last 25: %.4f)",
                     ll_hist[end], mean(ll_hist[end-24:end])))
    log_out(@sprintf("  Monotonicity violations: %d/%d", n_viol, length(dll)))
    log_out(@sprintf("  Slopes (final avg): [%s]",
            join([@sprintf("%.4f", par_avg.a_Q[2,l]/sigma_y) for l in 1:L], ", ")))

    # Check non-crossing at every iteration
    eta_lag_stk = vec(eta_true[:, 1:T-1])
    worst_cross = 0
    for s in 1:cfg.maxiter
        a_s = hist.a_Q[:, :, s]
        nc = check_non_crossing_full(a_s, eta_lag_stk, K, sigma_y)
        worst_cross = max(worst_cross, nc)
    end
    log_out(@sprintf("  Max crossings across all iterations: %d", worst_cross))
end

# ================================================================
#  TEST 4: Full EM parameter recovery (3 seeds at N=500)
# ================================================================
log_out("\n", "="^70)
log_out("  TEST 4: Full EM parameter recovery (3 seeds at N=500, S=50)")
log_out("="^70)
log_out("Limited budget test — HPC will do 200 seeds at S=200")

cfg = Config(500, T, K, L, tau, sigma_y, 50, 200, 50, fill(0.05, T))
seeds_t4 = [101, 202, 303]

log_out(@sprintf("%6s  %s  %s  %s  %12s  %12s",
        "seed", "slopes(QR)", "slopes(MLE)", "slopes(true)", "t_QR(s)", "t_MLE(s)"))
for seed in seeds_t4
    y, eta_true = generate_data_abb(500, par_true, tau, sigma_y, K; seed=seed)
    t_qr = @elapsed par_qr, _, _, _, _ = estimate(y, cfg; method=:qr, verbose=false)
    t_mle = @elapsed par_mle, _, _, _, _ = estimate(y, cfg; method=:mle, verbose=false)
    log_out(@sprintf("%6d  [%.3f,%.3f,%.3f]  [%.3f,%.3f,%.3f]  [0.800,0.800,0.800]  %10.1f  %10.1f",
            seed,
            par_qr.a_Q[2,1]/sigma_y, par_qr.a_Q[2,2]/sigma_y, par_qr.a_Q[2,3]/sigma_y,
            par_mle.a_Q[2,1]/sigma_y, par_mle.a_Q[2,2]/sigma_y, par_mle.a_Q[2,3]/sigma_y,
            t_qr, t_mle))
end

# ================================================================
#  TEST 5: Timing QR vs MLE at multiple N (single M-step only)
# ================================================================
log_out("\n", "="^70)
log_out("  TEST 5: Timing QR vs MLE at varying N (observed η, single M-step)")
log_out("="^70)

log_out(@sprintf("%6s  %10s  %10s  %6s", "N", "t_QR(s)", "t_MLE(s)", "ratio"))
for N_t5 in [200, 500, 1000, 2000, 5000]
    y, eta_true = generate_data_abb(N_t5, par_true, tau, sigma_y, K; seed=42)
    y_clean = copy(eta_true)
    M = 1
    cfg = Config(N_t5, T, K, L, tau, sigma_y, 1, 1, M, fill(0.05, T))
    eta_all = zeros(N_t5, T, M); eta_all[:, :, 1] .= eta_true

    par_qr = copy_params(par_true); par_qr.a_Q .= par_true.a_Q .* 0.5
    par_qr.b1_Q = 3.0; par_qr.bL_Q = 3.0
    t_qr = @elapsed m_step_qr!(par_qr, eta_all, y_clean, cfg)

    par_mle = copy_params(par_true); par_mle.a_Q .= par_true.a_Q .* 0.5
    par_mle.b1_Q = 3.0; par_mle.bL_Q = 3.0
    t_mle = @elapsed m_step_mle!(par_mle, eta_all, y_clean, cfg)

    log_out(@sprintf("%6d  %10.3f  %10.3f  %6.2f", N_t5, t_qr, t_mle, (t_mle/t_qr)))
end

log_out("\n", "="^70)
log_out("  END")
log_out("="^70)

close(io)
println("\nResults saved to results_mle_comprehensive.txt")

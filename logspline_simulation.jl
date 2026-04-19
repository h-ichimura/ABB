#=
logspline_simulation.jl — Full simulation study for logspline MLE vs QR.

Step 1: Generate non-Gaussian AR(1) data (κ=0.1)
Step 2: Estimate by QR (warm start), then logspline MLE
Step 3: Use estimated logspline as "true" DGP
Step 4: Monte Carlo: generate from logspline truth, estimate by QR and LS-MLE
Step 5: Coverage and comparison
=#

include("ABB_three_period.jl")
include("logspline.jl")
include("ABB_logspline.jl")  # for estimate_ls, ls_e_step!, m_step_logspline!

using Serialization, Plots; gr()

# ================================================================
#  STEP 1: Generate non-Gaussian AR(1) data
# ================================================================

function transform_ng(x::Float64, kappa::Float64)
    x + kappa * x^3 / 3.0
end

function generate_ngar1(N::Int, T::Int; rho=0.8, sigma_v=0.5,
                        sigma_eps=0.3, kappa=0.1, seed=42)
    rng = MersenneTwister(seed)
    sigma_eta = sigma_v / sqrt(1 - rho^2)
    eta = zeros(N, T); y = zeros(N, T)
    for i in 1:N
        eta[i, 1] = transform_ng(sigma_eta * randn(rng), kappa)
    end
    for t in 2:T, i in 1:N
        eta_star = rho * eta[i, t-1] + sigma_v * randn(rng)
        eta[i, t] = transform_ng(eta_star, kappa)
    end
    for t in 1:T, i in 1:N
        y[i, t] = eta[i, t] + sigma_eps * randn(rng)
    end
    y, eta
end

# ================================================================
#  STEP 2: Estimate on initial data → get "true" logspline params
# ================================================================

println("="^70)
println("  LOGSPLINE SIMULATION STUDY")
println("="^70)

N_init = 2000; T = 3; K = 2; L = 3; sigma_y = 1.0
tau = collect(range(1/(L+1), stop=L/(L+1), length=L))

println("\nStep 1: Generating non-Gaussian AR(1) data (N=$N_init, κ=0.1)...")
y_init, eta_init = generate_ngar1(N_init, T; seed=42)
@printf("  var(y): %.3f, corr(y1,y2): %.3f\n", var(y_init[:,1]), cor(y_init[:,1], y_init[:,2]))

# QR warm start
println("\nStep 2a: QR estimation (warm start)...")
cfg_init = Config(N_init, T, K, L, tau, sigma_y, 100, 200, 50, fill(0.05, T))
par_qr, _, eta_qr, ll_qr, _ = estimate(y_init, cfg_init; method=:qr)
println("  QR persistence: ", round.(par_qr.a_Q[2,:] ./ sigma_y, digits=4))

# Logspline MLE starting from QR
# Use knots at sample quantiles of the pooled η draws
eta_pool = vcat([vec(eta_qr[:, t, :]) for t in 1:T]...)
knots_sp = quantile(eta_pool, [0.1, 0.3, 0.5, 0.7, 0.9])
println("  Logspline knots (data quantiles): ", round.(knots_sp, digits=3))

println("\nStep 2b: Logspline MLE refinement...")
a_trans_est, par_ls_est, eta_ls, ll_ls = estimate_ls(y_init, cfg_init; knots_sp=knots_sp)

println("\n  Logspline coefficients (a_trans):")
display(round.(a_trans_est, digits=4))
println()

# ================================================================
#  STEP 3: Use estimated logspline as "true" DGP
# ================================================================

println("\nStep 3: Setting estimated logspline as truth...")
ls_true = LogsplineTransition(a_trans_est, knots_sp, sigma_y, K)

# Also store the "true" marginals from the estimation
par_true_ls = copy_params(par_ls_est)

# Verify: generate data from logspline and check moments
println("  Generating verification sample from logspline DGP...")
rng_verify = MersenneTwister(999)
N_verify = 5000
eta_verify = zeros(N_verify, T); y_verify = zeros(N_verify, T)

# Initial: draw from estimated initial distribution
for i in 1:N_verify
    # Use inverse CDF of piecewise-uniform initial
    u = rand(rng_verify)
    if u < tau[1]
        eta_verify[i, 1] = par_true_ls.a_init[1] + log(u / tau[1]) / par_true_ls.b1_init
    elseif u > tau[L]
        eta_verify[i, 1] = par_true_ls.a_init[L] - log((1-u) / (1-tau[L])) / par_true_ls.bL_init
    else
        for l in 1:L-1
            if u <= tau[l+1]
                eta_verify[i, 1] = par_true_ls.a_init[l] +
                    (u - tau[l]) / (tau[l+1] - tau[l]) * (par_true_ls.a_init[l+1] - par_true_ls.a_init[l])
                break
            end
        end
    end
end

# Transition: draw from logspline
for t in 2:T, i in 1:N_verify
    eta_verify[i, t] = logspline_draw(rng_verify, eta_verify[i, t-1], ls_true)
end

# Epsilon: draw from estimated eps distribution (piecewise-uniform)
for t in 1:T, i in 1:N_verify
    u = rand(rng_verify)
    if u < tau[1]
        eps = par_true_ls.a_eps[1] + log(u / tau[1]) / par_true_ls.b1_eps
    elseif u > tau[L]
        eps = par_true_ls.a_eps[L] - log((1-u) / (1-tau[L])) / par_true_ls.bL_eps
    else
        eps = 0.0
        for l in 1:L-1
            if u <= tau[l+1]
                eps = par_true_ls.a_eps[l] +
                    (u - tau[l]) / (tau[l+1] - tau[l]) * (par_true_ls.a_eps[l+1] - par_true_ls.a_eps[l])
                break
            end
        end
    end
    y_verify[i, t] = eta_verify[i, t] + eps
end

println("  Verification data moments:")
for t in 1:T
    @printf("    var(y_%d) = %.4f\n", t, var(y_verify[:, t]))
end
for t in 2:T
    @printf("    corr(y_%d, y_%d) = %.4f\n", t-1, t, cor(y_verify[:, t-1], y_verify[:, t]))
end

if any(abs.(eta_verify) .> 50)
    println("  WARNING: η exploded! max |η| = $(maximum(abs.(eta_verify)))")
    println("  Logspline DGP may be unstable. Aborting.")
    exit(1)
end

# ================================================================
#  STEP 4: Monte Carlo simulation
# ================================================================

println("\n" * "="^70)
println("  MONTE CARLO: QR vs Logspline MLE")
println("="^70)

N_sim = 500  # sample size per replication
R = 20       # replications (use more on HPC)
S_sim = 100  # EM iterations per replication
M_sim = 50   # draws per E-step

cfg_sim = Config(N_sim, T, K, L, tau, sigma_y, S_sim, 200, M_sim, fill(0.05, T))

# Storage for point estimates (ergodic averages)
# QR: persistence = a_Q[2, l] / sigma_y for l=1:L
# LS: the full a_trans matrix
qr_slopes = zeros(R, L)
ls_coeffs = zeros(R, size(a_trans_est)...)
qr_times = zeros(R)
ls_times = zeros(R)

function generate_from_logspline_dgp(N, T, ls_true, par_true, tau, seed)
    rng = MersenneTwister(seed)
    L = length(tau)
    eta = zeros(N, T); y = zeros(N, T)

    # Initial: piecewise-uniform
    for i in 1:N
        u = rand(rng)
        if u < tau[1]
            eta[i, 1] = par_true.a_init[1] + log(u / tau[1]) / par_true.b1_init
        elseif u > tau[L]
            eta[i, 1] = par_true.a_init[L] - log((1-u) / (1-tau[L])) / par_true.bL_init
        else
            for l in 1:L-1
                if u <= tau[l+1]
                    eta[i, 1] = par_true.a_init[l] +
                        (u - tau[l]) / (tau[l+1] - tau[l]) * (par_true.a_init[l+1] - par_true.a_init[l])
                    break
                end
            end
        end
    end

    # Transition: logspline
    for t in 2:T, i in 1:N
        eta[i, t] = logspline_draw(rng, eta[i, t-1], ls_true)
    end

    # Epsilon: piecewise-uniform
    for t in 1:T, i in 1:N
        u = rand(rng)
        if u < tau[1]
            eps = par_true.a_eps[1] + log(u / tau[1]) / par_true.b1_eps
        elseif u > tau[L]
            eps = par_true.a_eps[L] - log((1-u) / (1-tau[L])) / par_true.bL_eps
        else
            eps = 0.0
            for l in 1:L-1
                if u <= tau[l+1]
                    eps = par_true.a_eps[l] +
                        (u - tau[l]) / (tau[l+1] - tau[l]) * (par_true.a_eps[l+1] - par_true.a_eps[l])
                    break
                end
            end
        end
        y[i, t] = eta[i, t] + eps
    end

    y, eta
end

for r in 1:R
    seed_r = 1000 + r
    @printf("\n--- Replication %d/%d (seed=%d) ---\n", r, R, seed_r)

    y_r, _ = generate_from_logspline_dgp(N_sim, T, ls_true, par_true_ls, tau, seed_r)

    # QR
    t_qr = @elapsed par_qr_r, _, _, _, hist_qr_r = estimate(y_r, cfg_sim; method=:qr, verbose=false)
    S2 = div(S_sim, 2); rng_avg = (S_sim - S2 + 1):S_sim
    for l in 1:L
        qr_slopes[r, l] = mean(hist_qr_r.a_Q[2, l, rng_avg]) / sigma_y
    end
    qr_times[r] = t_qr

    # Logspline MLE
    t_ls = @elapsed a_trans_r, par_ls_r, _, ll_ls_r =
        estimate_ls(y_r, cfg_sim; knots_sp=knots_sp, verbose=false)
    ls_coeffs[r, :, :] = a_trans_r
    ls_times[r] = t_ls

    @printf("  QR slopes: [%s]  (%.1fs)\n",
            join([@sprintf("%.4f", qr_slopes[r,l]) for l in 1:L], ", "), t_qr)
    @printf("  LS a[2,:]: [%s]  (%.1fs)\n",
            join([@sprintf("%.4f", a_trans_r[2,j]) for j in 1:K+1], ", "), t_ls)
end

# ================================================================
#  STEP 5: Results
# ================================================================

# True QR slopes: compute quantile slopes from logspline truth
# At η_{t-1}=0, the conditional quantiles can be computed numerically
true_qr_slopes = par_qr.a_Q[2, :] ./ sigma_y  # use initial QR estimate as reference
true_ls_a2 = a_trans_est[2, :]

open("results_logspline_simulation.txt", "w") do io
    println(io, "Logspline Simulation Results")
    println(io, "="^70)
    @printf(io, "N=%d, R=%d, S=%d, M=%d\n", N_sim, R, S_sim, M_sim)
    @printf(io, "Logspline knots: [%s]\n", join([@sprintf("%.3f", k) for k in knots_sp], ", "))

    println(io, "\nQR slope estimates (persistence):")
    @printf(io, "  Reference (from initial QR): [%s]\n",
            join([@sprintf("%.4f", s) for s in true_qr_slopes], ", "))
    for l in 1:L
        m = mean(qr_slopes[:, l])
        s = std(qr_slopes[:, l])
        @printf(io, "  τ=%.2f: mean=%.4f, std=%.4f, bias=%+.4f\n",
                tau[l], m, s, m - true_qr_slopes[l])
    end

    println(io, "\nLogspline β₁ coefficients (a_trans[2,:]):")
    @printf(io, "  True: [%s]\n", join([@sprintf("%.4f", c) for c in true_ls_a2], ", "))
    for j in 1:K+1
        vals = ls_coeffs[:, 2, j]
        m = mean(vals); s = std(vals)
        @printf(io, "  a[2,%d]: mean=%.4f, std=%.4f, bias=%+.4f\n",
                j, m, s, m - true_ls_a2[j])
    end

    @printf(io, "\nTiming: QR=%.1f±%.1fs, LS=%.1f±%.1fs\n",
            mean(qr_times), std(qr_times), mean(ls_times), std(ls_times))
end

println("\nResults saved to results_logspline_simulation.txt")

# Save for later analysis
serialize("logspline_simulation.jls",
          (ls_true=ls_true, par_true=par_true_ls, knots_sp=knots_sp,
           qr_slopes=qr_slopes, ls_coeffs=ls_coeffs,
           true_qr_slopes=true_qr_slopes, true_ls_a2=true_ls_a2,
           qr_times=qr_times, ls_times=ls_times))
println("Saved logspline_simulation.jls")

println("\nDone!")

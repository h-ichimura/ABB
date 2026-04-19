#=
test_fixed_points.jl — Is θ₀ the unique fixed point of ABB's QR iteration?

Run the full ABB iteration (E-step + QR M-step) from multiple starting points
on the SAME dataset. Track parameter trajectories across iterations.
If all trajectories converge to the same point → unique fixed point.
If some converge elsewhere → multiple fixed points.
=#

include("ABB_three_period.jl")

# ================================================================
#  MODIFIED ESTIMATE: accepts initial parameters, returns full trajectory
# ================================================================
function estimate_from(y::Matrix{Float64}, cfg::Config, par_start::Params;
                       verbose::Bool=true, label::String="")
    N,T,M = cfg.N, cfg.T, cfg.M
    S = cfg.maxiter; K,L = cfg.K, cfg.L

    par = copy_params(par_start)

    # Initialize eta_all at 0.6*y
    eta_all = zeros(N, T, M)
    for m in 1:M; eta_all[:,:,m] .= 0.6 .* y; end

    ll_hist = zeros(S); q_buf = zeros(L)

    # Store full trajectory
    slope_hist = zeros(L, S)       # a_Q[2,l]/sigma_y
    intercept_hist = zeros(L, S)   # a_Q[1,l]
    eps_q_hist = zeros(L, S)       # a_eps[l]
    b1Q_hist = zeros(S)
    bLQ_hist = zeros(S)
    b1eps_hist = zeros(S)
    bLeps_hist = zeros(S)

    for iter in 1:S
        acc = e_step!(eta_all, y, par, cfg)
        m_step_qr!(par, eta_all, y, cfg)

        # Record
        for l in 1:L
            slope_hist[l, iter] = par.a_Q[2, l] / cfg.sigma_y
            intercept_hist[l, iter] = par.a_Q[1, l]
            eps_q_hist[l, iter] = par.a_eps[l]
        end
        b1Q_hist[iter] = par.b1_Q
        bLQ_hist[iter] = par.bL_Q
        b1eps_hist[iter] = par.b1_eps
        bLeps_hist[iter] = par.bL_eps

        # Log-likelihood
        ll = 0.0
        for m in 1:M, i in 1:N
            ll += full_loglik(view(y,i,:), view(eta_all,i,:,m), par, cfg, q_buf)
        end
        ll_hist[iter] = ll / (N*M)

        if verbose && (iter % 50 == 0 || iter <= 3 || iter == S)
            @printf("  [%s] %3d/%d | ll %8.4f | slopes [%.3f,%.3f,%.3f] | acc %.2f/%.2f/%.2f\n",
                    label, iter, S, ll_hist[iter],
                    slope_hist[1,iter], slope_hist[2,iter], slope_hist[3,iter],
                    acc...)
        end
    end

    (par=par, ll=ll_hist,
     slopes=slope_hist, intercepts=intercept_hist,
     eps_q=eps_q_hist, b1Q=b1Q_hist, bLQ=bLQ_hist,
     b1eps=b1eps_hist, bLeps=bLeps_hist)
end

# ================================================================
#  SETUP
# ================================================================
K = 2; L = 3; sigma_y = 1.0; T = 3
tau = [0.25, 0.50, 0.75]
N = 5000; M = 200; n_draws = 500; maxiter = 200
vp = fill(0.05, T)
cfg = Config(N, T, K, L, tau, sigma_y, maxiter, n_draws, M, vp)

par_true = make_true_params_linear(tau=tau, sigma_y=sigma_y, K=K)
y, eta_true = generate_data_abb(N, par_true, tau, sigma_y, K; seed=42)

io = open("results_fixed_points.txt", "w")
function pb(args...)  # print both
    print(stdout, args...); print(io, args...)
end
function plb(args...)  # println both
    println(stdout, args...); println(io, args...)
end

plb("="^80)
plb("  FIXED POINT TEST: Multiple Starting Points")
plb("  N=$N, M=$M, n_draws=$n_draws, S=$maxiter")
plb("="^80)
pb(@sprintf("  True slopes:     [%.4f, %.4f, %.4f]\n", par_true.a_Q[2,:]...))
pb(@sprintf("  True intercepts: [%.4f, %.4f, %.4f]\n", par_true.a_Q[1,:]...))
pb(@sprintf("  True ε quantiles:[%.4f, %.4f, %.4f]\n", par_true.a_eps...))
pb(@sprintf("  True b1_Q=%.4f  bL_Q=%.4f  b1_eps=%.4f  bL_eps=%.4f\n",
            par_true.b1_Q, par_true.bL_Q, par_true.b1_eps, par_true.bL_eps))

# ================================================================
#  DEFINE STARTING POINTS
# ================================================================
starts = Dict{String, Params}()

# 1. True parameters
starts["TRUE"] = copy_params(par_true)

# 2. Small σ_ε (0.5× true → larger tail rates, tighter ε quantiles)
p = copy_params(par_true)
p.a_eps .*= 0.5; p.b1_eps *= 2.0; p.bL_eps *= 2.0
starts["EPS_SMALL"] = p

# 3. Large σ_ε (2× true)
p = copy_params(par_true)
p.a_eps .*= 2.0; p.b1_eps *= 0.5; p.bL_eps *= 0.5
starts["EPS_LARGE"] = p

# 4. Very large σ_ε (3× true)
p = copy_params(par_true)
p.a_eps .*= 3.0; p.b1_eps /= 3.0; p.bL_eps /= 3.0
starts["EPS_VLARGE"] = p

# 5. Low persistence (ρ=0.4)
p = copy_params(par_true)
p.a_Q[2,:] .= 0.4 * sigma_y
starts["RHO_LOW"] = p

# 6. High persistence (ρ=0.95)
p = copy_params(par_true)
p.a_Q[2,:] .= 0.95 * sigma_y
starts["RHO_HIGH"] = p

# 7. Wrong ρ + wrong σ_ε
p = copy_params(par_true)
p.a_Q[2,:] .= 0.4 * sigma_y
p.a_eps .*= 2.0; p.b1_eps *= 0.5; p.bL_eps *= 0.5
starts["RHO_LOW_EPS_LARGE"] = p

# ================================================================
#  RUN
# ================================================================
results = Dict{String, Any}()
start_names = ["TRUE", "EPS_SMALL", "EPS_LARGE", "EPS_VLARGE",
               "RHO_LOW", "RHO_HIGH", "RHO_LOW_EPS_LARGE"]

for name in start_names
    plb("\n", "-"^80)
    plb("  Starting point: $name")
    plb("-"^80)
    p = starts[name]
    pb(@sprintf("  Start slopes:     [%.4f, %.4f, %.4f]\n", p.a_Q[2,:]...))
    pb(@sprintf("  Start intercepts: [%.4f, %.4f, %.4f]\n", p.a_Q[1,:]...))
    pb(@sprintf("  Start ε quantiles:[%.4f, %.4f, %.4f]\n", p.a_eps...))
    pb(@sprintf("  Start b1_eps=%.4f  bL_eps=%.4f\n", p.b1_eps, p.bL_eps))

    # Capture output
    old_stdout = stdout
    rd, wr = redirect_stdout()
    res = estimate_from(y, cfg, p; verbose=true, label=name)
    redirect_stdout(old_stdout)
    close(wr)
    output = read(rd, String)
    print(output); print(io, output)

    results[name] = res
    flush(io)
end

# ================================================================
#  SUMMARY: Final parameter values from each starting point
# ================================================================
plb("\n\n", "="^80)
plb("  SUMMARY: Final parameters (averaged over last 50 iterations)")
plb("="^80)

S2 = 50  # average last 50
rng = (maxiter - S2 + 1):maxiter

pb(@sprintf("\n  %-22s", ""))
for l in 1:L
    pb(@sprintf("  slope(τ=%.2f)", tau[l]))
end
for l in 1:L
    pb(@sprintf("  intcp(τ=%.2f)", tau[l]))
end
for l in 1:L
    pb(@sprintf("  ε_q(τ=%.2f)", tau[l]))
end
pb("     b1_Q    bL_Q  b1_eps  bL_eps")
plb("")

# True values
pb(@sprintf("  %-22s", "TRUE VALUES"))
for l in 1:L; pb(@sprintf("  %12.4f", par_true.a_Q[2,l]/sigma_y)); end
for l in 1:L; pb(@sprintf("  %12.4f", par_true.a_Q[1,l])); end
for l in 1:L; pb(@sprintf("  %12.4f", par_true.a_eps[l])); end
pb(@sprintf("  %7.4f %7.4f %7.4f %7.4f", par_true.b1_Q, par_true.bL_Q,
            par_true.b1_eps, par_true.bL_eps))
plb("")
plb("  " * "-"^200)

for name in start_names
    res = results[name]
    pb(@sprintf("  %-22s", name))
    for l in 1:L; pb(@sprintf("  %12.4f", mean(res.slopes[l, rng]))); end
    for l in 1:L; pb(@sprintf("  %12.4f", mean(res.intercepts[l, rng]))); end
    for l in 1:L; pb(@sprintf("  %12.4f", mean(res.eps_q[l, rng]))); end
    pb(@sprintf("  %7.4f %7.4f %7.4f %7.4f",
                mean(res.b1Q[rng]), mean(res.bLQ[rng]),
                mean(res.b1eps[rng]), mean(res.bLeps[rng])))
    plb("")
end

# ================================================================
#  CONVERGENCE: Max difference between starting points (last 50 avg)
# ================================================================
plb("\n\n  CONVERGENCE CHECK: max |diff| between any two starting points (last 50 avg)")
plb("  " * "-"^80)

# Collect all final slope vectors
final_slopes = Dict{String, Vector{Float64}}()
final_intcpts = Dict{String, Vector{Float64}}()
final_epsq = Dict{String, Vector{Float64}}()
for name in start_names
    res = results[name]
    final_slopes[name] = [mean(res.slopes[l, rng]) for l in 1:L]
    final_intcpts[name] = [mean(res.intercepts[l, rng]) for l in 1:L]
    final_epsq[name] = [mean(res.eps_q[l, rng]) for l in 1:L]
end

max_slope_diff = 0.0
max_intcpt_diff = 0.0
max_epsq_diff = 0.0
for i in 1:length(start_names), j in (i+1):length(start_names)
    n1, n2 = start_names[i], start_names[j]
    sd = maximum(abs.(final_slopes[n1] .- final_slopes[n2]))
    id = maximum(abs.(final_intcpts[n1] .- final_intcpts[n2]))
    ed = maximum(abs.(final_epsq[n1] .- final_epsq[n2]))
    global max_slope_diff = max(max_slope_diff, sd)
    global max_intcpt_diff = max(max_intcpt_diff, id)
    global max_epsq_diff = max(max_epsq_diff, ed)
end

pb(@sprintf("  Max slope difference:     %.4f\n", max_slope_diff))
pb(@sprintf("  Max intercept difference: %.4f\n", max_intcpt_diff))
pb(@sprintf("  Max ε quantile difference:%.4f\n", max_epsq_diff))

if max_slope_diff < 0.05 && max_intcpt_diff < 0.05 && max_epsq_diff < 0.05
    plb("  → All starting points converged to approximately the SAME fixed point")
else
    plb("  → DIFFERENT fixed points detected from different starting points")
end

plb("\n" * "="^80)
plb("  END OF FIXED POINT TEST")
plb("="^80)

close(io)
println("\nResults saved to results_fixed_points.txt")

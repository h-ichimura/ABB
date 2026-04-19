#=
test_profiled.jl — Profile out non-knot parameters using closed-form MLE

For each grid value of the knot parameter being varied:
  1. Set the knot parameter to the grid value
  2. Compute closed-form MLE of tail rates b1_Q, bL_Q given knots
  3. Set marginal params (a_eps, a_init, their tails) at their MLE
     (sample quantiles from draws — independent of transition knots)
  4. Evaluate the concentrated CDLL

This is the profiled (concentrated) log-likelihood.
=#

include("ABB_three_period.jl")

K=2; L=3; sigma_y=1.0; T=3
tau = [0.25, 0.50, 0.75]
N=2000; M=50; n_draws=200
vp = fill(0.05, T)
cfg = Config(N, T, K, L, tau, sigma_y, 1, n_draws, M, vp)

par_true = make_true_params_linear(tau=tau, sigma_y=sigma_y, K=K)
y, eta_true = generate_data_abb(N, par_true, tau, sigma_y, K; seed=42)

println("Running E-step at true parameters...")
eta_all = zeros(N, T, M)
for m in 1:M; eta_all[:,:,m] .= 0.6 .* y; end
acc = e_step!(eta_all, y, par_true, cfg)
@printf("Acceptance: %.2f/%.2f/%.2f\n", acc...)

eta_t, eta_lag = stack_transition(eta_all, cfg)
H = hermite_basis(eta_lag, K, sigma_y)
n_obs = length(eta_t)
Q_mat = zeros(n_obs, L)

# Compute MLE of marginal params (independent of transition knots)
eta1_all = stack_initial(eta_all, cfg)
eps_all = stack_eps(eta_all, y, cfg)

a_init_mle = [quantile(eta1_all, tau[l]) for l in 1:L]
b1_init_mle, bL_init_mle = update_tails(eta1_all, a_init_mle[1], a_init_mle[L])

a_eps_mle = [quantile(eps_all, tau[l]) for l in 1:L]
a_eps_mle .-= mean(a_eps_mle)
b1_eps_mle, bL_eps_mle = update_tails(eps_all, a_eps_mle[1], a_eps_mle[L])

a_true = copy(par_true.a_Q)

"""
Profiled CDLL: given transition knot matrix a_Q, compute closed-form
MLE of b1_Q, bL_Q, set marginals at their MLE, return avg CDLL.
"""
function profiled_cdll(a_Q::Matrix{Float64})
    mul!(Q_mat, H, a_Q)

    # Closed-form MLE of tail rates given knots
    r1 = eta_t .- view(Q_mat,:,1)
    rL = eta_t .- view(Q_mat,:,L)
    ml = r1 .<= 0; mh = rL .>= 0
    s1 = sum(r1[ml]); sL = sum(rL[mh])
    b1v = s1 < -1e-10 ? -count(ml)/s1 : 2.0
    bLv = sL >  1e-10 ?  count(mh)/sL : 2.0

    par = Params(a_Q, b1v, bLv,
                 a_init_mle, b1_init_mle, bL_init_mle,
                 a_eps_mle, b1_eps_mle, bL_eps_mle)

    q_buf = zeros(L); ll = 0.0
    for m in 1:M, i in 1:N
        ll += full_loglik(view(y,i,:), view(eta_all,i,:,m), par, cfg, q_buf)
    end
    ll / (N*M), b1v, bLv
end

ngrid = 81
slope_grid = collect(range(0.4, 1.2, length=ngrid))

# ================================================================
println("\n" * "="^70)
println("PROFILED CDLL: slope profiles (non-knot params at closed-form MLE)")
println("="^70)

println("\n--- Slope(τ=0.50) profile, other knots at truth ---")
vals = zeros(ngrid); b1s = zeros(ngrid); bLs = zeros(ngrid)
for (ig, s) in enumerate(slope_grid)
    a = copy(a_true); a[2,2] = s
    vals[ig], b1s[ig], bLs[ig] = profiled_cdll(a)
end
best = slope_grid[argmax(vals)]
@printf("  Profiled CDLL maximizer: %.4f (true: 0.80)\n", best)
@printf("  b1_Q at maximizer: %.4f, bL_Q at maximizer: %.4f\n",
        b1s[argmax(vals)], bLs[argmax(vals)])

# Print profile
println("\n  slope    profiled_CDLL    b1_Q    bL_Q")
for s in [0.50, 0.60, 0.70, 0.75, 0.78, 0.80, 0.82, 0.85, 0.90, 1.00]
    ig = argmin(abs.(slope_grid .- s))
    @printf("  %.2f    %10.6f    %.4f  %.4f\n", slope_grid[ig], vals[ig], b1s[ig], bLs[ig])
end

# ================================================================
println("\n--- All three slopes profiled separately ---")
for l in 1:L
    local vals_l = zeros(ngrid)
    for (ig, s) in enumerate(slope_grid)
        a = copy(a_true); a[2,l] = s
        vals_l[ig], _, _ = profiled_cdll(a)
    end
    best_l = slope_grid[argmax(vals_l)]
    @printf("  slope(τ=%.2f): profiled maximizer = %.4f (true: 0.80)\n",
            tau[l], best_l)
end

# ================================================================
println("\n--- Intercept profiles ---")
intcpt_grid = collect(range(-0.8, 0.8, length=ngrid))
for l in 1:L
    local vals_l = zeros(ngrid)
    for (ig, a0) in enumerate(intcpt_grid)
        a = copy(a_true); a[1,l] = a0
        vals_l[ig], _, _ = profiled_cdll(a)
    end
    best_l = intcpt_grid[argmax(vals_l)]
    @printf("  intcpt(τ=%.2f): profiled maximizer = %.4f (true: %.4f)\n",
            tau[l], best_l, a_true[1,l])
end

# ================================================================
println("\n" * "="^70)
println("PROFILED CDLL: slope(τ=0.50) when OTHER knots are wrong")
println("="^70)

knot_scenarios = [
    ("all knots at truth",                      a_true),
    ("intcpt(τ=0.25) = 0 (true=-0.337)",       begin a=copy(a_true); a[1,1]=0.0; a end),
    ("intcpt(τ=0.75) = 0 (true=0.337)",        begin a=copy(a_true); a[1,3]=0.0; a end),
    ("slope(τ=0.25) = 0.5 (true=0.8)",         begin a=copy(a_true); a[2,1]=0.5; a end),
    ("slope(τ=0.75) = 0.5 (true=0.8)",         begin a=copy(a_true); a[2,3]=0.5; a end),
    ("all slopes = 0.5",                        begin a=copy(a_true); a[2,:].=0.5; a end),
    ("all intercepts = 0",                      begin a=copy(a_true); a[1,:].=0.0; a end),
]

for (label, a_base) in knot_scenarios
    local vals_s = zeros(ngrid)
    for (ig, s) in enumerate(slope_grid)
        a = copy(a_base); a[2,2] = s
        vals_s[ig], _, _ = profiled_cdll(a)
    end
    best_s = slope_grid[argmax(vals_s)]
    @printf("  %-45s  maximizer: %.4f\n", label, best_s)
end

# ================================================================
println("\n" * "="^70)
println("COMPARISON: Profiled CDLL vs Fixed-at-truth CDLL vs QR")
println("="^70)
println("All vary slope(τ=0.50), other knots at truth")

# Profiled CDLL (already computed above)
# Fixed CDLL
function fixed_cdll(a_Q::Matrix{Float64})
    par = Params(a_Q, par_true.b1_Q, par_true.bL_Q,
                 a_init_mle, b1_init_mle, bL_init_mle,
                 a_eps_mle, b1_eps_mle, bL_eps_mle)
    q_buf = zeros(L); ll = 0.0
    for m in 1:M, i in 1:N
        ll += full_loglik(view(y,i,:), view(eta_all,i,:,m), par, cfg, q_buf)
    end
    ll / (N*M)
end

# QR check loss
function qr_loss(slope_val, l)
    a = copy(a_true); a[2,l] = slope_val
    r = eta_t .- H * a[:,l]
    mean(r .* (tau[l] .- (r .< 0)))
end

println("\n  slope   profiled_CDLL  fixed_CDLL   QR_loss(τ=0.5)")
for s in [0.50, 0.60, 0.70, 0.75, 0.78, 0.80, 0.82, 0.85, 0.90, 1.00]
    a = copy(a_true); a[2,2] = s
    pc, _, _ = profiled_cdll(a)
    fc = fixed_cdll(a)
    ql = qr_loss(s, 2)
    @printf("  %.2f    %10.6f  %10.6f  %10.6f\n", s, pc, fc, ql)
end

best_prof = slope_grid[argmax([let a=copy(a_true); a[2,2]=s; profiled_cdll(a)[1]; end for s in slope_grid])]
best_fix  = slope_grid[argmax([let a=copy(a_true); a[2,2]=s; fixed_cdll(a); end for s in slope_grid])]
best_qr   = slope_grid[argmin([qr_loss(s, 2) for s in slope_grid])]
@printf("\n  Maximizers: profiled=%.4f  fixed=%.4f  QR=%.4f  (true=0.80)\n",
        best_prof, best_fix, best_qr)

println("\n" * "="^70)
println("DONE")
println("="^70)

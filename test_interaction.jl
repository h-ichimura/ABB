#=
test_interaction.jl — How do wrong non-knot parameters affect knot estimation?

Q1: QR check-loss profile for slope when tail rates are wrong
Q2: QR check-loss profile for slope when eps quantiles are wrong
Q3: CDLL profile for slope when tail rates are wrong
Q4: Tail rate profile when knots are wrong
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

a_true = copy(par_true.a_Q)
ngrid = 81

# Helper: avg CDLL
function avg_cdll(par)
    q_buf = zeros(L); ll = 0.0
    for m in 1:M, i in 1:N
        ll += full_loglik(view(y,i,:), view(eta_all,i,:,m), par, cfg, q_buf)
    end
    ll / (N*M)
end

# ================================================================
println("\n" * "="^70)
println("Q1: QR check-loss for slope(τ=0.50) under different tail rates")
println("="^70)
println("True b1_Q = $(par_true.b1_Q), bL_Q = $(par_true.bL_Q)")
println()
println("NOTE: QR check-loss does NOT depend on tail rates at all.")
println("The check function ρ_τ(Y_j - h_j'a) only involves knot params a.")
println("Tail rates only enter the density, not the check function.")
println()

# Verify: QR loss for slope at τ=0.50 is the same regardless of tail rates
slope_grid = collect(range(0.4, 1.2, length=ngrid))
tau_l = tau[2]  # τ = 0.50

qr_losses = zeros(ngrid)
for (ig, s) in enumerate(slope_grid)
    a_test = copy(a_true)
    a_test[2, 2] = s
    r = eta_t .- H * a_test[:, 2]
    qr_losses[ig] = mean(r .* (tau_l .- (r .< 0)))
end
best_slope_qr = slope_grid[argmin(qr_losses)]
@printf("QR minimizer for slope(τ=0.50): %.4f (true: 0.80)\n", best_slope_qr)
println("This is the SAME regardless of tail rates, because QR doesn't use tails.")

# ================================================================
println("\n" * "="^70)
println("Q2: CDLL profile for slope(τ=0.50) under WRONG tail rates")
println("="^70)

cdll_scenarios = [
    ("b1_Q=2, bL_Q=2 (truth)", 2.0, 2.0),
    ("b1_Q=1, bL_Q=1 (too low)", 1.0, 1.0),
    ("b1_Q=5, bL_Q=5 (too high)", 5.0, 5.0),
    ("b1_Q=0.5, bL_Q=0.5 (very low)", 0.5, 0.5),
    ("b1_Q=10, bL_Q=10 (very high)", 10.0, 10.0),
]

for (label, b1v, bLv) in cdll_scenarios
    cdll_vals = zeros(ngrid)
    for (ig, s) in enumerate(slope_grid)
        p = copy_params(par_true)
        p.a_Q[2, 2] = s
        p.b1_Q = b1v; p.bL_Q = bLv
        cdll_vals[ig] = avg_cdll(p)
    end
    best_slope = slope_grid[argmax(cdll_vals)]
    @printf("  %-35s  CDLL maximizer: %.4f\n", label, best_slope)
end

# ================================================================
println("\n" * "="^70)
println("Q3: CDLL profile for slope(τ=0.50) under WRONG eps quantiles")
println("="^70)

eps_scenarios = [
    ("eps true (±0.202)",             par_true.a_eps),
    ("eps half (±0.101)",             par_true.a_eps .* 0.5),
    ("eps double (±0.404)",           par_true.a_eps .* 2.0),
    ("eps zero",                      [0.0, 0.0, 0.0]),
    ("eps shifted (+0.1)",            par_true.a_eps .+ 0.1),
]

for (label, eps_q) in eps_scenarios
    cdll_vals = zeros(ngrid)
    for (ig, s) in enumerate(slope_grid)
        p = copy_params(par_true)
        p.a_Q[2, 2] = s
        p.a_eps .= eps_q
        cdll_vals[ig] = avg_cdll(p)
    end
    best_slope = slope_grid[argmax(cdll_vals)]
    @printf("  %-35s  CDLL maximizer: %.4f\n", label, best_slope)
end

# ================================================================
println("\n" * "="^70)
println("Q4: Tail rate profile when knots are WRONG")
println("="^70)
println("Profile b1_Q when slope is off from truth")

b1_grid = collect(range(0.5, 5.0, length=ngrid))

slope_offsets = [0.0, -0.1, +0.1, -0.2, +0.2]
for ds in slope_offsets
    cdll_vals = zeros(ngrid)
    for (ig, b1v) in enumerate(b1_grid)
        p = copy_params(par_true)
        p.a_Q[2, :] .= 0.8 + ds  # all slopes shifted
        p.b1_Q = b1v
        cdll_vals[ig] = avg_cdll(p)
    end
    best_b1 = b1_grid[argmax(cdll_vals)]
    @printf("  slope=%.2f: CDLL-maximizing b1_Q = %.3f (true: 2.0)\n",
            0.8+ds, best_b1)
end

# ================================================================
println("\n" * "="^70)
println("Q5: CDLL profile for slope(τ=0.50) when OTHER knots are wrong")
println("="^70)

knot_scenarios = [
    ("all knots at truth",           a_true),
    ("intcpt(τ=0.25) = 0 (true=-0.337)", begin a=copy(a_true); a[1,1]=0.0; a end),
    ("intcpt(τ=0.75) = 0 (true=0.337)",  begin a=copy(a_true); a[1,3]=0.0; a end),
    ("slope(τ=0.25) = 0.5 (true=0.8)",   begin a=copy(a_true); a[2,1]=0.5; a end),
    ("slope(τ=0.75) = 0.5 (true=0.8)",   begin a=copy(a_true); a[2,3]=0.5; a end),
]

for (label, a_base) in knot_scenarios
    cdll_vals = zeros(ngrid)
    for (ig, s) in enumerate(slope_grid)
        p = copy_params(par_true)
        p.a_Q .= a_base
        p.a_Q[2, 2] = s  # vary slope at τ=0.50
        cdll_vals[ig] = avg_cdll(p)
    end
    best_slope = slope_grid[argmax(cdll_vals)]
    @printf("  %-45s  CDLL maximizer: %.4f\n", label, best_slope)
end

println("\n" * "="^70)
println("DONE")
println("="^70)

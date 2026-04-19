#=
plot_qr_objective.jl — Plot ABB's objective functions along each parameter direction

Part 1: QR check-loss profiles for transition quantile parameters (a_Q)
  L(a_ℓ) = E_y E_{η|y,θ₀} [ ρ_{τ_ℓ}(η_t - h(η_{t-1})' a_ℓ) ]

Part 2: Complete-data log-likelihood profiles for ALL parameters
  (transition tail rates, ε quantiles/tails, initial η₁ quantiles/tails)
  Q(θ) = E_y E_{η|y,θ₀} [ ℓ(θ; y, η) ]

Both are approximated with posterior draws from E-step at true θ₀.
Profile: vary one parameter at a time, hold rest at truth.
=#

include("ABB_three_period.jl")

using Plots; gr()

# ================================================================
#  SETUP: Generate data, run E-step at true θ₀
# ================================================================
K = 2; L = 3; sigma_y = 1.0; T = 3
tau = [0.25, 0.50, 0.75]
N = 5000; M = 500; n_draws = 2000
vp = fill(0.05, T)
cfg = Config(N, T, K, L, tau, sigma_y, 1, n_draws, M, vp)

par_true = make_true_params_linear(tau=tau, sigma_y=sigma_y, K=K)
y, eta_true = generate_data_abb(N, par_true, tau, sigma_y, K; seed=42)

println("Running E-step with true parameters (N=$N, M=$M, n_draws=$n_draws)...")
eta_all = zeros(N, T, M)
for m in 1:M; eta_all[:,:,m] .= 0.6 .* y; end
acc = e_step!(eta_all, y, par_true, cfg)
@printf("  Acceptance: %.2f/%.2f/%.2f\n", acc...)

# Stack data for transition, initial, eps
eta_t, eta_lag = stack_transition(eta_all, cfg)
H = hermite_basis(eta_lag, K, sigma_y)
n_obs = length(eta_t)
eta1_all = stack_initial(eta_all, cfg)
eps_all = stack_eps(eta_all, y, cfg)

println("  Transition pairs: $n_obs")
println("  Initial η₁ draws: $(length(eta1_all))")
println("  ε draws: $(length(eps_all))")

# ================================================================
#  HELPER: complete-data log-likelihood averaged over posterior draws
# ================================================================
function avg_cdll(par::Params, eta_all::Array{Float64,3},
                  y::Matrix{Float64}, cfg::Config)
    N,T,M = cfg.N, cfg.T, cfg.M
    q_buf = zeros(cfg.L)
    ll = 0.0
    for m in 1:M, i in 1:N
        ll += full_loglik(view(y,i,:), view(eta_all,i,:,m), par, cfg, q_buf)
    end
    ll / (N * M)
end

# True values
a_true = copy(par_true.a_Q)

# Open output file
io = open("results_qr_objective.txt", "w")
function pb(args...); print(stdout, args...); print(io, args...); end
function plb(args...); println(stdout, args...); println(io, args...); end

plb("="^70)
plb("  ABB OBJECTIVE PROFILES")
plb("  N=$N, M=$M, n_obs=$n_obs")
plb("="^70)

ngrid = 101

# ================================================================
#  PART 1: QR CHECK-LOSS PROFILES (transition a_Q parameters)
# ================================================================
plb("\n--- PART 1: QR Check-Loss Profiles (transition quantile params) ---")

directions_aQ = [
    ("slope(τ=0.25)",     2, 1, range(0.4, 1.2, length=ngrid)),
    ("slope(τ=0.50)",     2, 2, range(0.4, 1.2, length=ngrid)),
    ("slope(τ=0.75)",     2, 3, range(0.4, 1.2, length=ngrid)),
    ("intcpt(τ=0.25)",    1, 1, range(-0.8, 0.2, length=ngrid)),
    ("intcpt(τ=0.50)",    1, 2, range(-0.4, 0.4, length=ngrid)),
    ("intcpt(τ=0.75)",    1, 3, range(-0.2, 0.8, length=ngrid)),
    ("quad(τ=0.25)",      3, 1, range(-0.3, 0.3, length=ngrid)),
    ("quad(τ=0.50)",      3, 2, range(-0.3, 0.3, length=ngrid)),
    ("quad(τ=0.75)",      3, 3, range(-0.3, 0.3, length=ngrid)),
]

aQ_plots = []
for (name, row, col, grid) in directions_aQ
    true_val = a_true[row, col]
    tau_l = tau[col]
    losses = zeros(length(grid))
    for (ig, gval) in enumerate(grid)
        a_test = copy(a_true)
        a_test[row, col] = gval
        r = eta_t .- H * a_test[:, col]
        losses[ig] = mean(r .* (tau_l .- (r .< 0)))
    end
    min_idx = argmin(losses)
    min_val = collect(grid)[min_idx]
    plb(@sprintf("  %s: true=%.4f, min=%.4f, diff=%+.4f",
                 name, true_val, min_val, min_val - true_val))
    p = plot(collect(grid), losses,
             xlabel=name, ylabel="check loss (τ=$(tau[col]))",
             title=name, legend=false, linewidth=2)
    vline!(p, [true_val], linestyle=:dash, color=:red, linewidth=1.5)
    vline!(p, [min_val], linestyle=:dot, color=:blue, linewidth=1.5)
    push!(aQ_plots, p)
end

fig1 = plot(aQ_plots..., layout=(3,3), size=(1400,1000),
            plot_title="QR Check-Loss Profiles: Transition Parameters\n(red=true, blue=minimizer)")
savefig(fig1, "qr_profiles_transition.png")
plb("  Saved qr_profiles_transition.png")

# ================================================================
#  PART 2: COMPLETE-DATA LOG-LIKELIHOOD PROFILES
#  Vary each parameter one at a time, compute avg CDLL
# ================================================================
plb("\n--- PART 2: Complete-Data Log-Likelihood Profiles (all params) ---")

cdll_true = avg_cdll(par_true, eta_all, y, cfg)
plb(@sprintf("  CDLL at truth: %.6f", cdll_true))

# Helper: profile one parameter
function profile_cdll(par_base::Params, set_fn!, grid, name, true_val)
    losses = zeros(length(grid))
    for (ig, gval) in enumerate(grid)
        p = copy_params(par_base)
        set_fn!(p, gval)
        losses[ig] = avg_cdll(p, eta_all, y, cfg)
    end
    min_idx = argmax(losses)  # maximize log-likelihood
    min_val = collect(grid)[min_idx]
    plb(@sprintf("  %s: true=%.4f, max=%.4f, diff=%+.4f",
                 name, true_val, min_val, min_val - true_val))
    p = plot(collect(grid), losses,
             xlabel=name, ylabel="avg CDLL",
             title=name, legend=false, linewidth=2)
    vline!(p, [true_val], linestyle=:dash, color=:red, linewidth=1.5)
    vline!(p, [min_val], linestyle=:dot, color=:blue, linewidth=1.5)
    p
end

cdll_plots = []

# --- Transition slopes ---
for l in 1:L
    grid = range(0.4, 1.2, length=ngrid)
    name = "a_Q[2,$l] (slope τ=$(tau[l]))"
    tv = par_true.a_Q[2,l]
    p = profile_cdll(par_true,
        (par, v) -> (par.a_Q[2,l] = v),
        grid, name, tv)
    push!(cdll_plots, p)
end

# --- Transition intercepts ---
for l in 1:L
    grid = range(par_true.a_Q[1,l]-0.5, par_true.a_Q[1,l]+0.5, length=ngrid)
    name = "a_Q[1,$l] (intcpt τ=$(tau[l]))"
    tv = par_true.a_Q[1,l]
    p = profile_cdll(par_true,
        (par, v) -> (par.a_Q[1,l] = v),
        grid, name, tv)
    push!(cdll_plots, p)
end

# --- Transition tail rates ---
for (name, field, tv) in [
    ("b1_Q", :b1_Q, par_true.b1_Q),
    ("bL_Q", :bL_Q, par_true.bL_Q)]
    grid = range(0.5, 5.0, length=ngrid)
    p = profile_cdll(par_true,
        (par, v) -> setfield!(par, Symbol(field), v),
        grid, name, tv)
    push!(cdll_plots, p)
end

fig2 = plot(cdll_plots[1:8]..., layout=(2,4), size=(1600,800),
            plot_title="CDLL Profiles: Transition Parameters\n(red=true, blue=maximizer)")
savefig(fig2, "cdll_profiles_transition.png")
plb("  Saved cdll_profiles_transition.png")

# --- ε quantiles ---
eps_plots = []
for l in 1:L
    grid = range(par_true.a_eps[l]-0.3, par_true.a_eps[l]+0.3, length=ngrid)
    name = "a_eps[$l] (τ=$(tau[l]))"
    tv = par_true.a_eps[l]
    p = profile_cdll(par_true,
        (par, v) -> (par.a_eps[l] = v),
        grid, name, tv)
    push!(eps_plots, p)
end

# --- ε tail rates ---
for (name, field, tv) in [
    ("b1_eps", :b1_eps, par_true.b1_eps),
    ("bL_eps", :bL_eps, par_true.bL_eps)]
    grid = range(0.5, 8.0, length=ngrid)
    p = profile_cdll(par_true,
        (par, v) -> setfield!(par, Symbol(field), v),
        grid, name, tv)
    push!(eps_plots, p)
end

# --- Initial η₁ quantiles ---
for l in 1:L
    grid = range(par_true.a_init[l]-0.5, par_true.a_init[l]+0.5, length=ngrid)
    name = "a_init[$l] (τ=$(tau[l]))"
    tv = par_true.a_init[l]
    p = profile_cdll(par_true,
        (par, v) -> (par.a_init[l] = v),
        grid, name, tv)
    push!(eps_plots, p)
end

# --- Initial η₁ tail rates ---
for (name, field, tv) in [
    ("b1_init", :b1_init, par_true.b1_init),
    ("bL_init", :bL_init, par_true.bL_init)]
    grid = range(0.2, 4.0, length=ngrid)
    p = profile_cdll(par_true,
        (par, v) -> setfield!(par, Symbol(field), v),
        grid, name, tv)
    push!(eps_plots, p)
end

fig3 = plot(eps_plots..., layout=(3,4), size=(1600,1000),
            plot_title="CDLL Profiles: ε and Initial η₁ Parameters\n(red=true, blue=maximizer)")
savefig(fig3, "cdll_profiles_eps_init.png")
plb("  Saved cdll_profiles_eps_init.png")

# ================================================================
#  PART 3: 2D CONTOURS
# ================================================================
plb("\n--- PART 3: 2D Contours ---")

# Joint slope variation
plb("  Computing joint slope profile...")
rho_grid = range(0.3, 1.1, length=ngrid)
joint_losses = zeros(length(rho_grid))
for (ig, rho) in enumerate(rho_grid)
    a_test = copy(a_true)
    a_test[2, :] .= rho * sigma_y
    joint_losses[ig] = sum(begin
        r = eta_t .- H * a_test[:, l]
        mean(r .* (tau[l] .- (r .< 0)))
    end for l in 1:L)
end
min_idx = argmin(joint_losses)
min_rho = collect(rho_grid)[min_idx]
plb(@sprintf("  Joint ρ: true=0.8000, min=%.4f, diff=%+.4f", min_rho, min_rho-0.8))

p_joint = plot(collect(rho_grid), joint_losses,
               xlabel="ρ (common slope/σ_y)", ylabel="total check loss",
               title="Total Check Loss vs Common ρ",
               legend=false, linewidth=2)
vline!(p_joint, [0.8], linestyle=:dash, color=:red, linewidth=1.5)
vline!(p_joint, [min_rho], linestyle=:dot, color=:blue, linewidth=1.5)
savefig(p_joint, "qr_objective_joint_slope.png")
plb("  Saved qr_objective_joint_slope.png")

# 2D: slope(τ=0.25) vs slope(τ=0.75)
plb("  Computing 2D contour: slope(τ=0.25) vs slope(τ=0.75)...")
ng2 = 51
r1g = range(0.5, 1.1, length=ng2)
r3g = range(0.5, 1.1, length=ng2)
loss_2d = zeros(ng2, ng2)
for (i1, rv1) in enumerate(r1g), (i3, rv3) in enumerate(r3g)
    a_test = copy(a_true)
    a_test[2,1] = rv1 * sigma_y; a_test[2,3] = rv3 * sigma_y
    loss_2d[i3,i1] = sum(begin
        r = eta_t .- H * a_test[:, l]
        mean(r .* (tau[l] .- (r .< 0)))
    end for l in 1:L)
end
p_c1 = contour(collect(r1g), collect(r3g), loss_2d,
               xlabel="slope(τ=0.25)/σ_y", ylabel="slope(τ=0.75)/σ_y",
               title="Total Check Loss", fill=true, levels=30)
scatter!(p_c1, [0.8], [0.8], color=:red, markersize=8, markershape=:star5, label="true")
savefig(p_c1, "qr_objective_contour_slopes.png")
plb("  Saved qr_objective_contour_slopes.png")

# 2D: slope(τ=0.50) vs intercept(τ=0.50)
plb("  Computing 2D contour: slope vs intercept at τ=0.50...")
sg = range(0.5, 1.1, length=ng2)
ig2 = range(-0.3, 0.3, length=ng2)
loss_2d_si = zeros(ng2, ng2)
for (is, sv) in enumerate(sg), (ii, iv) in enumerate(ig2)
    a_test = copy(a_true)
    a_test[2,2] = sv * sigma_y; a_test[1,2] = iv
    r = eta_t .- H * a_test[:, 2]
    loss_2d_si[ii,is] = mean(r .* (tau[2] .- (r .< 0)))
end
p_c2 = contour(collect(sg), collect(ig2), loss_2d_si,
               xlabel="slope(τ=0.50)/σ_y", ylabel="intcpt(τ=0.50)",
               title="Check Loss (τ=0.50)", fill=true, levels=30)
scatter!(p_c2, [0.8], [0.0], color=:red, markersize=8, markershape=:star5, label="true")
savefig(p_c2, "qr_objective_contour_slope_intcpt.png")
plb("  Saved qr_objective_contour_slope_intcpt.png")

# 2D CDLL: b1_eps vs bL_eps
plb("  Computing 2D CDLL contour: b1_eps vs bL_eps...")
b1g = range(1.0, 6.0, length=ng2)
bLg = range(1.0, 6.0, length=ng2)
cdll_2d = zeros(ng2, ng2)
for (i1, b1v) in enumerate(b1g), (i2, bLv) in enumerate(bLg)
    p = copy_params(par_true)
    p.b1_eps = b1v; p.bL_eps = bLv
    cdll_2d[i2,i1] = avg_cdll(p, eta_all, y, cfg)
end
p_c3 = contour(collect(b1g), collect(bLg), cdll_2d,
               xlabel="b1_eps", ylabel="bL_eps",
               title="CDLL: b1_eps vs bL_eps", fill=true, levels=30)
scatter!(p_c3, [par_true.b1_eps], [par_true.bL_eps], color=:red, markersize=8,
         markershape=:star5, label="true")
savefig(p_c3, "cdll_contour_eps_tails.png")
plb("  Saved cdll_contour_eps_tails.png")

plb("\n" * "="^70)
plb("  DONE")
plb("="^70)
close(io)
println("\nResults saved to results_qr_objective.txt")

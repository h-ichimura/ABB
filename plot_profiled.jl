#=
plot_profiled.jl — Re-plot ABB objective profiles with non-knot parameters
profiled out using their closed-form MLE.

For each grid value of the knot parameter being varied:
  - Transition tail rates b1_Q, bL_Q: closed-form MLE given current knots
  - eps quantiles and tails: sample quantiles/MLE from draws (constant)
  - init quantiles and tails: sample quantiles/MLE from draws (constant)

Plots: profiled CDLL profiles for all knot parameters, same layout
as plot_qr_objective.jl
=#

include("ABB_three_period.jl")

using Plots; gr()

K=2; L=3; sigma_y=1.0; T=3
tau = [0.25, 0.50, 0.75]
N=2000; M=50; n_draws=200
vp = fill(0.05, T)
cfg = Config(N, T, K, L, tau, sigma_y, 1, n_draws, M, vp)

par_true = make_true_params_linear(tau=tau, sigma_y=sigma_y, K=K)
y, eta_true = generate_data_abb(N, par_true, tau, sigma_y, K; seed=42)

println("Running E-step at true parameters (N=$N, M=$M)...")
eta_all = zeros(N, T, M)
for m in 1:M; eta_all[:,:,m] .= 0.6 .* y; end
acc = e_step!(eta_all, y, par_true, cfg)
@printf("Acceptance: %.2f/%.2f/%.2f\n", acc...)

eta_t, eta_lag = stack_transition(eta_all, cfg)
H = hermite_basis(eta_lag, K, sigma_y)
n_obs = length(eta_t)
Q_mat = zeros(n_obs, L)

# MLE of marginal params (independent of transition knots)
eta1_all = stack_initial(eta_all, cfg)
eps_all = stack_eps(eta_all, y, cfg)
a_init_mle = [quantile(eta1_all, tau[l]) for l in 1:L]
b1_init_mle, bL_init_mle = update_tails(eta1_all, a_init_mle[1], a_init_mle[L])
a_eps_mle = [quantile(eps_all, tau[l]) for l in 1:L]
a_eps_mle .-= mean(a_eps_mle)
b1_eps_mle, bL_eps_mle = update_tails(eps_all, a_eps_mle[1], a_eps_mle[L])

a_true = copy(par_true.a_Q)

"""Profiled CDLL: given a_Q, compute closed-form MLE of b1_Q/bL_Q,
set marginals at their MLE, return avg CDLL."""
function profiled_cdll(a_Q)
    mul!(Q_mat, H, a_Q)
    r1 = eta_t .- view(Q_mat,:,1); rL = eta_t .- view(Q_mat,:,L)
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
    ll / (N*M)
end

"""Fixed CDLL: all non-knot params fixed at truth."""
function fixed_cdll(a_Q)
    par = Params(a_Q, par_true.b1_Q, par_true.bL_Q,
                 par_true.a_init, par_true.b1_init, par_true.bL_init,
                 par_true.a_eps, par_true.b1_eps, par_true.bL_eps)
    q_buf = zeros(L); ll = 0.0
    for m in 1:M, i in 1:N
        ll += full_loglik(view(y,i,:), view(eta_all,i,:,m), par, cfg, q_buf)
    end
    ll / (N*M)
end

"""QR check-loss at quantile level l."""
function qr_loss(a_Q, l)
    r = eta_t .- H * a_Q[:, l]
    mean(r .* (tau[l] .- (r .< 0)))
end

ngrid = 81

# ================================================================
#  SLOPE PROFILES
# ================================================================
println("Computing slope profiles...")
slope_grid = collect(range(0.4, 1.2, length=ngrid))

slope_plots = []
for l in 1:L
    prof_vals = zeros(ngrid)
    qr_vals   = zeros(ngrid)
    for (ig, s) in enumerate(slope_grid)
        a = copy(a_true); a[2,l] = s
        prof_vals[ig] = profiled_cdll(a)
        qr_vals[ig]   = qr_loss(a, l)
    end
    true_val = a_true[2,l]

    # Plot negative CDLL (so both are minimized), shift to same minimum
    neg_cdll = -prof_vals
    neg_cdll_shifted = neg_cdll .- minimum(neg_cdll)
    qr_shifted = qr_vals .- minimum(qr_vals)

    p = plot(slope_grid, neg_cdll_shifted, label="-CDLL (profiled)",
             linewidth=2, color=:blue)
    plot!(p, slope_grid, qr_shifted, label="QR check-loss",
          linewidth=2, color=:green)
    vline!(p, [true_val], label="truth", color=:red, linestyle=:dash, linewidth=1.5)
    xlabel!(p, "slope (τ=$(tau[l]))")
    ylabel!(p, "objective (shifted, min=0)")
    title!(p, "slope(τ=$(tau[l]))")
    plot!(p, legend=:top)

    push!(slope_plots, p)
end

fig1 = plot(slope_plots..., layout=(1,3), size=(1500,400),
            plot_title="Slope Profiles: CDLL (blue) vs QR (green), red=truth")
savefig(fig1, "profiled_slopes.png")
println("  Saved profiled_slopes.png")

# ================================================================
#  INTERCEPT PROFILES
# ================================================================
println("Computing intercept profiles...")
intcpt_grid = collect(range(-0.8, 0.8, length=ngrid))

intcpt_plots = []
for l in 1:L
    prof_vals = zeros(ngrid)
    qr_vals   = zeros(ngrid)
    for (ig, a0) in enumerate(intcpt_grid)
        a = copy(a_true); a[1,l] = a0
        prof_vals[ig] = profiled_cdll(a)
        qr_vals[ig]   = qr_loss(a, l)
    end
    true_val = a_true[1,l]

    neg_cdll = -prof_vals
    neg_cdll_shifted = neg_cdll .- minimum(neg_cdll)
    qr_shifted = qr_vals .- minimum(qr_vals)

    p = plot(intcpt_grid, neg_cdll_shifted, label="-CDLL (profiled)", linewidth=2, color=:blue)
    plot!(p, intcpt_grid, qr_shifted, label="QR check-loss", linewidth=2, color=:green)
    vline!(p, [true_val], label="truth", color=:red, linestyle=:dash, linewidth=1.5)
    xlabel!(p, "intercept (τ=$(tau[l]))")
    ylabel!(p, "objective (shifted, min=0)")
    title!(p, "intcpt(τ=$(tau[l]))")
    plot!(p, legend=:top)

    push!(intcpt_plots, p)
end

fig2 = plot(intcpt_plots..., layout=(1,3), size=(1500,400),
            plot_title="Intercept Profiles: CDLL (blue) vs QR (green), red=truth")
savefig(fig2, "profiled_intercepts.png")
println("  Saved profiled_intercepts.png")

# ================================================================
#  QUADRATIC PROFILES
# ================================================================
println("Computing quadratic profiles...")
quad_grid = collect(range(-0.3, 0.3, length=ngrid))

quad_plots = []
for l in 1:L
    prof_vals = zeros(ngrid)
    qr_vals   = zeros(ngrid)
    for (ig, a2) in enumerate(quad_grid)
        a = copy(a_true); a[3,l] = a2
        prof_vals[ig] = profiled_cdll(a)
        qr_vals[ig]   = qr_loss(a, l)
    end
    true_val = a_true[3,l]

    neg_cdll = -prof_vals
    neg_cdll_shifted = neg_cdll .- minimum(neg_cdll)
    qr_shifted = qr_vals .- minimum(qr_vals)

    p = plot(quad_grid, neg_cdll_shifted, label="-CDLL (profiled)", linewidth=2, color=:blue)
    plot!(p, quad_grid, qr_shifted, label="QR check-loss", linewidth=2, color=:green)
    vline!(p, [true_val], label="truth", color=:red, linestyle=:dash, linewidth=1.5)
    xlabel!(p, "quadratic (τ=$(tau[l]))")
    ylabel!(p, "objective (shifted, min=0)")
    title!(p, "quad(τ=$(tau[l]))")
    plot!(p, legend=:top)

    push!(quad_plots, p)
end

fig3 = plot(quad_plots..., layout=(1,3), size=(1500,400),
            plot_title="Quadratic Profiles: CDLL (blue) vs QR (green), red=truth")
savefig(fig3, "profiled_quadratic.png")
println("  Saved profiled_quadratic.png")

println("\nDone. Generated: profiled_slopes.png, profiled_intercepts.png, profiled_quadratic.png")

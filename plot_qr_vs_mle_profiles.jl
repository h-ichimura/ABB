#=
plot_qr_vs_mle_profiles.jl — Profile QR check-loss and -CDLL over the same
slope parameter, aligned so minima are at 0. Shows that both objectives
are minimized at the true value, but -CDLL has sharper curvature (more
information → more efficient).
=#

include("ABB_three_period.jl")
using Plots; gr()

K = 2; L = 3; sigma_y = 1.0; T = 3
tau = [0.25, 0.50, 0.75]
N = 5000; M = 500; n_draws = 2000
vp = fill(0.05, T)
cfg = Config(N, T, K, L, tau, sigma_y, 1, n_draws, M, vp)

par_true = make_true_params_linear(tau=tau, sigma_y=sigma_y, K=K)
y, eta_true = generate_data_abb(N, par_true, tau, sigma_y, K; seed=42)

println("Running E-step at truth (N=$N, M=$M)...")
eta_all = zeros(N, T, M)
for m in 1:M; eta_all[:,:,m] .= 0.6 .* y; end
acc = e_step!(eta_all, y, par_true, cfg)
@printf("Acceptance: %.2f/%.2f/%.2f\n", acc...)

eta_t, eta_lag = stack_transition(eta_all, cfg)
H = hermite_basis(eta_lag, K, sigma_y)
n_obs = length(eta_t)

# CDLL helper
function avg_cdll(par)
    q_buf = zeros(L)
    ll = 0.0
    for m in 1:M, i in 1:N
        ll += full_loglik(view(y,i,:), view(eta_all,i,:,m), par, cfg, q_buf)
    end
    ll / (N * M)
end

ngrid = 201

# ================================================================
#  SLOPE PROFILES: QR check-loss and -CDLL side by side
# ================================================================
println("Computing profiles...")

p_combined = plot(layout=(1,3), size=(1400, 450))

for (col, tau_l) in enumerate(tau)
    grid = collect(range(0.4, 1.2, length=ngrid))
    true_val = par_true.a_Q[2, col]

    # QR check-loss at this quantile level
    qr_loss = zeros(ngrid)
    for (ig, gval) in enumerate(grid)
        a_test = copy(par_true.a_Q)
        a_test[2, col] = gval
        r = eta_t .- H * a_test[:, col]
        qr_loss[ig] = mean(r .* (tau_l .- (r .< 0)))
    end

    # -CDLL (neg log-lik, profiled over other params)
    neg_cdll = zeros(ngrid)
    for (ig, gval) in enumerate(grid)
        par_test = copy_params(par_true)
        par_test.a_Q[2, col] = gval
        neg_cdll[ig] = -avg_cdll(par_test)
    end

    # Normalize: shift both so minimum = 0
    qr_loss .-= minimum(qr_loss)
    neg_cdll .-= minimum(neg_cdll)

    # Scale: normalize so maximum over grid = 1 (for visual comparison)
    qr_scale = maximum(qr_loss)
    cdll_scale = maximum(neg_cdll)
    qr_normed = qr_loss ./ qr_scale
    cdll_normed = neg_cdll ./ cdll_scale

    plot!(p_combined[col], grid, qr_normed,
          label="QR check-loss", color=:blue, lw=2)
    plot!(p_combined[col], grid, cdll_normed,
          label="-CDLL", color=:red, lw=2, ls=:dash)
    vline!(p_combined[col], [true_val], label="truth",
           color=:black, ls=:dot, lw=1.5)
    title!(p_combined[col], "τ = $(tau_l)")
    xlabel!(p_combined[col], "slope a_{1,$(col)}")
    ylabel!(p_combined[col], "normalized loss")
end

savefig(p_combined, "profile_qr_vs_cdll_slopes.png")
println("Saved profile_qr_vs_cdll_slopes.png")

# ================================================================
#  INTERCEPT PROFILES
# ================================================================
p_intcpt = plot(layout=(1,3), size=(1400, 450))

for (col, tau_l) in enumerate(tau)
    true_val = par_true.a_Q[1, col]
    grid = collect(range(true_val - 0.5, true_val + 0.5, length=ngrid))

    qr_loss = zeros(ngrid)
    for (ig, gval) in enumerate(grid)
        a_test = copy(par_true.a_Q)
        a_test[1, col] = gval
        r = eta_t .- H * a_test[:, col]
        qr_loss[ig] = mean(r .* (tau_l .- (r .< 0)))
    end

    neg_cdll = zeros(ngrid)
    for (ig, gval) in enumerate(grid)
        par_test = copy_params(par_true)
        par_test.a_Q[1, col] = gval
        neg_cdll[ig] = -avg_cdll(par_test)
    end

    qr_loss .-= minimum(qr_loss)
    neg_cdll .-= minimum(neg_cdll)
    qr_normed = qr_loss ./ max(maximum(qr_loss), 1e-10)
    cdll_normed = neg_cdll ./ max(maximum(neg_cdll), 1e-10)

    plot!(p_intcpt[col], grid, qr_normed,
          label="QR check-loss", color=:blue, lw=2)
    plot!(p_intcpt[col], grid, cdll_normed,
          label="-CDLL", color=:red, lw=2, ls=:dash)
    vline!(p_intcpt[col], [true_val], label="truth",
           color=:black, ls=:dot, lw=1.5)
    title!(p_intcpt[col], "τ = $(tau_l)")
    xlabel!(p_intcpt[col], "intercept a_{0,$(col)}")
    ylabel!(p_intcpt[col], "normalized loss")
end

savefig(p_intcpt, "profile_qr_vs_cdll_intercepts.png")
println("Saved profile_qr_vs_cdll_intercepts.png")

# ================================================================
#  UNNORMALIZED: show raw curvature difference
# ================================================================
p_raw = plot(layout=(1,3), size=(1400, 450))

for (col, tau_l) in enumerate(tau)
    grid = collect(range(0.5, 1.1, length=ngrid))
    true_val = par_true.a_Q[2, col]

    qr_loss = zeros(ngrid)
    neg_cdll = zeros(ngrid)
    for (ig, gval) in enumerate(grid)
        a_test = copy(par_true.a_Q)
        a_test[2, col] = gval
        r = eta_t .- H * a_test[:, col]
        qr_loss[ig] = mean(r .* (tau_l .- (r .< 0)))

        par_test = copy_params(par_true)
        par_test.a_Q[2, col] = gval
        neg_cdll[ig] = -avg_cdll(par_test)
    end

    qr_loss .-= minimum(qr_loss)
    neg_cdll .-= minimum(neg_cdll)

    plot!(p_raw[col], grid, qr_loss,
          label="QR check-loss", color=:blue, lw=2, yaxis=:left)
    plot!(twinx(p_raw[col]), grid, neg_cdll,
          label="-CDLL", color=:red, lw=2, ls=:dash, yaxis=:right,
          legend=:topright)
    vline!(p_raw[col], [true_val], label="", color=:black, ls=:dot, lw=1.5)
    title!(p_raw[col], "τ = $(tau_l)")
    xlabel!(p_raw[col], "slope a_{1,$(col)}")
end

savefig(p_raw, "profile_qr_vs_cdll_raw.png")
println("Saved profile_qr_vs_cdll_raw.png")

println("\nDone!")

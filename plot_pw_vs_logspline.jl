#=
plot_pw_vs_logspline.jl — Plot ABB's piecewise-uniform conditional density
alongside a logspline initialized from the same parameters.

Shows: for several values of η_{t-1}, the conditional density f(η_t | η_{t-1})
under (a) ABB's piecewise-uniform, (b) logspline fit to the same knots.
=#

include("ABB_three_period.jl")
include("logspline.jl")
using Plots; gr()

# ── Run ABB QR to get parameter estimates ─────────────────────────
N = 1000; K = 2; L = 3; sigma_y = 1.0
tau = collect(range(1/(L+1), stop=L/(L+1), length=L))  # [0.25, 0.50, 0.75]
par_true = make_true_params_linear(tau=tau, sigma_y=sigma_y, K=K)
y, eta_true = generate_data_abb(N, par_true, tau, sigma_y, K; seed=42)

cfg = Config(N, 3, K, L, tau, sigma_y, 100, 200, 50, fill(0.05, 3))
println("Running QR estimation...")
par_qr, _, _, _, _ = estimate(y, cfg; method=:qr)

println("\nQR estimates:")
println("  a_Q = ", round.(par_qr.a_Q, digits=4))
println("  b1_Q = ", round(par_qr.b1_Q, digits=4))
println("  bL_Q = ", round(par_qr.bL_Q, digits=4))
println("  Persistence: ", round.(par_qr.a_Q[2,:] ./ sigma_y, digits=4))

println("\nTrue params:")
println("  a_Q = ", round.(par_true.a_Q, digits=4))
println("  b1_Q = ", round(par_true.b1_Q, digits=4))
println("  bL_Q = ", round(par_true.bL_Q, digits=4))

# ── Piecewise-uniform density function ────────────────────────────
function pw_density(x::Float64, q::Vector{Float64}, tau::Vector{Float64},
                    b1::Float64, bL::Float64)
    exp(pw_logdens(x, q, tau, b1, bL))
end

# ── Fit logspline to piecewise-uniform at given η_{t-1} ──────────
"""
Given ABB parameters and a value of η_{t-1}, fit a logspline whose
knots are at the ABB quantile values Q_1, Q_2, Q_3.

The logspline is: log f(x) = β₀ + β₁x + Σγₖ(x - Qₖ)₊³ - log C

We fit (β₀, β₁, γ₁,...,γ_L) by minimizing KL divergence from the
piecewise-uniform density, using numerical integration.
"""
function fit_logspline_to_pw(eta_lag::Float64, par::Params, cfg::Config)
    q = zeros(cfg.L)
    transition_quantiles!(q, eta_lag, par.a_Q, cfg.K, cfg.sigma_y)

    # Use ABB quantile locations as logspline knots
    knots = copy(q)
    K_sp = length(knots)

    # Target: piecewise-uniform density
    function pw_logf(x)
        pw_logdens(x, q, cfg.tau, par.b1_Q, par.bL_Q)
    end

    # Fit by minimizing negative expected log-likelihood under pw density:
    # min_θ -∫ f_pw(x) * log f_ls(x; θ) dx
    # = min_θ -∫ f_pw(x) * [s(x;θ) - log C(θ)] dx
    # = min_θ -E_pw[s(x;θ)] + log C(θ)

    # Integration range
    lo = q[1] - 3.0 / par.b1_Q
    hi = q[end] + 3.0 / par.bL_Q

    function neg_expected_loglik(theta)
        beta0 = theta[1]
        beta1 = theta[2]
        gamma = theta[3:end]

        # log C
        f_unnorm(x) = exp(logspline_s(x, beta0, beta1, gamma, knots))

        # Split integration at knots for accuracy
        bps = vcat(lo, knots[knots .> lo .&& knots .< hi], hi)
        C = 0.0
        for i in 1:length(bps)-1
            C += gl_integrate(f_unnorm, bps[i], bps[i+1];
                              nodes=GL16_nodes, weights=GL16_weights)
        end
        C = max(C, 1e-300)
        logC = log(C)

        # E_pw[s(x)] = ∫ f_pw(x) * s(x) dx
        function integrand(x)
            exp(pw_logf(x)) * logspline_s(x, beta0, beta1, gamma, knots)
        end
        E_s = 0.0
        for i in 1:length(bps)-1
            E_s += gl_integrate(integrand, bps[i], bps[i+1];
                                nodes=GL16_nodes, weights=GL16_weights)
        end

        -E_s + logC
    end

    # Initial: β₀=0, β₁=0, γ=-0.1 (mild tail decay)
    theta0 = zeros(K_sp + 2)
    theta0[3:end] .= -0.1

    result = optimize(neg_expected_loglik, theta0, NelderMead(),
                      Optim.Options(iterations=2000))

    theta = Optim.minimizer(result)
    beta0 = theta[1]; beta1 = theta[2]; gamma = theta[3:end]

    # Compute log C for the fitted logspline
    f_unnorm(x) = exp(logspline_s(x, beta0, beta1, gamma, knots))
    bps = vcat(lo, knots[knots .> lo .&& knots .< hi], hi)
    C = sum(gl_integrate(f_unnorm, bps[i], bps[i+1];
                         nodes=GL16_nodes, weights=GL16_weights) for i in 1:length(bps)-1)
    logC = log(max(C, 1e-300))

    (beta0=beta0, beta1=beta1, gamma=gamma, knots=knots, logC=logC,
     lo=lo, hi=hi, loss=Optim.minimum(result))
end

# ── Plot ──────────────────────────────────────────────────────────
eta_lag_values = [-1.0, -0.5, 0.0, 0.5, 1.0]
n_panels = length(eta_lag_values)

p = plot(layout=(1, n_panels), size=(300 * n_panels, 350),
         xlabel="η_t", ylabel="density")

open("results_pw_vs_logspline.txt", "w") do io
    println(io, "Piecewise-uniform vs logspline conditional densities")
    println(io, "="^60)

    for (j, eta_lag) in enumerate(eta_lag_values)
        q = zeros(L)
        transition_quantiles!(q, eta_lag, par_qr.a_Q, K, sigma_y)

        # Piecewise-uniform density on a grid
        lo = q[1] - 3.0 / par_qr.b1_Q
        hi = q[end] + 3.0 / par_qr.bL_Q
        xgrid = range(lo, hi, length=500)
        pw_dens = [pw_density(x, q, tau, par_qr.b1_Q, par_qr.bL_Q) for x in xgrid]

        # Fit logspline
        ls_fit = fit_logspline_to_pw(eta_lag, par_qr, cfg)

        ls_dens = [exp(logspline_s(x, ls_fit.beta0, ls_fit.beta1,
                                    ls_fit.gamma, ls_fit.knots) - ls_fit.logC)
                    for x in xgrid]

        # Plot
        plot!(p[j], xgrid, pw_dens, label="PW-uniform", color=:blue, lw=2)
        plot!(p[j], xgrid, ls_dens, label="Logspline", color=:red, lw=2, ls=:dash)
        vline!(p[j], q, label="", color=:gray, ls=:dot, alpha=0.5)
        title!(p[j], "η_{t-1} = $eta_lag")

        # Report
        @printf(io, "\nη_{t-1} = %+.2f:\n", eta_lag)
        @printf(io, "  Quantile knots: Q = [%s]\n",
                join([@sprintf("%.4f", qi) for qi in q], ", "))
        @printf(io, "  Logspline β₀=%.4f, β₁=%.4f\n", ls_fit.beta0, ls_fit.beta1)
        @printf(io, "  Logspline γ = [%s]\n",
                join([@sprintf("%.4f", gi) for gi in ls_fit.gamma], ", "))
        @printf(io, "  KL fit loss = %.6f\n", ls_fit.loss)
    end
end

savefig(p, "pw_vs_logspline.png")
println("\nSaved pw_vs_logspline.png")
println("Saved results_pw_vs_logspline.txt")

# Also plot with TRUE parameters
p2 = plot(layout=(1, n_panels), size=(300 * n_panels, 350),
          xlabel="η_t", ylabel="density")

for (j, eta_lag) in enumerate(eta_lag_values)
    q = zeros(L)
    transition_quantiles!(q, eta_lag, par_true.a_Q, K, sigma_y)

    lo = q[1] - 3.0 / par_true.b1_Q
    hi = q[end] + 3.0 / par_true.bL_Q
    xgrid = range(lo, hi, length=500)
    pw_dens = [pw_density(x, q, tau, par_true.b1_Q, par_true.bL_Q) for x in xgrid]

    ls_fit = fit_logspline_to_pw(eta_lag, par_true, cfg)
    ls_dens = [exp(logspline_s(x, ls_fit.beta0, ls_fit.beta1,
                                ls_fit.gamma, ls_fit.knots) - ls_fit.logC)
                for x in xgrid]

    plot!(p2[j], xgrid, pw_dens, label="PW-uniform", color=:blue, lw=2)
    plot!(p2[j], xgrid, ls_dens, label="Logspline", color=:red, lw=2, ls=:dash)
    vline!(p2[j], q, label="", color=:gray, ls=:dot, alpha=0.5)
    title!(p2[j], "η_{t-1} = $eta_lag (TRUE)")
end

savefig(p2, "pw_vs_logspline_true.png")
println("Saved pw_vs_logspline_true.png")

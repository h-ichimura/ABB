#=
test_init_sensitivity.jl — Test QR EM sensitivity to η_1 parameter initialization

Run EM with QR M-step starting from different initial values for a_init and
tail rates. If QR is self-referential for f_init, the final estimates should
track the initial values rather than converge to truth.
=#

include("ABB_three_period.jl")
using Printf, Serialization

K = 2; L = 3; sigma_y = 1.0; T = 3; N = 200; M = 50
tau = [0.25, 0.50, 0.75]

par_true = make_true_params_linear(tau=tau, sigma_y=sigma_y, K=K)
y, eta_true = generate_data_abb(N, par_true, tau, sigma_y, K; seed=42)
vp = fill(0.05, T)
cfg = Config(N, T, K, L, tau, sigma_y, 100, 200, M, vp)

println("="^70)
println("  QR EM — sensitivity to η_1 initial values")
println("="^70)
@printf("True:  a_init = [%.4f, %.4f, %.4f]  b1_init = %.2f  bL_init = %.2f\n\n",
        par_true.a_init..., par_true.b1_init, par_true.bL_init)

# Four initial configurations for η_1 parameters (others match truth or ABB init)
init_configs = [
    ("Narrow", [-0.3, 0.0, 0.3], 2.0, 2.0),
    ("True",   [-0.6742, 0.0, 0.6742], 1.0, 1.0),
    ("Wide",   [-1.5, 0.0, 1.5], 0.5, 0.5),
    ("Off-center", [0.0, 0.5, 1.0], 1.0, 1.0),
]

results = Dict{String, Any}()

for (label, a_init_init, b1_init_init, bL_init_init) in init_configs
    @printf("--- Starting config: %s ---\n", label)
    @printf("  a_init start = [%.4f, %.4f, %.4f]  b1=%.2f  bL=%.2f\n",
            a_init_init..., b1_init_init, bL_init_init)

    # Run EM with custom initial η_1 params.
    # Seed the RNG so all runs share the same data and initial η.
    par = init_params(y, cfg)
    par.a_init .= a_init_init
    par.b1_init = b1_init_init
    par.bL_init = bL_init_init

    # Rest of the parameters: start at truth so we isolate the η_1 effect
    par.a_Q .= par_true.a_Q
    par.b1_Q = par_true.b1_Q; par.bL_Q = par_true.bL_Q
    par.a_eps .= par_true.a_eps
    par.b1_eps = par_true.b1_eps; par.bL_eps = par_true.bL_eps

    # Pre-allocate eta draws matrix
    eta_all = zeros(N, T, M)
    for m in 1:M; eta_all[:, :, m] .= 0.6 .* y; end

    # Run EM with QR M-step
    S = cfg.maxiter
    hist_a_init = zeros(L, S)
    hist_b1_init = zeros(S)
    hist_bL_init = zeros(S)

    for iter in 1:S
        e_step!(eta_all, y, par, cfg)
        m_step_qr!(par, eta_all, y, cfg)
        hist_a_init[:, iter] .= par.a_init
        hist_b1_init[iter] = par.b1_init
        hist_bL_init[iter] = par.bL_init
    end

    # Average over last S/2 iterations
    S2 = div(S, 2); rng = S-S2+1:S
    a_init_avg = [mean(hist_a_init[l, rng]) for l in 1:L]
    b1_avg = mean(hist_b1_init[rng])
    bL_avg = mean(hist_bL_init[rng])

    @printf("  FINAL avg: a_init = [%.4f, %.4f, %.4f]  b1=%.4f  bL=%.4f\n",
            a_init_avg..., b1_avg, bL_avg)
    @printf("  Bias vs truth: Δa = [%.4f, %.4f, %.4f]  Δb1=%.4f  ΔbL=%.4f\n\n",
            (a_init_avg .- par_true.a_init)..., b1_avg - 1.0, bL_avg - 1.0)

    results[label] = (a=a_init_avg, b1=b1_avg, bL=bL_avg,
                       hist_a=hist_a_init, hist_b1=hist_b1_init, hist_bL=hist_bL_init)
end

println("="^70)
println("  SUMMARY")
println("="^70)
println("\nTrue a_init = $(par_true.a_init), b1=1.0, bL=1.0")
println()
@printf("%-12s  %-35s  %-18s\n", "Init", "Final a_init", "Final b1/bL")
for (label, _, _, _) in init_configs
    r = results[label]
    @printf("%-12s  [%7.4f, %7.4f, %7.4f]  %.4f / %.4f\n",
            label, r.a..., r.b1, r.bL)
end

serialize("results_init_sensitivity.jls", results)
println("\nSaved results_init_sensitivity.jls")

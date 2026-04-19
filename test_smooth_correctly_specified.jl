#=
test_smooth_correctly_specified.jl

1. Pick known true parameters (a_Q, b_coef, marginals)
2. Generate data FROM THE SMOOTH MODEL
3. Estimate by exact ML with LBFGS
4. Check that estimates converge to truth

This is the correctly-specified case: fit the same model that generated
the data. If MLE works here, the finite-sample behavior is interpretable.
=#

include("smooth_ml.jl")
using Printf, Serialization

K = 2; L = 3; sigma_y = 1.0; T = 3
tau = [0.25, 0.50, 0.75]

# Pick the TRUE parameters
par_true = make_true_params_linear(tau=tau, sigma_y=sigma_y, K=K,
                                    rho=0.8, sigma_v=0.5, sigma_eps=0.3)
b_true = init_smooth_b_coef(K, L)  # β_1=2, γ_l=-0.1 each

println("="^60)
println("  CORRECTLY-SPECIFIED SMOOTH MODEL TEST")
println("="^60)
println("\nTrue a_Q (transition):")
@printf("  slopes:    [%.4f, %.4f, %.4f]\n", par_true.a_Q[2,:]...)
@printf("  intercepts:[%.4f, %.4f, %.4f]\n", par_true.a_Q[1,:]...)
println("\nTrue b_coef (spline):")
@printf("  β_1 = %.2f (first Hermite coeff)\n", b_true[1, 2])
@printf("  γ = %.3f (each l, first Hermite coeff)\n", b_true[1, 3])

# Generate data from SMOOTH model
for N in [500]
    println("\n"*"-"^60)
    @printf("  N = %d\n", N)
    println("-"^60)

    println("Generating data from smooth model...")
    @time y, eta_true = generate_data_smooth(N, par_true, b_true, tau, sigma_y, K; seed=42)
    @printf("  η range: [%.3f, %.3f]\n", extrema(eta_true)...)
    @printf("  y range: [%.3f, %.3f]\n", extrema(y)...)

    cfg = Config(N, T, K, L, tau, sigma_y, 1, 1, 1, fill(0.05, T))

    # Evaluate likelihood at truth (sanity check)
    nll_truth = smooth_neg_loglik(par_true, b_true, y, cfg; G=80)
    @printf("\nneg-ll at TRUTH = %.4f\n", nll_truth)

    # Estimate by full smooth ML (a_Q and b_coef both free)
    println("\nRunning smooth ML (all params)...")
    par0 = init_params(y, cfg)
    eta_all = zeros(N, T, 20)
    for m in 1:20; eta_all[:, :, m] .= 0.6 .* y; end
    cfg_em = Config(N, T, K, L, tau, sigma_y, 5, 100, 20, fill(0.05, T))
    for _ in 1:5
        e_step!(eta_all, y, par0, cfg_em)
        m_step_qr!(par0, eta_all, y, cfg_em)
    end

    t = @elapsed par_sml, b_opt = estimate_smooth_ml(y, cfg, par0; G=80,
                                                    maxiter=100, verbose=true)

    @printf("\n==== RESULTS (N=%d) ====\n", N)
    @printf("True slopes:    [%.4f, %.4f, %.4f]\n", par_true.a_Q[2,:]...)
    @printf("QR slopes:      [%.4f, %.4f, %.4f]\n", par0.a_Q[2,:]...)
    @printf("SmoothML slopes:[%.4f, %.4f, %.4f]\n", par_sml.a_Q[2,:]...)

    @printf("\nTrue intercepts:    [%.4f, %.4f, %.4f]\n", par_true.a_Q[1,:]...)
    @printf("QR intercepts:      [%.4f, %.4f, %.4f]\n", par0.a_Q[1,:]...)
    @printf("SmoothML intercepts:[%.4f, %.4f, %.4f]\n", par_sml.a_Q[1,:]...)

    @printf("\nTrue b_coef column 2 (β_1): %s\n", round.(b_true[:,2], digits=4))
    @printf("Est  b_coef column 2 (β_1): %s\n", round.(b_opt[:,2], digits=4))
    @printf("True b_coef column 3 (γ_1): %s\n", round.(b_true[:,3], digits=4))
    @printf("Est  b_coef column 3 (γ_1): %s\n", round.(b_opt[:,3], digits=4))

    nll_est = smooth_neg_loglik(par_sml, b_opt, y, cfg; G=80)
    @printf("\nneg-ll at truth:    %.6f\n", nll_truth)
    @printf("neg-ll at estimate: %.6f  (diff: %+.6f)\n", nll_est, nll_est-nll_truth)
    @printf("Time: %.1fs\n", t)
end

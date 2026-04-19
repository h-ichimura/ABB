#=
test_mle_warm_debug.jl — Why doesn't MLE warm move from QR?

Evaluate: neg-ll at truth, at QR, gradient at QR, one step from QR toward truth.
=#

include("logistic_direct.jl")
using Printf

K = 2; σy = 1.0; N = 500
par_true = make_true_direct()
y, η = generate_data_direct(N, par_true, K, σy; seed=42)

# 1. neg-ll at truth
nll_true = direct_neg_loglik(par_true, y, K, σy; G=201)
@printf("neg-ll at truth: %.6f\n", nll_true)

# 2. QR estimation
par_qr0 = copy_params(par_true)
par_qr0.a_Q[2, :] .= 0.5
par_qr = estimate_qr_direct(y, K, σy, par_qr0; S=30, M=20, n_draws=100)
nll_qr = direct_neg_loglik(par_qr, y, K, σy; G=201)
@printf("neg-ll at QR:    %.6f\n", nll_qr)
@printf("QR slopes:  [%.4f, %.4f, %.4f]\n", par_qr.a_Q[2,:]...)
@printf("QR intcpts: [%.4f, %.4f, %.4f]\n", par_qr.a_Q[1,:]...)
flush(stdout)

# 3. Gradient at QR
v_qr = pack_direct(par_qr)
g_qr = zeros(length(v_qr))
nll_qr2, g_qr = negll_and_grad(v_qr, y, K, σy; G=201)
@printf("\nneg-ll at QR (from grad fn): %.6f\n", nll_qr2)
@printf("Gradient norm at QR: %.4e\n", norm(g_qr))
@printf("Gradient components (first 9 = a_Q):\n")
labels = ["a[1,1]","a[2,1]","a[3,1]","a[1,2]","a[2,2]","a[3,2]","a[1,3]","a[2,3]","a[3,3]"]
for i in 1:9
    @printf("  %s: v=%.4f  g=%+.4e\n", labels[i], v_qr[i], g_qr[i])
end
flush(stdout)

# 4. Is truth feasible from QR's perspective?
v_true = pack_direct(par_true)
@printf("\nTrue params packed: %s\n", round.(v_true[1:9], digits=4))
@printf("QR params packed:   %s\n", round.(v_qr[1:9], digits=4))

# 5. Check: does moving from QR toward truth improve neg-ll?
println("\nLine search QR → truth:")
for α in [0.0, 0.01, 0.05, 0.1, 0.2, 0.5, 1.0]
    v_trial = v_qr .+ α .* (v_true .- v_qr)
    par_trial = copy_params(par_true)
    unpack_direct!(par_trial, v_trial, K)
    nll_trial = direct_neg_loglik(par_trial, y, K, σy; G=201)
    cross = check_crossing(par_trial, K, σy)
    @printf("  α=%.2f: nll=%.6f  cross=%d\n", α, nll_trial, cross)
end
flush(stdout)

# 6. One MLE step from QR
println("\nMLE from QR (5 iterations):")
par_mle_w = copy_params(par_qr)
project_to_feasible!(par_mle_w, K, σy)
par_mle_w_fit, nll_mle_w = estimate_direct_ml_gs(y, K, σy, par_mle_w;
                                                   G=201, maxiter=5, verbose=true)
@printf("\nMLE warm slopes:  [%.4f, %.4f, %.4f]\n", par_mle_w_fit.a_Q[2,:]...)
@printf("MLE warm intcpts: [%.4f, %.4f, %.4f]\n", par_mle_w_fit.a_Q[1,:]...)

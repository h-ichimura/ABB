#=
test_mle_truth_start.jl — start the custom MLE at the truth and see if it
stays there. If the sample MLE is near truth, nll should decrease slightly
but slopes should remain close. If it drifts, the sample MLE is elsewhere.
=#

include("logistic_direct.jl")
using Printf

K = 2; σy = 1.0; N = 200; G = 101
par_true = make_true_direct()
y, _ = generate_data_direct(N, par_true, K, σy; seed=7)

nll_true = direct_neg_loglik(par_true, y, K, σy; G=G)
@printf("neg-ll @ truth  : %.6f\n", nll_true)

# Start AT truth
par0 = copy_params(par_true)
v0 = pack_direct(par0)

println("\n--- Starting AT truth ---")
_, g_truth = negll_and_grad(v0, y, K, σy; G=G)
@printf("|grad| @ truth = %.3e  (if sample MLE ≠ truth, this is nonzero)\n", norm(g_truth))

t = @elapsed par_mle, nll_mle = estimate_direct_ml_gs(
    y, K, σy, par0;
    G=G, maxiter=100, verbose=true, g_tol=1e-6, ls_tol=1e-6)

@printf("\nSample MLE from truth-start:\n")
@printf("  slopes:    [%.4f, %.4f, %.4f]\n", par_mle.a_Q[2,:]...)
@printf("  intcpts:   [%.4f, %.4f, %.4f]\n", par_mle.a_Q[1,:]...)
@printf("  quadratic: [%.4f, %.4f, %.4f]\n", par_mle.a_Q[3,:]...)
@printf("  nll = %.6f (truth %.6f, Δ=%+.4f), %.1fs\n",
        nll_mle, nll_true, nll_mle - nll_true, t)
@printf("  cross viol: %d\n", check_crossing(par_mle, K, σy))

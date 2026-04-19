#=
test_mle_gs.jl — small MLE test to verify estimate_direct_ml_gs.
Starts from equal-slope, zero-quadratic, ordered-intercept initial point.
=#

include("logistic_direct.jl")
using Printf, LinearAlgebra, Random

K = 2; σy = 1.0; N = 200; G = 101
par_true = make_true_direct()
y, η = generate_data_direct(N, par_true, K, σy; seed=7)
nll_true = direct_neg_loglik(par_true, y, K, σy; G=G)

println("="^70)
println("  MLE TEST: estimate_direct_ml_gs (N=$N, G=$G)")
println("="^70)
@printf("True slopes:   [%.4f, %.4f, %.4f]\n", par_true.a_Q[2,:]...)
@printf("True intcpts:  [%.4f, %.4f, %.4f]\n", par_true.a_Q[1,:]...)
@printf("neg-ll@truth:  %.6f\n\n", nll_true)

# Equal coefs across quantiles except intercept (as user requested).
# Values chosen to be inside the non-crossing cone but away from truth on slopes.
par0 = copy_params(par_true)
par0.a_Q[1, :] = [-0.3028, 0.0, 0.3028]  # intercepts at truth (ordered, different)
par0.a_Q[2, :] .= 0.5                      # all slopes = 0.5 (truth = 0.8)
par0.a_Q[3, :] .= 0.0                      # all quadratic = 0 (truth = 0)
par0.a_init .= [-0.6057, 0.0, 0.6057]
par0.a_eps  .= [-0.1817, 0.0, 0.1817]
v0 = pack_direct(par0)

@printf("Start v0 feasible? %s\n", is_feasible(v0, K, σy, -8.0, 8.0))
@printf("Start intercepts: [%.4f, %.4f, %.4f]\n", par0.a_Q[1,:]...)
@printf("Start slopes:     [%.4f, %.4f, %.4f]\n", par0.a_Q[2,:]...)
@printf("Start quadratic:  [%.4f, %.4f, %.4f]\n", par0.a_Q[3,:]...)
@printf("Start a_init:     [%.4f, %.4f, %.4f]\n", par0.a_init...)
@printf("Start a_eps:      [%.4f, %.4f, %.4f]\n", par0.a_eps...)
nll_start = direct_neg_loglik(par0, y, K, σy; G=G)
@printf("neg-ll@start:     %.6f (diff vs truth: %+.4f)\n\n", nll_start,
        nll_start - nll_true)

# ------ Custom LBFGS + golden section ------
println("--- estimate_direct_ml_gs (custom LBFGS + golden section + feasibility) ---")
t_gs = @elapsed par_gs, nll_gs = estimate_direct_ml_gs(
    y, K, σy, par0;
    G=G, maxiter=50, verbose=true, grid_min=-8.0, grid_max=8.0,
    g_tol=1e-5, ls_tol=1e-5)

cross_gs = check_crossing(par_gs, K, σy)
@printf("\n[GS]  RESULT:\n")
@printf("  slopes:    [%.4f, %.4f, %.4f]  (true %.4f)\n",
        par_gs.a_Q[2,:]..., par_true.a_Q[2,1])
@printf("  intcpts:   [%.4f, %.4f, %.4f]  (true [%.4f, %.4f, %.4f])\n",
        par_gs.a_Q[1,:]..., par_true.a_Q[1,:]...)
@printf("  quadratic: [%.4f, %.4f, %.4f]\n", par_gs.a_Q[3,:]...)
@printf("  a_init:    [%.4f, %.4f, %.4f]  (true [%.4f, %.4f, %.4f])\n",
        par_gs.a_init..., par_true.a_init...)
@printf("  a_eps:     [%.4f, %.4f, %.4f]  (true [%.4f, %.4f, %.4f])\n",
        par_gs.a_eps..., par_true.a_eps...)
@printf("  nll = %.6f  (truth %.6f, diff %+.4f)  time %.1fs\n",
        nll_gs, nll_true, nll_gs - nll_true, t_gs)
@printf("  crossing violations: %d\n", cross_gs)

# ------ Reference: Optim LBFGS with same analytical gradient ------
println("\n--- estimate_direct_ml (Optim LBFGS, analytical grad, no feasibility ctrl) ---")
t_optim = @elapsed par_opt, nll_opt = estimate_direct_ml(
    y, K, σy, par0; G=G, maxiter=50, verbose=false)

cross_opt = check_crossing(par_opt, K, σy)
@printf("\n[OPT] RESULT:\n")
@printf("  slopes:    [%.4f, %.4f, %.4f]\n", par_opt.a_Q[2,:]...)
@printf("  intcpts:   [%.4f, %.4f, %.4f]\n", par_opt.a_Q[1,:]...)
@printf("  quadratic: [%.4f, %.4f, %.4f]\n", par_opt.a_Q[3,:]...)
@printf("  nll = %.6f  (truth %.6f, diff %+.4f)  time %.1fs\n",
        nll_opt, nll_true, nll_opt - nll_true, t_optim)
@printf("  crossing violations: %d\n", cross_opt)

println("\nDone.")

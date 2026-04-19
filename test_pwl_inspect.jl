#=
test_pwl_inspect.jl — Inspect PWL-z MLE parameters to see if degeneracy occurs
=#

include("pwl_abb.jl")
using Printf

K = 2; σy = 1.0
p_true = make_true_pwl_logistic(; ρ=0.8, σ_v=0.5, σ_eps=0.3, σ_η1=1.0, K=K)
N = 800
y, η = generate_data_pwl(N, p_true, K, σy; seed=42)

println("="^70)
println("  INSPECT PWL-z MLE behavior")
println("="^70)
@printf("N=%d, η range [%.3f, %.3f], y range [%.3f, %.3f]\n",
        N, extrema(η)..., extrema(y)...)

nll_true = pwl_neg_loglik(p_true, y, K, σy; G=120)
@printf("neg-ll at truth: %.6f\n\n", nll_true)

# Print true params
function show_pwl(p::PWLParams, label::String)
    println("--- $label ---")
    @printf("  a_Q slopes:  [%.4f, %.4f, %.4f]\n", p.a_Q[2, :]...)
    @printf("  a_Q intcpts: [%.4f, %.4f, %.4f]\n", p.a_Q[1, :]...)
    @printf("  a_Q quad:    [%.4f, %.4f, %.4f]\n", p.a_Q[3, :]...)
    @printf("  b_L (log α_L): [%.4f, %.4f, %.4f] → α_L(0)=%.4f\n",
            p.b_L..., exp(p.b_L[1]))
    @printf("  b_R (log α_R): [%.4f, %.4f, %.4f] → α_R(0)=%.4f\n",
            p.b_R..., exp(p.b_R[1]))
    @printf("  q_init: [%.4f, %.4f, %.4f], αL=%.4f, αR=%.4f\n",
            p.q_init..., p.αL_init, p.αR_init)
    @printf("  q_eps:  [%.4f, %.4f, %.4f], αL=%.4f, αR=%.4f\n",
            p.q_eps..., p.αL_eps, p.αR_eps)

    # Show conditional quantile gaps at several η values
    println("  Conditional quantile gaps:")
    for η_val in [-1.0, 0.0, 1.0]
        q = cond_q(η_val, p.a_Q, K, σy)
        αL, αR = cond_α_tails(η_val, p.b_L, p.b_R, K, σy)
        gap1 = q[2] - q[1]; gap2 = q[3] - q[2]
        α1 = gap1 > 1e-10 ? log(3)/gap1 : Inf
        α2 = gap2 > 1e-10 ? log(3)/gap2 : Inf
        @printf("    η=%.1f: q=(%.4f, %.4f, %.4f) gaps=(%.4f, %.4f) int_slopes=(%.1f, %.1f) tail_slopes=(%.1f, %.1f)\n",
                η_val, q..., gap1, gap2, α1, α2, αL, αR)
    end
    # Show eps and init gaps
    gap1_init = p.q_init[2] - p.q_init[1]; gap2_init = p.q_init[3] - p.q_init[2]
    gap1_eps = p.q_eps[2] - p.q_eps[1]; gap2_eps = p.q_eps[3] - p.q_eps[2]
    @printf("  Init gaps: (%.4f, %.4f), Eps gaps: (%.4f, %.4f)\n",
            gap1_init, gap2_init, gap1_eps, gap2_eps)
end

show_pwl(p_true, "TRUE")

# Run MLE for a few iterations and inspect
println()
for maxiter in [5, 15, 30]
    p0 = copy_pwl(p_true)
    p0.a_Q[2, :] .= 0.5
    p0.b_L[1] = log(2.0); p0.b_R[1] = log(2.0)

    p_fit, res = estimate_pwl_ml(y, K, σy, p0; G=100, maxiter=maxiter, verbose=false)
    nll = pwl_neg_loglik(p_fit, y, K, σy; G=120)
    @printf("\nAfter %d LBFGS iterations: neg-ll = %.4f (truth %.4f)\n", maxiter, nll, nll_true)
    show_pwl(p_fit, "FIT (iter=$maxiter)")
end

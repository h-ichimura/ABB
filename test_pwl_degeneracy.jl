#=
test_pwl_degeneracy.jl — Can we create unbounded likelihood by
simultaneously concentrating ε and transition densities?
=#

include("pwl_abb.jl")
using Printf

K = 2; σy = 1.0
p_true = make_true_pwl_logistic(; ρ=0.8, σ_v=0.5, σ_eps=0.3, σ_η1=1.0, K=K)
N = 800
y, η = generate_data_pwl(N, p_true, K, σy; seed=42)

nll_true = pwl_neg_loglik(p_true, y, K, σy; G=120)
@printf("neg-ll at truth: %.4f\n", nll_true)

# Test: progressively squeeze ε distribution while keeping other params at truth
println("\n--- Squeezing ε only (transition at truth) ---")
for ε_scale in [1.0, 0.5, 0.2, 0.1, 0.05, 0.01]
    p = copy_pwl(p_true)
    p.q_eps .*= ε_scale
    p.αL_eps /= ε_scale; p.αR_eps /= ε_scale
    nll = pwl_neg_loglik(p, y, K, σy; G=120)
    @printf("  ε_scale=%.3f: q_eps=[%.4f, 0, %.4f], α_eps=%.1f, neg-ll=%.4f\n",
            ε_scale, p.q_eps[1], p.q_eps[3], p.αL_eps, nll)
end

# Test: squeeze both ε AND transition (shrink gaps)
println("\n--- Squeezing both ε and transition ---")
for scale in [1.0, 0.5, 0.2, 0.1, 0.05]
    p = copy_pwl(p_true)
    # Shrink ε gaps
    p.q_eps .*= scale
    p.αL_eps /= scale; p.αR_eps /= scale
    # Shrink transition gaps (bring quantile intercepts closer to 0)
    p.a_Q[1, :] .*= scale  # intercepts closer to 0
    # Increase tail slopes to match
    p.b_L[1] = log(exp(p_true.b_L[1]) / scale)
    p.b_R[1] = log(exp(p_true.b_R[1]) / scale)
    nll = pwl_neg_loglik(p, y, K, σy; G=120)
    @printf("  scale=%.3f: trans_intcpts=[%.4f, 0, %.4f], α_tail=%.1f, ε_gap=%.4f, neg-ll=%.4f\n",
            scale, p.a_Q[1, 1], p.a_Q[1, 3], exp(p.b_L[1]), p.q_eps[3]-p.q_eps[2], nll)
end

# Test: squeeze transition only (ε at truth)
println("\n--- Squeezing transition only (ε at truth) ---")
for scale in [1.0, 0.5, 0.2, 0.1, 0.05]
    p = copy_pwl(p_true)
    p.a_Q[1, :] .*= scale
    p.b_L[1] = log(exp(p_true.b_L[1]) / scale)
    p.b_R[1] = log(exp(p_true.b_R[1]) / scale)
    nll = pwl_neg_loglik(p, y, K, σy; G=120)
    @printf("  scale=%.3f: trans_intcpts=[%.4f, 0, %.4f], α_tail=%.1f, neg-ll=%.4f\n",
            scale, p.a_Q[1, 1], p.a_Q[1, 3], exp(p.b_L[1]), nll)
end

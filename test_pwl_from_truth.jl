#=
test_pwl_from_truth.jl — Start PWL-z MLE from truth & from perturbed points
=#

include("pwl_abb.jl")
using Printf

K = 2; σy = 1.0
p_true = make_true_pwl_logistic(; ρ=0.8, σ_v=0.5, σ_eps=0.3, σ_η1=1.0, K=K)
N = 500
y, η = generate_data_pwl(N, p_true, K, σy; seed=42)

nll_true = pwl_neg_loglik(p_true, y, K, σy; G=100)
@printf("neg-ll at truth: %.6f\n\n", nll_true)

# Test 1: LBFGS from truth
println("="^60)
println("  TEST 1: LBFGS from TRUTH")
println("="^60)
p1 = copy_pwl(p_true)
t1 = @elapsed p1_fit, res1 = estimate_pwl_ml(y, K, σy, p1; G=100, maxiter=50, verbose=true)
nll1 = pwl_neg_loglik(p1_fit, y, K, σy; G=100)
@printf("\nneg-ll: %.6f (diff: %+.6f)\n", nll1, nll1 - nll_true)
@printf("slopes: [%.4f, %.4f, %.4f]\n", p1_fit.a_Q[2,:]...)
@printf("time: %.1fs\n", t1)

# Test 2: LBFGS from 5% perturbed truth
println("\n","="^60)
println("  TEST 2: LBFGS from 5% perturbed truth")
println("="^60)
p2 = copy_pwl(p_true)
p2.a_Q .*= 1.05; p2.b_L .*= 1.05; p2.b_R .*= 1.05
t2 = @elapsed p2_fit, res2 = estimate_pwl_ml(y, K, σy, p2; G=100, maxiter=50, verbose=true)
nll2 = pwl_neg_loglik(p2_fit, y, K, σy; G=100)
@printf("\nneg-ll: %.6f (diff: %+.6f)\n", nll2, nll2 - nll_true)
@printf("slopes: [%.4f, %.4f, %.4f]\n", p2_fit.a_Q[2,:]...)
@printf("time: %.1fs\n", t2)

# Test 3: Coordinate descent from 20% perturbed truth
println("\n","="^60)
println("  TEST 3: Coordinate descent from 20% perturbed")
println("="^60)
p3 = copy_pwl(p_true)
p3.a_Q .*= 1.20; p3.b_L .*= 0.80; p3.b_R .*= 0.80

v3 = pack_pwl(p3, K)
p3w = copy_pwl(p3)
@printf("Start neg-ll: %.6f\n", pwl_neg_loglik(p3, y, K, σy; G=100))

for cyc in 1:20
    max_step = 0.0
    for i in eachindex(v3)
        cur = v3[i]
        lo = cur - 0.3; hi = cur + 0.3

        f1d(vi) = (v3[i] = vi; p3w = unpack_pwl(v3, K); pwl_neg_loglik(p3w, y, K, σy; G=100))

        res = optimize(f1d, lo, hi)
        new_val = Optim.minimizer(res)
        max_step = max(max_step, abs(new_val - cur))
        v3[i] = new_val
    end
    p3w = unpack_pwl(v3, K)
    nll_cur = pwl_neg_loglik(p3w, y, K, σy; G=100)
    (cyc <= 5 || cyc % 5 == 0) && @printf("  CD cycle %2d: neg-ll=%.6f  step=%.2e\n", cyc, nll_cur, max_step)
    max_step < 1e-5 && break
end

p3_fit = unpack_pwl(v3, K)
nll3 = pwl_neg_loglik(p3_fit, y, K, σy; G=100)
@printf("\nneg-ll: %.6f (diff: %+.6f)\n", nll3, nll3 - nll_true)
@printf("slopes: [%.4f, %.4f, %.4f]\n", p3_fit.a_Q[2,:]...)

#=
test_analytical_grad.jl — verify the analytical gradient against central
finite differences, and benchmark the rewritten negll_and_grad.
=#

include("logistic_direct.jl")
using Printf, LinearAlgebra, Random

K = 2; σy = 1.0; N = 100; G = 101
par_true = make_true_direct()

println("="^70)
println("  Analytical-gradient correctness check (vs central differences)")
println("="^70)
y, _ = generate_data_direct(N, par_true, K, σy; seed=0)

# Slightly perturb away from truth so derivatives are more informative.
# Small perturbation so constraints stay slack.
v0 = pack_direct(par_true)
rng = MersenneTwister(0)
v0 .+= 0.01 .* randn(rng, length(v0))
if !is_feasible(v0, K, σy, -8.0, 8.0)
    # shrink until feasible
    while !is_feasible(v0, K, σy, -8.0, 8.0)
        v0 = pack_direct(par_true) .+ 0.5 .* (v0 .- pack_direct(par_true))
    end
end
@assert is_feasible(v0, K, σy, -8.0, 8.0)

nll, g_an = negll_and_grad(v0, y, K, σy; G=G)
@printf("nll at v0 = %.8f\n", nll)

# Central differences
step = 1e-5
g_num = similar(g_an)
for j in eachindex(v0)
    vp = copy(v0); vp[j] += step
    vm = copy(v0); vm[j] -= step
    fp = direct_neg_loglik(
        let p=copy_params(par_true); unpack_direct!(p, vp, K); p end,
        y, K, σy; G=G)
    fm = direct_neg_loglik(
        let p=copy_params(par_true); unpack_direct!(p, vm, K); p end,
        y, K, σy; G=G)
    g_num[j] = (fp - fm) / (2*step)
end

println("\n   j  analytical         numerical          abs diff       rel diff")
println("  ", "-"^70)
for j in eachindex(v0)
    d = g_an[j] - g_num[j]
    r = abs(d) / max(abs(g_num[j]), 1e-10)
    flag = abs(d) < 1e-4 ? "  ✓" : "  ✗"
    @printf("  %2d  %+14.8e  %+14.8e  %10.2e  %10.2e%s\n",
            j, g_an[j], g_num[j], abs(d), r, flag)
end
max_abs = maximum(abs.(g_an .- g_num))
@printf("\nmax |analytical - numerical| = %.3e\n", max_abs)
if max_abs < 1e-4
    println("  PASS: gradients agree to tolerance")
else
    println("  FAIL: analytical gradient has a bug")
end

println("\n", "="^70)
println("  Benchmark: negll_and_grad at N=$N, G=$G")
println("="^70)
# Warm-up compile
_ = negll_and_grad(v0, y, K, σy; G=G)
# Timing
t_grad = @elapsed for _ in 1:5; negll_and_grad(v0, y, K, σy; G=G); end
t_eval = @elapsed for _ in 1:5; direct_neg_loglik(par_true, y, K, σy; G=G); end
@printf("  negll_and_grad:   %.3fs / call (avg of 5)\n", t_grad/5)
@printf("  direct_neg_loglik: %.3fs / call (avg of 5)\n", t_eval/5)

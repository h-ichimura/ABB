#=
test_init_identification.jl — Explicitly compute f(y) as a function of f_init parameters
and show the likelihood sensitivity.

For a single observation y = (y_1, y_2, y_3):
  L(θ) = ∫∫∫ f_init(η_1) f_ε(y_1-η_1) f(η_2|η_1) f_ε(y_2-η_2) f(η_3|η_2) f_ε(y_3-η_3) dη

The integral over η_1 can be computed piecewise (since f_init is piecewise uniform
with exponential tails). This gives f(y) as a function of:
  - f_init parameters (a_init_1, a_init_2, a_init_3, b1_init, bL_init)
  - other parameters (transition, ε)

We vary ONE f_init parameter at a time and plot L(θ).
If the likelihood has non-zero curvature in that parameter, it's identified.
If the likelihood is flat, it's not.
=#

include("exact_ml.jl")
using Printf

K = 2; L = 3; sigma_y = 1.0; T = 3
tau = [0.25, 0.50, 0.75]
N = 300

par_true = make_true_params_linear(tau=tau, sigma_y=sigma_y, K=K)
y, eta_true = generate_data_abb(N, par_true, tau, sigma_y, K; seed=42)
cfg = Config(N, T, K, L, tau, sigma_y, 1, 1, 1, fill(0.05, T))

println("="^70)
println("  IDENTIFICATION OF f_init PARAMETERS")
println("="^70)
println("\nTrue init quantiles: ", round.(par_true.a_init, digits=4))
println("True init tails: b1=$(par_true.b1_init), bL=$(par_true.bL_init)")

ll_true = -N * exact_neg_loglik(par_true, y, cfg; G=100)
@printf("\nLog-lik at truth: %.4f\n", ll_true)

# ================================================================
#  Vary a_init[1] (τ=0.25 quantile of η_1)
# ================================================================
println("\n"*"-"^60)
println("Vary a_init[1] (τ=0.25 quantile)")
println("-"^60)
println("  value    log-lik     diff from truth")

for val in [-1.5, -1.0, -0.9, -0.8, -0.7, -0.6742, -0.6, -0.5, -0.3, 0.0]
    par = copy_params(par_true)
    par.a_init[1] = val
    # Ensure ordering
    if par.a_init[2] < val; par.a_init[2] = val + 0.01; end
    ll = -N * exact_neg_loglik(par, y, cfg; G=100)
    @printf("  %7.4f  %10.4f  %+10.4f\n", val, ll, ll - ll_true)
end

# ================================================================
#  Vary a_init[3] (τ=0.75 quantile of η_1)
# ================================================================
println("\n"*"-"^60)
println("Vary a_init[3] (τ=0.75 quantile)")
println("-"^60)
println("  value    log-lik     diff from truth")

for val in [0.0, 0.3, 0.5, 0.6, 0.6742, 0.7, 0.8, 0.9, 1.0, 1.5]
    par = copy_params(par_true)
    par.a_init[3] = val
    if par.a_init[2] > val; par.a_init[2] = val - 0.01; end
    ll = -N * exact_neg_loglik(par, y, cfg; G=100)
    @printf("  %7.4f  %10.4f  %+10.4f\n", val, ll, ll - ll_true)
end

# ================================================================
#  Vary b1_init (left tail rate)
# ================================================================
println("\n"*"-"^60)
println("Vary b1_init (left tail rate; true = 1.0)")
println("-"^60)
println("  value    log-lik     diff from truth")

for val in [0.3, 0.5, 0.8, 1.0, 1.2, 1.5, 2.0, 3.0, 5.0]
    par = copy_params(par_true)
    par.b1_init = val
    ll = -N * exact_neg_loglik(par, y, cfg; G=100)
    @printf("  %7.4f  %10.4f  %+10.4f\n", val, ll, ll - ll_true)
end

# ================================================================
#  Vary bL_init (right tail rate)
# ================================================================
println("\n"*"-"^60)
println("Vary bL_init (right tail rate; true = 1.0)")
println("-"^60)
println("  value    log-lik     diff from truth")

for val in [0.3, 0.5, 0.8, 1.0, 1.2, 1.5, 2.0, 3.0, 5.0]
    par = copy_params(par_true)
    par.bL_init = val
    ll = -N * exact_neg_loglik(par, y, cfg; G=100)
    @printf("  %7.4f  %10.4f  %+10.4f\n", val, ll, ll - ll_true)
end

# ================================================================
#  For comparison: vary slope (known to be well-identified)
# ================================================================
println("\n"*"-"^60)
println("COMPARISON: vary slope(τ=0.50) (true = 0.80)")
println("-"^60)
println("  value    log-lik     diff from truth")

for val in [0.5, 0.6, 0.7, 0.75, 0.78, 0.80, 0.82, 0.85, 0.9, 1.0, 1.2]
    par = copy_params(par_true)
    par.a_Q[2, 2] = val
    ll = -N * exact_neg_loglik(par, y, cfg; G=100)
    @printf("  %7.4f  %10.4f  %+10.4f\n", val, ll, ll - ll_true)
end

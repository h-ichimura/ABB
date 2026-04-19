#=
test_exact_sanity.jl — Sanity tests for exact_ml.jl

1. Does the likelihood agree between independent implementations?
2. Is the likelihood gradient zero at truth (for large N)?
3. Does the forward filter integrate correctly?
=#

include("exact_ml.jl")
using Printf

K=2; L=3; sigma_y=1.0; T=3
tau = [0.25, 0.50, 0.75]
par_true = make_true_params_linear(tau=tau, sigma_y=sigma_y, K=K)

# ================================================================
#  TEST 1: Direct 3D integration vs forward filter
# ================================================================
println("="^60)
println("  TEST 1: Forward filter vs direct 3D integration")
println("="^60)
println("Compute L(θ₀) = ∫∫∫ f_init(η₁) Π f_ε(y_t-η_t) Π f(η_t|η_{t-1}) dη")
println("for a SINGLE observation, two ways:")
println("  (a) Forward filter as in exact_ml.jl")
println("  (b) Direct triple integral on fine grid")

# Generate one observation
N = 1
y_one, eta_true = generate_data_abb(N, par_true, tau, sigma_y, K; seed=42)
println("y = ", round.(y_one[1,:], digits=4))
println("true η = ", round.(eta_true[1,:], digits=4))

# Method (a): forward filter
cfg = Config(N, T, K, L, tau, sigma_y, 1, 1, 1, fill(0.05, T))
ll_ff = -N * exact_neg_loglik(par_true, y_one, cfg; G=200)
@printf("(a) Forward filter: log L = %.6f\n", ll_ff)

# Method (b): direct triple integration
function direct_loglik(y, par, cfg; G=80, lo=-4.0, hi=4.0)
    grid = collect(range(lo, hi, length=G))
    dgrid = (hi-lo)/(G-1)
    q_buf = zeros(cfg.L)
    total_L = 0.0
    for g1 in 1:G, g2 in 1:G, g3 in 1:G
        η1, η2, η3 = grid[g1], grid[g2], grid[g3]
        # Density of (η1, η2, η3)
        f_η = exp(pw_logdens(η1, par.a_init, cfg.tau, par.b1_init, par.bL_init))
        transition_quantiles!(q_buf, η1, par.a_Q, cfg.K, cfg.sigma_y); sort!(q_buf)
        f_η *= exp(pw_logdens(η2, q_buf, cfg.tau, par.b1_Q, par.bL_Q))
        transition_quantiles!(q_buf, η2, par.a_Q, cfg.K, cfg.sigma_y); sort!(q_buf)
        f_η *= exp(pw_logdens(η3, q_buf, cfg.tau, par.b1_Q, par.bL_Q))
        # Likelihood of y given η
        f_y = exp(pw_logdens(y[1]-η1, par.a_eps, cfg.tau, par.b1_eps, par.bL_eps)) *
              exp(pw_logdens(y[2]-η2, par.a_eps, cfg.tau, par.b1_eps, par.bL_eps)) *
              exp(pw_logdens(y[3]-η3, par.a_eps, cfg.tau, par.b1_eps, par.bL_eps))
        total_L += f_η * f_y * dgrid^3
    end
    log(total_L)
end

ll_direct = direct_loglik(view(y_one,1,:), par_true, cfg; G=80)
@printf("(b) Direct 3D integral (G=80): log L = %.6f\n", ll_direct)
@printf("Difference: %.6f\n", ll_ff - ll_direct)

# ================================================================
#  TEST 2: Does the likelihood penalize wrong parameters?
#  Perturb ONE parameter and check log L decreases
# ================================================================
println("\n"*"="^60)
println("  TEST 2: Likelihood penalty for wrong parameters (N=500)")
println("="^60)

N = 500
y, eta_true = generate_data_abb(N, par_true, tau, sigma_y, K; seed=42)
cfg = Config(N, T, K, L, tau, sigma_y, 1, 1, 1, fill(0.05, T))

ll_truth = -N * exact_neg_loglik(par_true, y, cfg; G=100)
@printf("log L at truth: %.4f\n", ll_truth)

# Perturb slope at τ=0.50
for slope in [0.70, 0.75, 0.79, 0.80, 0.81, 0.85, 0.90]
    par = copy_params(par_true)
    par.a_Q[2, 2] = slope
    ll = -N * exact_neg_loglik(par, y, cfg; G=100)
    @printf("  slope(τ=0.50) = %.2f:  log L = %.4f  Δ = %+.4f\n",
            slope, ll, ll - ll_truth)
end

# Perturb intercept at τ=0.25
println()
for intcpt in [-0.45, -0.40, -0.35, -0.337, -0.30, -0.25, -0.20]
    par = copy_params(par_true)
    par.a_Q[1, 1] = intcpt
    ll = -N * exact_neg_loglik(par, y, cfg; G=100)
    @printf("  intcpt(τ=0.25) = %+.3f: log L = %.4f  Δ = %+.4f\n",
            intcpt, ll, ll - ll_truth)
end

# ================================================================
#  TEST 3: At optimum found by exact_ml, is the gradient small?
# ================================================================
println("\n"*"="^60)
println("  TEST 3: Check gradient at exact ML optimum")
println("="^60)

# Run exact ML from QR warm start
N = 200
y, eta_true = generate_data_abb(N, par_true, tau, sigma_y, K; seed=42)
cfg_em = Config(N, T, K, L, tau, sigma_y, 5, 100, 20, fill(0.05, T))
par0 = init_params(y, cfg_em)
eta_all = zeros(N, T, 20)
for m in 1:20; eta_all[:,:,m] .= 0.6 .* y; end
for _ in 1:5
    e_step!(eta_all, y, par0, cfg_em)
    m_step_qr!(par0, eta_all, y, cfg_em)
end

cfg = Config(N, T, K, L, tau, sigma_y, 1, 1, 1, fill(0.05, T))
par_ml = estimate_exact_ml(y, cfg, par0; G=80, maxiter=200, verbose=false)

# Numerical gradient at truth and at ML optimum
function num_grad(par, obj_fn, h=1e-4)
    v = params_to_vec(par)
    g = zeros(length(v))
    par_work = copy_params(par)
    for i in eachindex(v)
        v_p = copy(v); v_p[i] += h
        vec_to_params!(par_work, v_p, K, L)
        f_p = obj_fn(par_work)
        v_m = copy(v); v_m[i] -= h
        vec_to_params!(par_work, v_m, K, L)
        f_m = obj_fn(par_work)
        g[i] = (f_p - f_m) / (2h)
    end
    g
end

obj(par) = exact_neg_loglik(par, y, cfg; G=80)

g_truth = num_grad(par_true, obj)
g_ml = num_grad(par_ml, obj)
@printf("N=%d, grad components:\n", N)
@printf("%-22s %12s %12s\n", "Param", "at truth", "at ML")
np_aQ = (K+1)*L
labels = String[]
for l in 1:L, k in 1:K+1
    push!(labels, "a_Q[$k,$l]")
end
push!(labels, "log b1_Q", "log bL_Q")
for l in 1:L; push!(labels, "a_init[$l]"); end
push!(labels, "log b1_init", "log bL_init")
for l in 1:L; push!(labels, "a_eps[$l]"); end
push!(labels, "log b1_eps", "log bL_eps")

for i in eachindex(g_truth)
    @printf("%-22s %+12.4e %+12.4e\n", labels[i], g_truth[i], g_ml[i])
end
@printf("\n|grad at truth| = %.4f\n", sqrt(sum(g_truth.^2)))
@printf("|grad at ML|    = %.4f\n", sqrt(sum(g_ml.^2)))

@printf("\nAt truth:     slopes = [%.4f, %.4f, %.4f]\n", par_true.a_Q[2,:]...)
@printf("At ML:        slopes = [%.4f, %.4f, %.4f]\n", par_ml.a_Q[2,:]...)
@printf("neg-loglik at truth: %.6f\n", obj(par_true))
@printf("neg-loglik at ML:    %.6f\n", obj(par_ml))

#=
calibrate_logspline.jl — Fit logspline to ABB's piecewise-uniform transition

1. Generate large sample from ABB model (true η observed)
2. Fit logspline to (η_t, η_{t-1}) transition pairs
3. Save fitted logspline parameters as "true" params for comparison
=#

include("ABB_three_period.jl")
include("logspline.jl")

using Optim, Serialization

K = 2; L = 3; sigma_y = 1.0; T = 3
tau = [0.25, 0.50, 0.75]
knots_sp = [-1.0, -0.3, 0.0, 0.3, 1.0]
K_sp = length(knots_sp)
J = K  # Hermite order

# Step 1: Generate large sample from ABB model
println("Step 1: Generating data from ABB piecewise-uniform model...")
N_large = 1000
par_true = make_true_params_linear(tau=tau, sigma_y=sigma_y, K=K)
y_large, eta_large = generate_data_abb(N_large, par_true, tau, sigma_y, K; seed=42)

# Step 2: Stack transition pairs from true η
println("Step 2: Stacking transition pairs...")
n_pairs = N_large * (T - 1)
eta_t = Vector{Float64}(undef, n_pairs)
eta_lag = Vector{Float64}(undef, n_pairs)
global idx = 0
for t in 2:T, i in 1:N_large
    global idx += 1
    eta_t[idx] = eta_large[i, t]
    eta_lag[idx] = eta_large[i, t-1]
end
H_lag = hermite_basis(eta_lag, J, sigma_y)
@printf("  %d transition pairs\n", n_pairs)

# Step 3: Fit logspline by MLE
println("Step 3: Fitting logspline to transition pairs...")

# Initialize: small cubics, moderate linear term
a_init = zeros(K_sp + 2, J + 1)
a_init[2, 2] = 0.5 * sigma_y
for k in 1:K_sp
    a_init[k + 2, 1] = -0.2
end

function neg_ll(theta)
    logspline_neg_loglik(theta, eta_t, H_lag, knots_sp)
end

function neg_ll_grad!(g, theta)
    g .= logspline_neg_loglik_grad(theta, eta_t, H_lag, knots_sp)
end

theta0 = vec(copy(a_init))
@printf("  neg-ll at start: %.6f\n", neg_ll(theta0))

od = Optim.OnceDifferentiable(neg_ll, neg_ll_grad!, theta0)
res = optimize(od, theta0, LBFGS(),
               Optim.Options(iterations=200, g_tol=1e-6, show_trace=false))

a_fitted = reshape(Optim.minimizer(res), K_sp + 2, J + 1)
@printf("  neg-ll at MLE:   %.6f  (iters=%d)\n",
        Optim.minimum(res), Optim.iterations(res))

# Print fitted parameters
println("\nFitted logspline parameters:")
println("  β₀ (intercept): ", round.(a_fitted[1,:], digits=4))
println("  β₁ (linear):    ", round.(a_fitted[2,:], digits=4))
for k in 1:K_sp
    @printf("  γ_%d (knot %.1f):  %s\n", k, knots_sp[k],
            round.(a_fitted[k+2,:], digits=4))
end

# Verify: check density at a few points
println("\nVerification: logspline density vs ABB density at η_{t-1}=0")
ls = LogsplineTransition(a_fitted, knots_sp, sigma_y, J)
q_buf = zeros(L)
transition_quantiles!(q_buf, 0.0, par_true.a_Q, K, sigma_y)
@printf("  ABB quantile knots at η_{t-1}=0: [%.4f, %.4f, %.4f]\n", q_buf...)

for x in [-1.0, -0.5, -0.3, 0.0, 0.3, 0.5, 1.0]
    ls_dens = exp(logspline_transition_logdens(x, 0.0, ls))
    abb_dens = exp(pw_logdens(x, q_buf, tau, par_true.b1_Q, par_true.bL_Q))
    @printf("  x=%.1f: logspline=%.4f  ABB=%.4f  ratio=%.4f\n",
            x, ls_dens, abb_dens, ls_dens/abb_dens)
end

# Save
serialize("logspline_true_params.jls",
          (a_trans=a_fitted, knots_sp=knots_sp, par_true=par_true,
           sigma_y=sigma_y, K=K, J=J, tau=tau))
println("\nSaved logspline_true_params.jls")

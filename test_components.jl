#=
test_components.jl — Verify correctness of each component
=#

using LinearAlgebra, Statistics, Random, Printf, Optim

# Load functions (but not the run block at the end)
# We'll include everything up to the run block by parsing

# ---- Inline the needed functions ----

@inline function He(n::Int, x::Float64)
    n == 0 && return 1.0
    n == 1 && return x
    hm2, hm1 = 1.0, x
    for k in 2:n; hm2, hm1 = hm1, x*hm1 - (k-1)*hm2; end
    hm1
end

function hermite_basis(x::AbstractVector{Float64}, K::Int, sigma::Float64)
    n = length(x)
    H = Matrix{Float64}(undef, n, K+1)
    @inbounds for i in 1:n
        z = x[i]/sigma; H[i,1] = 1.0
        K >= 1 && (H[i,2] = z)
        for k in 2:K; H[i,k+1] = z*H[i,k] - (k-1)*H[i,k-1]; end
    end
    H
end

function pw_logdens(x::Float64, q::AbstractVector{Float64},
                    tau::Vector{Float64}, b1::Float64, bL::Float64)
    L = length(q)
    x <= q[1] && return log(max(tau[1]*b1, 1e-300)) + b1*(x - q[1])
    x > q[L]  && return log(max((1.0-tau[L])*bL, 1e-300)) - bL*(x - q[L])
    @inbounds for l in 1:L-1
        if x <= q[l+1]
            dq = q[l+1] - q[l]
            return dq > 1e-12 ? log(tau[l+1]-tau[l]) - log(dq) : -700.0
        end
    end
    -700.0
end

function pw_draw(rng::AbstractRNG, q::AbstractVector{Float64},
                 tau::Vector{Float64}, b1::Float64, bL::Float64)
    L = length(q); u = rand(rng)
    u < tau[1]  && return q[1] + log(u/tau[1])/b1
    u >= tau[L] && return q[L] - log((1.0-u)/(1.0-tau[L]))/bL
    @inbounds for l in 1:L-1
        u < tau[l+1] && return q[l] + (q[l+1]-q[l])*(u-tau[l])/(tau[l+1]-tau[l])
    end
    q[L]
end

function transition_quantiles!(q::Vector{Float64}, eta_lag::Float64,
                                a_Q::Matrix{Float64}, K::Int, sigma::Float64)
    z = eta_lag/sigma; L = size(a_Q,2)
    hv = Vector{Float64}(undef, K+1); hv[1]=1.0
    K>=1 && (hv[2]=z)
    for k in 2:K; hv[k+1] = z*hv[k]-(k-1)*hv[k-1]; end
    @inbounds for l in 1:L
        s=0.0; for k in 1:K+1; s += a_Q[k,l]*hv[k]; end; q[l]=s
    end
end

function norminv(p::Float64)
    p<=0.0 && return -Inf; p>=1.0 && return Inf
    t = p<0.5 ? sqrt(-2.0*log(p)) : sqrt(-2.0*log(1.0-p))
    x = t-(2.515517+0.802853t+0.010328t^2)/(1.0+1.432788t+0.189269t^2+0.001308t^3)
    p<0.5 ? -x : x
end

passed = 0
failed = 0
function check(name, condition)
    global passed, failed
    if condition
        println("  PASS: $name")
        passed += 1
    else
        println("  FAIL: $name")
        failed += 1
    end
end

# ================================================================
println("="^60)
println("TEST 1: Hermite polynomials")
println("="^60)
# He_0(x) = 1, He_1(x) = x, He_2(x) = x^2 - 1, He_3(x) = x^3 - 3x
check("He_0(2.5) = 1", He(0, 2.5) == 1.0)
check("He_1(2.5) = 2.5", He(1, 2.5) == 2.5)
check("He_2(2.5) = 2.5^2 - 1 = 5.25", abs(He(2, 2.5) - 5.25) < 1e-12)
check("He_3(2.5) = 2.5^3 - 3*2.5 = 8.125", abs(He(3, 2.5) - 8.125) < 1e-12)

# Check hermite_basis matches He
x = [0.5, 1.0, -1.5]
H = hermite_basis(x, 3, 1.0)
for (i,xi) in enumerate(x)
    for k in 0:3
        check("H[$i,$(k+1)] = He($k,$xi)", abs(H[i,k+1] - He(k,xi)) < 1e-12)
    end
end

# With sigma != 1
H2 = hermite_basis(x, 2, 2.0)
check("H_sigma: He_1(0.5/2) = 0.25", abs(H2[1,2] - 0.25) < 1e-12)

# ================================================================
println("\n","="^60)
println("TEST 2: Density integrates to 1")
println("="^60)

tau = [0.25, 0.50, 0.75]
q = [-0.5, 0.0, 0.8]
b1 = 2.0; bL = 3.0

# Numerical integration
dx = 0.0001
xgrid = collect(-10.0:dx:10.0)
integral = sum(exp(pw_logdens(x, q, tau, b1, bL)) for x in xgrid) * dx
@printf("  Integral of density = %.6f (should be 1.0)\n", integral)
check("density integrates to ~1", abs(integral - 1.0) < 0.01)

# Check density values at specific points
# Left tail: f = τ_1 * b1 * exp(b1*(x-q1))
x_left = -2.0
f_left = exp(pw_logdens(x_left, q, tau, b1, bL))
f_left_expected = tau[1] * b1 * exp(b1*(x_left - q[1]))
check("left tail density", abs(f_left - f_left_expected) < 1e-12)

# Interior segment 1: f = (τ_2 - τ_1) / (q_2 - q_1)
x_mid1 = -0.2
f_mid1 = exp(pw_logdens(x_mid1, q, tau, b1, bL))
f_mid1_expected = (tau[2] - tau[1]) / (q[2] - q[1])
check("interior seg 1 density", abs(f_mid1 - f_mid1_expected) < 1e-12)

# Interior segment 2: f = (τ_3 - τ_2) / (q_3 - q_2)
x_mid2 = 0.5
f_mid2 = exp(pw_logdens(x_mid2, q, tau, b1, bL))
f_mid2_expected = (tau[3] - tau[2]) / (q[3] - q[2])
check("interior seg 2 density", abs(f_mid2 - f_mid2_expected) < 1e-12)

# Right tail: f = (1-τ_L) * bL * exp(-bL*(x-qL))
x_right = 2.0
f_right = exp(pw_logdens(x_right, q, tau, b1, bL))
f_right_expected = (1 - tau[3]) * bL * exp(-bL*(x_right - q[3]))
check("right tail density", abs(f_right - f_right_expected) < 1e-12)

# ================================================================
println("\n","="^60)
println("TEST 3: pw_draw is consistent with pw_logdens")
println("="^60)

rng = MersenneTwister(123)
N_sample = 100000
samples = [pw_draw(rng, q, tau, b1, bL) for _ in 1:N_sample]

# Check empirical quantiles match q
emp_q = quantile(samples, tau)
@printf("  Empirical quantiles: [%.4f, %.4f, %.4f]\n", emp_q...)
@printf("  True quantiles:      [%.4f, %.4f, %.4f]\n", q...)
for l in 1:3
    check("quantile $l matches (tol 0.02)", abs(emp_q[l] - q[l]) < 0.02)
end

# Check empirical density against theoretical in interior
# Bin counts in segment 1: (q1, q2]
n_seg1 = count(q[1] .< samples .<= q[2])
f_seg1_emp = n_seg1 / (N_sample * (q[2]-q[1]))
f_seg1_true = (tau[2]-tau[1]) / (q[2]-q[1])
@printf("  Seg1 density: emp=%.4f true=%.4f\n", f_seg1_emp, f_seg1_true)
check("seg1 density matches (tol 0.02)", abs(f_seg1_emp - f_seg1_true) < 0.02)

# ================================================================
println("\n","="^60)
println("TEST 4: transition_quantiles! is correct")
println("="^60)

K = 2; sigma_y = 1.0
# a_Q[:, l] = [intercept, slope, quadratic]
a_Q = [[-0.3, 0.0, 0.3] [0.8, 0.8, 0.8] [0.0, 0.0, 0.0]]'
# Wait - a_Q should be (K+1) x L = 3 x 3
# a_Q[k+1, l] for k=0,1,2 and l=1,2,3
a_Q = zeros(3, 3)
a_Q[1, :] = [-0.337, 0.0, 0.337]   # intercepts
a_Q[2, :] = [0.8, 0.8, 0.8]         # slopes
a_Q[3, :] = [0.0, 0.0, 0.0]         # quadratic

q_buf = zeros(3)

# At eta_lag = 0: He_0=1, He_1=0, He_2=-1
# q_l = a_Q[1,l]*1 + a_Q[2,l]*0 + a_Q[3,l]*(-1) = a_Q[1,l]
transition_quantiles!(q_buf, 0.0, a_Q, K, sigma_y)
@printf("  q at eta_lag=0: [%.4f, %.4f, %.4f]\n", q_buf...)
check("q_1(0) = -0.337", abs(q_buf[1] - (-0.337)) < 1e-10)
check("q_2(0) = 0.0", abs(q_buf[2] - 0.0) < 1e-10)
check("q_3(0) = 0.337", abs(q_buf[3] - 0.337) < 1e-10)

# At eta_lag = 1: He_0=1, He_1=1, He_2=1^2-1=0
# q_l = a_Q[1,l] + a_Q[2,l]*1 + a_Q[3,l]*0 = a_Q[1,l] + 0.8
transition_quantiles!(q_buf, 1.0, a_Q, K, sigma_y)
@printf("  q at eta_lag=1: [%.4f, %.4f, %.4f]\n", q_buf...)
check("q_1(1) = -0.337+0.8 = 0.463", abs(q_buf[1] - 0.463) < 1e-10)
check("q_2(1) = 0+0.8 = 0.8", abs(q_buf[2] - 0.8) < 1e-10)
check("q_3(1) = 0.337+0.8 = 1.137", abs(q_buf[3] - 1.137) < 1e-10)

# Check monotonicity of knots (required for density to be valid)
for eta_lag in [-2.0, -1.0, 0.0, 1.0, 2.0]
    transition_quantiles!(q_buf, eta_lag, a_Q, K, sigma_y)
    mono = q_buf[1] < q_buf[2] < q_buf[3]
    check("knots monotone at eta_lag=$eta_lag", mono)
    if !mono
        @printf("    q = [%.4f, %.4f, %.4f]\n", q_buf...)
    end
end

# ================================================================
println("\n","="^60)
println("TEST 5: Transition density integrates to 1 for various eta_lag")
println("="^60)

b1_Q = 2.0; bL_Q = 2.0
dx = 0.0001
for eta_lag in [-1.0, 0.0, 1.0, 2.0]
    transition_quantiles!(q_buf, eta_lag, a_Q, K, sigma_y)
    xgrid = collect(-10.0:dx:10.0)
    integral = sum(exp(pw_logdens(x, q_buf, tau, b1_Q, bL_Q)) for x in xgrid) * dx
    @printf("  eta_lag=%.1f: integral=%.6f\n", eta_lag, integral)
    check("integral ~1 at eta_lag=$eta_lag", abs(integral - 1.0) < 0.01)
end

# ================================================================
println("\n","="^60)
println("TEST 6: MLE with observed eta (no latent variables)")
println("="^60)
println("  Generate eta from transition, observe it directly.")
println("  Maximize transition log-likelihood over a_Q[2,l] (slope).")
println("  True slope = 0.8. Check that MLE is near 0.8.")

# Generate N transition pairs (eta_lag, eta_t) from true model
N = 10000
rng = MersenneTwister(42)
eta_lag_data = randn(rng, N)  # draw eta_lag ~ N(0,1)
eta_t_data = zeros(N)
for i in 1:N
    transition_quantiles!(q_buf, eta_lag_data[i], a_Q, K, sigma_y)
    eta_t_data[i] = pw_draw(rng, q_buf, tau, b1_Q, bL_Q)
end

H = hermite_basis(eta_lag_data, K, sigma_y)
Q_mat = zeros(N, 3)
q_sorted = zeros(3)

# Neg loglik as function of slope (a_Q[2,:] all equal)
function neg_loglik_slope(slope)
    a_test = copy(a_Q)
    a_test[2, :] .= slope
    mul!(Q_mat, H, a_test)
    ll = 0.0
    for j in 1:N
        for l in 1:3; q_sorted[l] = Q_mat[j,l]; end
        sort!(q_sorted)
        ll += pw_logdens(eta_t_data[j], q_sorted, tau, b1_Q, bL_Q)
    end
    -ll / N
end

# Evaluate at a grid of slopes
slopes = collect(0.0:0.05:1.5)
nlls = [neg_loglik_slope(s) for s in slopes]
best_idx = argmin(nlls)
best_slope = slopes[best_idx]
@printf("  Grid search: best slope = %.2f (neg-ll = %.4f)\n",
        best_slope, nlls[best_idx])
@printf("  True slope = 0.80, neg-ll at truth = %.4f\n",
        neg_loglik_slope(0.8))

check("MLE slope near 0.8 (grid)", abs(best_slope - 0.8) < 0.15)

# Fine grid around truth
slopes_fine = collect(0.6:0.01:1.0)
nlls_fine = [neg_loglik_slope(s) for s in slopes_fine]
best_fine = slopes_fine[argmin(nlls_fine)]
@printf("  Fine grid: best slope = %.2f (neg-ll = %.4f)\n",
        best_fine, nlls_fine[argmin(nlls_fine)])
check("MLE slope near 0.8 (fine grid)", abs(best_fine - 0.8) < 0.05)

# Print profile around truth
println("\n  Profile likelihood around true slope:")
for s in [0.5, 0.6, 0.7, 0.75, 0.78, 0.80, 0.82, 0.85, 0.9, 1.0]
    @printf("    slope=%.2f  neg-ll=%.6f\n", s, neg_loglik_slope(s))
end

# ================================================================
println("\n","="^60)
println("TEST 7: MLE with observed eta — fix all but intercept a_Q[1,1]")
println("="^60)
println("  True a_Q[1,1] = -0.337. Vary it, check minimum at truth.")

function neg_loglik_intercept1(a01)
    a_test = copy(a_Q)
    a_test[1, 1] = a01
    mul!(Q_mat, H, a_test)
    ll = 0.0
    for j in 1:N
        for l in 1:3; q_sorted[l] = Q_mat[j,l]; end
        sort!(q_sorted)
        ll += pw_logdens(eta_t_data[j], q_sorted, tau, b1_Q, bL_Q)
    end
    -ll / N
end

intcpts = collect(-1.0:0.02:0.5)
nlls_i = [neg_loglik_intercept1(a) for a in intcpts]
best_i = intcpts[argmin(nlls_i)]
@printf("  Grid search: best intercept = %.3f\n", best_i)
@printf("  True intercept = -0.337, neg-ll at truth = %.6f\n",
        neg_loglik_intercept1(-0.337))
check("MLE intercept near -0.337", abs(best_i - (-0.337)) < 0.05)

println("\n  Profile likelihood around true intercept:")
for a in [-0.6, -0.5, -0.4, -0.35, -0.337, -0.3, -0.2, -0.1, 0.0]
    @printf("    a01=%.3f  neg-ll=%.6f\n", a, neg_loglik_intercept1(a))
end

# ================================================================
println("\n","="^60)
println("TEST 8: MLE with observed eta — fix all but tail b1_Q")
println("="^60)
println("  True b1_Q = 2.0. Vary it, check minimum at truth.")

function neg_loglik_b1(b1_test)
    b1_test <= 0 && return Inf
    mul!(Q_mat, H, a_Q)
    ll = 0.0
    for j in 1:N
        for l in 1:3; q_sorted[l] = Q_mat[j,l]; end
        sort!(q_sorted)
        ll += pw_logdens(eta_t_data[j], q_sorted, tau, b1_test, bL_Q)
    end
    -ll / N
end

b1s = collect(0.5:0.1:5.0)
nlls_b = [neg_loglik_b1(b) for b in b1s]
best_b = b1s[argmin(nlls_b)]
@printf("  Grid search: best b1 = %.1f\n", best_b)
@printf("  True b1 = 2.0, neg-ll at truth = %.6f\n", neg_loglik_b1(2.0))
check("MLE b1 near 2.0", abs(best_b - 2.0) < 0.5)

println("\n  Profile:")
for b in [0.5, 1.0, 1.5, 1.8, 2.0, 2.2, 2.5, 3.0, 4.0]
    @printf("    b1=%.1f  neg-ll=%.6f\n", b, neg_loglik_b1(b))
end

# ================================================================
println("\n","="^60)
println("TEST 9: Joint MLE with Nelder-Mead (observed eta, all params)")
println("="^60)

function neg_loglik_all(theta)
    a_test = reshape(theta[1:9], 3, 3)
    b1v = exp(theta[10]); bLv = exp(theta[11])
    mul!(Q_mat, H, a_test)
    ll = 0.0
    for j in 1:N
        for l in 1:3; q_sorted[l] = Q_mat[j,l]; end
        sort!(q_sorted)
        ll += pw_logdens(eta_t_data[j], q_sorted, tau, b1v, bLv)
    end
    -ll / N
end

theta_true = vcat(vec(a_Q), log(b1_Q), log(bL_Q))
@printf("  neg-ll at truth = %.6f\n", neg_loglik_all(theta_true))

# Start at truth and verify NM doesn't move
res_at_truth = optimize(neg_loglik_all, theta_true, NelderMead(),
                        Optim.Options(iterations=5000, show_trace=false))
theta_nm = Optim.minimizer(res_at_truth)
a_nm = reshape(theta_nm[1:9], 3, 3)
@printf("  NM from truth: neg-ll = %.6f\n", Optim.minimum(res_at_truth))
@printf("  slopes: [%.4f, %.4f, %.4f] (true: 0.8)\n", a_nm[2,:]...)
@printf("  intercepts: [%.4f, %.4f, %.4f] (true: [-0.337, 0, 0.337])\n", a_nm[1,:]...)

check("NM from truth stays near truth (slope)",
      all(abs.(a_nm[2,:] .- 0.8) .< 0.1))

# Start away from truth
theta_start = vcat(vec(a_Q .* 0.5), log(3.0), log(3.0))
@printf("\n  neg-ll at start (away) = %.6f\n", neg_loglik_all(theta_start))
res_away = optimize(neg_loglik_all, theta_start, NelderMead(),
                    Optim.Options(iterations=10000, show_trace=false))
theta_nm2 = Optim.minimizer(res_away)
a_nm2 = reshape(theta_nm2[1:9], 3, 3)
@printf("  NM from away: neg-ll = %.6f (truth: %.6f)\n",
        Optim.minimum(res_away), neg_loglik_all(theta_true))
@printf("  slopes: [%.4f, %.4f, %.4f]\n", a_nm2[2,:]...)
@printf("  intercepts: [%.4f, %.4f, %.4f]\n", a_nm2[1,:]...)
@printf("  b1=%.4f bL=%.4f (true: 2.0, 2.0)\n",
        exp(theta_nm2[10]), exp(theta_nm2[11]))

check("NM from away finds good neg-ll",
      Optim.minimum(res_away) < neg_loglik_all(theta_true) + 0.01)

# ================================================================
println("\n","="^60)
@printf("SUMMARY: %d passed, %d failed\n", passed, failed)
println("="^60)

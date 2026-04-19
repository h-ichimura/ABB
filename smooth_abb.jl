#=
smooth_abb.jl — Smooth version of ABB density (cubic interior, exponential tails)

Same parameterization as ABB for quantile knots:
  q_l(η_{t-1}) = Σ_k a_{kl} · He_k(η_{t-1}/σ_y)   for l = 1,...,L

But the density is smooth: cubic spline in log-density in interior,
linear (in log) on the tails — giving exponential tails.

The log-density of η_t given η_{t-1}:
  log f(x) = s(x; η_{t-1}) - log C(η_{t-1})
  s(x) = β_0 + β_1 x + Σ_l γ_l · [(x - q_l)₊³ - (q_l - x)₊³] * 1{l interior}
       = β_0 + β_1 x + Σ_l γ_l · (x - q_l)₊³          (Stone et al style)

For exponential left tail (x < q_1): we need log f linear in x.
  Using only (x - q_l)₊³, for x < q_1, s(x) = β_0 + β_1 x (linear).
  So left tail is exp(β_0 + β_1 x) — exponential with rate -β_1 (need β_1 > 0).
For exponential right tail (x > q_L): need (L + β_1 x) linear and γ summing
to something that makes the right tail finite. Actually the rightmost segment
(x > q_L) has s(x) = β_0 + β_1 x + Σ_l γ_l (x - q_l)³ — this is a CUBIC,
not linear, so the right tail may grow polynomially if Σ γ_l > 0 or decay
cubically if Σ γ_l < 0 (dominant term -|γ_sum| x³).

Actually for the right tail to be integrable, the cubic coefficient Σ γ_l
must be ≤ 0. The density then decays super-exponentially (~exp(-|c|x³)),
which is fine but quite thin-tailed.

This matches Stone's 1997 logspline: use truncated power basis, impose
constraints on boundary coefficients to ensure integrability.

Parameters:
  - a_Q: (K+1) × L quantile knot coefficients (from ABB, via QR)
  - b_coef: spline coefficients, (K+1) × (L+2) matrix:
      cols 1,2   → β_0, β_1
      cols 3..L+2 → γ_1, ..., γ_L
=#

using LinearAlgebra, Statistics, Random, Optim, Printf

# ================================================================
#  LOG-DENSITY s(x)
# ================================================================

function smooth_abb_s(x::Float64, beta0::Float64, beta1::Float64,
                      gamma::AbstractVector{Float64},
                      qknots::AbstractVector{Float64})
    s = beta0 + beta1 * x
    @inbounds for l in eachindex(gamma)
        d = x - qknots[l]
        d > 0 && (s += gamma[l] * d * d * d)
    end
    s
end

# ================================================================
#  GAUSS-LEGENDRE QUADRATURE
# ================================================================

function gauss_legendre(n::Int)
    beta = [i / sqrt(4i^2 - 1) for i in 1:n-1]
    J = SymTridiagonal(zeros(n), beta)
    lam, V = eigen(J)
    w = 2.0 .* V[1,:].^2
    lam, w
end

const GL16_nodes, GL16_weights = gauss_legendre(16)

function gl_integrate(f, a::Float64, b::Float64;
                      nodes=GL16_nodes, weights=GL16_weights)
    mid = (a + b) / 2; half = (b - a) / 2
    s = 0.0
    for i in eachindex(nodes)
        s += weights[i] * f(mid + half * nodes[i])
    end
    s * half
end

# ================================================================
#  NORMALIZING CONSTANT
#
# Density has exponential left tail (x < q_1): f(x) = exp(β_0 + β_1 x)
# Interior and right: evaluated numerically.
# If β_1 > 0: left tail integrable, contributes exp(β_0 + β_1 q_1) / β_1
# Otherwise: improper density — return huge value to penalize.
# ================================================================

function smooth_abb_normalizer(beta0::Float64, beta1::Float64,
                               gamma::AbstractVector{Float64},
                               qknots::AbstractVector{Float64};
                               hi::Float64=15.0)
    q_sorted = sort(collect(qknots))
    L = length(q_sorted)
    q1 = q_sorted[1]; qL = q_sorted[L]

    # Left tail: f(x) = exp(β_0 + β_1 x) for x < q_1
    # Integral = exp(β_0 + β_1 q_1) / β_1  (valid if β_1 > 0)
    C_left = if beta1 > 1e-8
        exp(beta0 + beta1 * q1) / beta1
    else
        1e10  # improper — large penalty
    end

    # Interior: integrate numerically between sorted knots
    C_interior = 0.0
    for i in 1:L-1
        C_interior += gl_integrate(
            x -> exp(smooth_abb_s(x, beta0, beta1, gamma, q_sorted)),
            q_sorted[i], q_sorted[i+1])
    end

    # Right: numerical integration from q_L to hi
    # Valid if Σ γ_l ≤ 0 (cubic decays)
    gamma_sum = sum(gamma)
    C_right = if gamma_sum < -1e-8
        gl_integrate(
            x -> exp(smooth_abb_s(x, beta0, beta1, gamma, q_sorted)),
            qL, hi)
    else
        1e10  # improper right tail
    end

    C_left + C_interior + C_right
end

# ================================================================
#  CONDITIONAL LOG-DENSITY log f(η_t | η_{t-1})
# ================================================================

function compute_hermite(eta_lag::Float64, K::Int, sigma_y::Float64)
    z = eta_lag / sigma_y
    hv = Vector{Float64}(undef, K + 1)
    hv[1] = 1.0
    K >= 1 && (hv[2] = z)
    for k in 2:K
        hv[k+1] = z * hv[k] - (k - 1) * hv[k-1]
    end
    hv
end

function smooth_abb_logdens(eta_t::Float64, eta_lag::Float64,
                            a_Q::Matrix{Float64},
                            b_coef::Matrix{Float64},
                            K::Int, L::Int, sigma_y::Float64)
    hv = compute_hermite(eta_lag, K, sigma_y)

    qknots = zeros(L)
    for l in 1:L
        qknots[l] = dot(view(a_Q, :, l), hv)
    end
    sort!(qknots)

    beta0 = dot(view(b_coef, :, 1), hv)
    beta1 = dot(view(b_coef, :, 2), hv)
    gamma = zeros(L)
    for l in 1:L
        gamma[l] = dot(view(b_coef, :, l + 2), hv)
    end

    sx = smooth_abb_s(eta_t, beta0, beta1, gamma, qknots)
    C = smooth_abb_normalizer(beta0, beta1, gamma, qknots)
    sx - log(max(C, 1e-300))
end

# ================================================================
#  MLE OF SPLINE COEFFICIENTS GIVEN QUANTILE KNOTS FIXED
# ================================================================

function fit_spline_given_quantiles(eta_t::Vector{Float64},
                                    eta_lag::Vector{Float64},
                                    a_Q::Matrix{Float64},
                                    K::Int, L::Int, sigma_y::Float64;
                                    maxiter::Int=100)
    n_obs = length(eta_t)

    # Precompute Hermite basis
    H = zeros(n_obs, K + 1)
    for j in 1:n_obs
        H[j, :] .= compute_hermite(eta_lag[j], K, sigma_y)
    end
    Q_mat = H * a_Q

    # Initial: β_1 = 1 (positive for integrable left tail),
    # γ_l slightly negative summing to < 0 (integrable right tail)
    b_coef = zeros(K + 1, L + 2)
    b_coef[1, 2] = 1.0            # β_1 constant part
    for l in 1:L
        b_coef[1, l + 2] = -0.2 / L  # γ_l
    end

    q_sorted = zeros(L)
    gamma_local = zeros(L)

    function neg_ll(theta::Vector{Float64})
        b = reshape(theta, K + 1, L + 2)
        ll = 0.0
        for j in 1:n_obs
            hv = view(H, j, :)
            beta0 = dot(view(b, :, 1), hv)
            beta1 = dot(view(b, :, 2), hv)
            for l in 1:L
                gamma_local[l] = dot(view(b, :, l + 2), hv)
            end
            for l in 1:L; q_sorted[l] = Q_mat[j, l]; end
            sort!(q_sorted)
            sx = smooth_abb_s(eta_t[j], beta0, beta1, gamma_local, q_sorted)
            C = smooth_abb_normalizer(beta0, beta1, gamma_local, q_sorted)
            ll += sx - log(max(C, 1e-300))
        end
        -ll / n_obs
    end

    theta0 = vec(copy(b_coef))
    res = optimize(neg_ll, theta0, LBFGS(),
                   Optim.Options(iterations=maxiter, g_tol=1e-5,
                                 show_trace=false))
    b_coef .= reshape(Optim.minimizer(res), K + 1, L + 2)
    b_coef
end

# ================================================================
#  TEST
# ================================================================

function test_smooth_abb()
    println("="^60)
    println("  SMOOTH-ABB TESTS")
    println("="^60)

    L = 3
    qknots = [-0.3371, 0.0, 0.3371]
    # β_1 > 0 for integrable left tail, γ summing to < 0 for right tail
    # For symmetry, need γ_total that cancels β_1 on right (so right tail also exponential-like decay)
    # At right tail: log f ~ β_1 x + 3(Σγ)(x - <q>)² x (dominant cubic term negative)
    # Simpler: use a DGP-like density ~ N(0, σ²) with σ ≈ 0.5
    beta0 = 0.0; beta1 = 2.0; gamma = [-0.7, -0.7, -0.7]

    println("\nTest: β_0=$beta0, β_1=$beta1, γ=$gamma")
    println("Knots at: $qknots")

    C = smooth_abb_normalizer(beta0, beta1, gamma, qknots)
    println("  C = $C")

    # Integrate density numerically to verify unit mass
    dx = 0.001
    xgrid = collect(-15.0:dx:15.0)
    total = sum(exp(smooth_abb_s(x, beta0, beta1, gamma, qknots))
                for x in xgrid) * dx
    println("  Raw integral: $total")
    println("  Normalized integral = $(total / C) (should be ≈ 1)")

    println("\nDensity values:")
    for x in [-2.0, -1.0, -0.5, -0.3371, 0.0, 0.3371, 0.5, 1.0]
        f = exp(smooth_abb_s(x, beta0, beta1, gamma, qknots)) / C
        @printf("  f(%.4f) = %.4f\n", x, f)
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    test_smooth_abb()
end

#=
pwl_abb.jl — ABB-style model with PIECEWISE-LINEAR-z LOGISTIC conditional CDF.

MODEL:
  y_{it} = η_{it} + ε_{it}
  η_{i1}           ~ PWL-logistic(q^init, α_L^init, α_R^init)
  η_{it} | η_{i,t-1} ~ PWL-logistic(q(η_{i,t-1}), α_L(η_{i,t-1}), α_R(η_{i,t-1}))
  ε_{it}           ~ PWL-logistic(q^eps, α_L^eps, α_R^eps)

where the conditional CDF is F(x | η) = σ(z(x; η)) with z piecewise linear.

PARAMETERS (for L=3, K=2):
  TRANSITION:
    a_Q ∈ ℝ^{3×3}          — Hermite coeffs of q_ℓ(η)  [9 params]
    b_L ∈ ℝ^3              — Hermite coeffs of log α_L(η)  [3 params]
    b_R ∈ ℝ^3              — Hermite coeffs of log α_R(η)  [3 params]
  INITIAL η_1: q^init ∈ ℝ^3, α_L^init, α_R^init > 0          [5 params]
  NOISE ε:    q^eps ∈ ℝ^3, α_L^eps, α_R^eps > 0              [5 params]
  Total: 25 parameters

PWL-z vs plain logistic:
  plain logistic has 6 transition params (μ, α as Hermite); equivalently
  3-column a_Q (9 params) + 2 shared tail slopes (2 params) = 11, but
  restricts inner slopes to equal tail slopes & all slopes to be the same.
  PWL-z frees the 3 quantiles plus 2 tail slopes — strictly more flexible.

QR ignores cross-percentile restrictions inherent in any of these models.
=#

include("ABB_three_period.jl")
include("pwl_logistic.jl")
using Optim, LinearAlgebra, Printf

const τ_STD = (0.25, 0.50, 0.75)
const LOGIT_τ_STD = (logit(0.25), logit(0.50), logit(0.75))

# ================================================================
#  CONDITIONAL PARAMETER EVALUATION
# ================================================================

"""Evaluate Hermite basis vector h(η) of length K+1."""
@inline function hermite_vec(η::Float64, K::Int, σy::Float64)
    z = η / σy
    hv = Vector{Float64}(undef, K + 1)
    hv[1] = 1.0
    K >= 1 && (hv[2] = z)
    for k in 2:K; hv[k+1] = z * hv[k] - (k - 1) * hv[k-1]; end
    hv
end

"""
Evaluate q(η) = (q_1, q_2, q_3) as tuple via GAP REPARAMETERIZATION.
a_Q: (K+1) × 3 matrix.  Column 1 = median coeffs, column 2 = log gap_L coeffs,
     column 3 = log gap_U coeffs.
  q_2(η) = h(η)' a_Q[:, 1]                    (conditional median)
  q_1(η) = q_2(η) - exp(h(η)' a_Q[:, 2])      (always < q_2)
  q_3(η) = q_2(η) + exp(h(η)' a_Q[:, 3])      (always > q_2)
Non-crossing guaranteed by construction: q_1 < q_2 < q_3 for all η.
"""
@inline function cond_q(η::Float64, a_Q::Matrix{Float64}, K::Int, σy::Float64)
    hv = hermite_vec(η, K, σy)
    q2 = 0.0; log_gapL = 0.0; log_gapU = 0.0
    for k in 1:K+1
        q2      += a_Q[k, 1] * hv[k]
        log_gapL += a_Q[k, 2] * hv[k]
        log_gapU += a_Q[k, 3] * hv[k]
    end
    q1 = q2 - exp(log_gapL)
    q3 = q2 + exp(log_gapU)
    (q1, q2, q3)
end

"""Evaluate (α_L, α_R) at η using log parameterization for positivity."""
@inline function cond_α_tails(η::Float64, b_L::Vector{Float64}, b_R::Vector{Float64},
                               K::Int, σy::Float64)
    hv = hermite_vec(η, K, σy)
    lL = 0.0; lR = 0.0
    for k in 1:K+1
        lL += b_L[k] * hv[k]
        lR += b_R[k] * hv[k]
    end
    (exp(lL), exp(lR))
end

# ================================================================
#  PARAMETERS STRUCT
# ================================================================

mutable struct PWLParams
    # Transition
    a_Q::Matrix{Float64}    # (K+1) × 3: Hermite coeffs of q_ℓ(η)
    b_L::Vector{Float64}    # K+1: Hermite coeffs of log α_L(η)
    b_R::Vector{Float64}    # K+1: Hermite coeffs of log α_R(η)
    # Initial η_1 (unconditional)
    q_init::Vector{Float64}      # 3
    αL_init::Float64
    αR_init::Float64
    # Noise ε (unconditional)
    q_eps::Vector{Float64}       # 3
    αL_eps::Float64
    αR_eps::Float64
end

function copy_pwl(p::PWLParams)
    PWLParams(copy(p.a_Q), copy(p.b_L), copy(p.b_R),
              copy(p.q_init), p.αL_init, p.αR_init,
              copy(p.q_eps),  p.αL_eps,  p.αR_eps)
end

# ================================================================
#  TRUE PARAMS (PLAIN LOGISTIC AR(1) AS SPECIAL CASE)
# ================================================================
"""
Construct PWL params that exactly replicate a plain AR(1) logistic model:
  η_t | η_{t-1} ~ Logistic(ρ·η_{t-1}, α_v).
Verifies PWL-z reduces to plain logistic in this special case.
"""
function make_true_pwl_logistic(; ρ=0.8, σ_v=0.5, σ_eps=0.3, σ_η1=1.0, K::Int=2)
    α_v   = π / (sqrt(3) * σ_v)
    α_eps = π / (sqrt(3) * σ_eps)
    α_η1  = π / (sqrt(3) * σ_η1)
    L = 3
    logit_τ = LOGIT_τ_STD

    # Gap reparameterization:
    #   q_2(η) = ρ η  →  a_Q[:, 1] = [0, ρ, 0] (median)
    #   gap_L = q_2 - q_1 = -logit(0.25)/α_v = log(3)/α_v  (constant)
    #   gap_U = q_3 - q_2 = logit(0.75)/α_v = log(3)/α_v  (constant)
    #   a_Q[:, 2] = [log(gap_L), 0, 0] = [log(log(3)/α_v), 0, 0]
    #   a_Q[:, 3] = [log(gap_U), 0, 0] = [log(log(3)/α_v), 0, 0]
    gap = log(3) / α_v  # = logit(0.75) / α_v
    a_Q = zeros(K + 1, L)
    a_Q[1, 1] = 0.0   # median intercept
    a_Q[2, 1] = ρ      # median slope
    a_Q[1, 2] = log(gap)  # log gap_L (constant)
    a_Q[1, 3] = log(gap)  # log gap_U (constant)

    # Tail slopes: α_L(η) = α_R(η) = α_v (constant over η)
    b_L = zeros(K + 1); b_L[1] = log(α_v)
    b_R = zeros(K + 1); b_R[1] = log(α_v)

    # Initial η_1 ~ Logistic(0, α_η1) → q_init_ℓ = logit_τ[ℓ]/α_η1, α tails = α_η1
    q_init = [logit_τ[ℓ] / α_η1 for ℓ in 1:L]
    αL_init = α_η1; αR_init = α_η1

    # Noise ε ~ Logistic(0, α_eps) → similar
    q_eps = [logit_τ[ℓ] / α_eps for ℓ in 1:L]
    αL_eps = α_eps; αR_eps = α_eps

    PWLParams(a_Q, b_L, b_R, q_init, αL_init, αR_init, q_eps, αL_eps, αR_eps)
end

# ================================================================
#  ASYMMETRIC-TAIL DGP (where plain logistic CANNOT fit)
# ================================================================
"""
Parameters that cannot be represented by plain logistic:
asymmetric tails and quantile spacing that differs on the two sides of the median.
"""
function make_true_pwl_asymmetric(; ρ=0.8, σy=1.0, K::Int=2)
    L = 3
    logit_τ = LOGIT_τ_STD

    # Conditional quantiles: q_ℓ(η) = ρ η + c_ℓ, where c_ℓ are ASYMMETRIC
    # (not c_1 = -c_3). E.g., c = (-0.6, 0, 0.3) — left tail further from median.
    c = (-0.6, 0.0, 0.3)
    a_Q = zeros(K + 1, L)
    for ℓ in 1:L
        a_Q[1, ℓ] = c[ℓ]
        a_Q[2, ℓ] = ρ  # common slope
    end

    # Asymmetric tail slopes: faster left decay, slower right decay
    b_L = zeros(K + 1); b_L[1] = log(3.0)
    b_R = zeros(K + 1); b_R[1] = log(1.5)

    # Initial
    q_init = [-1.0, 0.0, 0.8]; αL_init = 1.5; αR_init = 1.5
    q_eps  = [-0.15, 0.0, 0.15]; αL_eps = 5.0; αR_eps = 5.0

    PWLParams(a_Q, b_L, b_R, q_init, αL_init, αR_init, q_eps, αL_eps, αR_eps)
end

# ================================================================
#  DATA GENERATION
# ================================================================
function generate_data_pwl(N::Int, p::PWLParams, K::Int, σy::Float64;
                           seed::Int=42)
    rng = MersenneTwister(seed)
    T = 3
    logit_τ = LOGIT_τ_STD; τt = τ_STD
    η = zeros(N, T); y = zeros(N, T)

    # η_1 ~ unconditional PWL
    q_init_t = (p.q_init[1], p.q_init[2], p.q_init[3])
    for i in 1:N
        η[i, 1] = pwl_draw(rand(rng), q_init_t, logit_τ, p.αL_init, p.αR_init, τt)
    end

    # Transitions
    for t in 2:T, i in 1:N
        q_t = cond_q(η[i, t-1], p.a_Q, K, σy)
        αL, αR = cond_α_tails(η[i, t-1], p.b_L, p.b_R, K, σy)
        η[i, t] = pwl_draw(rand(rng), q_t, logit_τ, αL, αR, τt)
    end

    # Observations: y = η + ε
    q_eps_t = (p.q_eps[1], p.q_eps[2], p.q_eps[3])
    for t in 1:T, i in 1:N
        ε = pwl_draw(rand(rng), q_eps_t, logit_τ, p.αL_eps, p.αR_eps, τt)
        y[i, t] = η[i, t] + ε
    end

    y, η
end

# ================================================================
#  EXACT LIKELIHOOD (FORWARD FILTER)
# ================================================================
function pwl_neg_loglik(p::PWLParams, y::Matrix{Float64}, K::Int, σy::Float64;
                        grid_min::Float64=-6.0, grid_max::Float64=6.0, G::Int=120)
    N, T = size(y)
    grid = collect(range(grid_min, grid_max, length=G))
    dgrid = (grid_max - grid_min) / (G - 1)
    logit_τ = LOGIT_τ_STD

    # Precompute transition density: T_mat[g1, g2] = f(grid[g2] | grid[g1])
    T_mat = zeros(G, G)
    @inbounds for g1 in 1:G
        η_lag = grid[g1]
        q_t = cond_q(η_lag, p.a_Q, K, σy)
        αL, αR = cond_α_tails(η_lag, p.b_L, p.b_R, K, σy)
        for g2 in 1:G
            T_mat[g1, g2] = pwl_pdf(grid[g2], q_t, logit_τ, αL, αR)
        end
    end

    # f_init, f_eps evaluated at grid (unconditional)
    q_init_t = (p.q_init[1], p.q_init[2], p.q_init[3])
    q_eps_t  = (p.q_eps[1], p.q_eps[2], p.q_eps[3])
    f_init = [pwl_pdf(grid[g], q_init_t, logit_τ, p.αL_init, p.αR_init) for g in 1:G]

    # Forward filter per individual
    p_vec = zeros(G); p_new = zeros(G); eps_vec = zeros(G)
    total_ll = 0.0

    for i in 1:N
        @inbounds for g in 1:G
            eps_vec[g] = pwl_pdf(y[i, 1] - grid[g], q_eps_t, logit_τ, p.αL_eps, p.αR_eps)
            p_vec[g] = f_init[g] * eps_vec[g]
        end
        L1 = sum(p_vec) * dgrid
        L1 < 1e-300 && return Inf
        total_ll += log(L1)
        p_vec ./= L1

        for t in 2:T
            mul!(p_new, transpose(T_mat), p_vec)
            p_new .*= dgrid
            @inbounds for g in 1:G
                eps_vec[g] = pwl_pdf(y[i, t] - grid[g], q_eps_t, logit_τ, p.αL_eps, p.αR_eps)
                p_new[g] *= eps_vec[g]
            end
            Lt = sum(p_new) * dgrid
            Lt < 1e-300 && return Inf
            total_ll += log(Lt)
            p_new ./= Lt
            p_vec, p_new = p_new, p_vec
        end
    end

    -total_ll / N
end

# ================================================================
#  PARAMETER VECTOR ⇔ PWLParams
# ================================================================
function pack_pwl(p::PWLParams, K::Int)
    v = Float64[]
    append!(v, vec(p.a_Q))                              # 3(K+1)
    append!(v, p.b_L); append!(v, p.b_R)                # 2(K+1)
    append!(v, p.q_init); push!(v, log(p.αL_init), log(p.αR_init))  # 5
    append!(v, p.q_eps);  push!(v, log(p.αL_eps),  log(p.αR_eps))   # 5
    v
end

function unpack_pwl(v::Vector{Float64}, K::Int)
    P = K + 1
    i = 0
    a_Q = reshape(v[i+1:i+3*P], P, 3); i += 3*P
    b_L = v[i+1:i+P]; i += P
    b_R = v[i+1:i+P]; i += P
    q_init = v[i+1:i+3]; i += 3
    αL_init = exp(v[i+1]); αR_init = exp(v[i+2]); i += 2
    q_eps = v[i+1:i+3]; i += 3
    αL_eps = exp(v[i+1]); αR_eps = exp(v[i+2])
    PWLParams(a_Q, b_L, b_R, q_init, αL_init, αR_init, q_eps, αL_eps, αR_eps)
end

# ================================================================
#  ESTIMATION
# ================================================================
function estimate_pwl_ml(y::Matrix{Float64}, K::Int, σy::Float64,
                         p0::PWLParams;
                         G::Int=120, maxiter::Int=200, verbose::Bool=false)
    v0 = pack_pwl(p0, K)
    obj(v) = pwl_neg_loglik(unpack_pwl(v, K), y, K, σy; G=G)
    res = optimize(obj, v0, LBFGS(),
                   Optim.Options(iterations=maxiter, g_tol=1e-6,
                                 show_trace=verbose, show_every=10))
    unpack_pwl(Optim.minimizer(res), K), res
end

# ================================================================
#  TEST
# ================================================================
if abspath(PROGRAM_FILE) == @__FILE__
    K = 2; σy = 1.0

    println("="^70)
    println("  TEST 1: PWL-z model with PLAIN LOGISTIC DGP (sanity)")
    println("="^70)

    p_true = make_true_pwl_logistic(; ρ=0.8, σ_v=0.5, σ_eps=0.3, σ_η1=1.0, K=K)
    N = 800
    y, η = generate_data_pwl(N, p_true, K, σy; seed=42)

    @printf("\nGenerated N=%d, η range [%.3f, %.3f], y range [%.3f, %.3f]\n",
            N, extrema(η)..., extrema(y)...)
    @printf("corr(η_1, η_2) = %.3f (target 0.8)\n", cor(η[:, 1], η[:, 2]))

    # True params check
    @printf("\nTrue params:\n")
    @printf("  a_Q (transition quantiles): slopes=[%.4f, %.4f, %.4f], intcpts=[%.4f, %.4f, %.4f]\n",
            p_true.a_Q[2, :]..., p_true.a_Q[1, :]...)
    @printf("  b_L[1]=%.4f (→ α_L=%.4f), b_R[1]=%.4f (→ α_R=%.4f)\n",
            p_true.b_L[1], exp(p_true.b_L[1]), p_true.b_R[1], exp(p_true.b_R[1]))

    # Eval neg-ll at truth
    t_eval = @elapsed nll_true = pwl_neg_loglik(p_true, y, K, σy; G=120)
    @printf("neg-ll at truth: %.6f (eval %.2fs)\n", nll_true, t_eval)

    # Fit
    println("\nFitting PWL-z MLE (starting from perturbed truth)...")
    p0 = copy_pwl(p_true)
    p0.a_Q[2, :] .= 0.5   # slopes perturbed from 0.8
    p0.b_L[1] = log(2.0); p0.b_R[1] = log(2.0)  # tail slopes perturbed from 3.628

    t_mle = @elapsed p_mle, res = estimate_pwl_ml(y, K, σy, p0;
                                                   G=100, maxiter=150, verbose=true)
    nll_mle = pwl_neg_loglik(p_mle, y, K, σy; G=120)

    @printf("\n--- Results (time %.1fs, iters=%d) ---\n", t_mle, Optim.iterations(res))
    @printf("neg-ll: truth=%.4f, MLE=%.4f (diff=%+.4f)\n", nll_true, nll_mle, nll_mle - nll_true)
    @printf("Slopes: MLE=[%.4f, %.4f, %.4f] (true all 0.8)\n", p_mle.a_Q[2, :]...)
    @printf("Intcpts: MLE=[%.4f, %.4f, %.4f] (true %s)\n",
            p_mle.a_Q[1, :]..., round.(p_true.a_Q[1, :], digits=4))
    @printf("Tail slopes: MLE α_L=%.4f α_R=%.4f (true both %.4f)\n",
            exp(p_mle.b_L[1]), exp(p_mle.b_R[1]), exp(p_true.b_L[1]))
end

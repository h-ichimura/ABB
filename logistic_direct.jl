#=
logistic_direct.jl — Smooth logistic ABB model with DIRECT parameterization.

Column ℓ of a_Q gives Q_ℓ(η) = h(η)' a_Q[:, ℓ] directly (no gap transform).
Non-crossing ensured by starting inside the cone {Q₁ < Q₂ < Q₃ ∀η}
(equal slopes + ordered intercepts) and searching locally.

Asymmetric logistic density:
  f(x | q₁, q₂, q₃) = w × logistic(x; q₂, α_L)  for x ≤ q₂
                       = w × logistic(x; q₂, α_R)  for x > q₂
  where α_L = log(3)/(q₂-q₁), α_R = log(3)/(q₃-q₂), w = 2α_Lα_R/(α_L+α_R)
  Normalized: f/C with C = w (since ∫ w×logistic dx = w×1 = w...
  actually ∫ w×logistic_L dx on (-∞,q₂) = w/2, ∫ w×logistic_R dx on (q₂,∞) = w/2,
  total = w. So f_normalized = f/w = logistic on each half.)

No sorting. No gap reparameterization. Direct and clean.
=#

include("ABB_three_period.jl")
using Optim, Printf, LinearAlgebra, Random

σ_stable(z) = z >= 0 ? 1/(1+exp(-z)) : let e=exp(z); e/(1+e) end

# ================================================================
#  ASYMMETRIC LOGISTIC DENSITY (direct quantile parameterization)
# ================================================================

@inline function logistic_logpdf_direct(x::Float64, μ::Float64, α::Float64)
    z = α * (x - μ)
    log(α) - z - 2 * log1p(exp(-z))
end

"""
Normalized asymmetric logistic log-density.
f(x) = logistic(x; q₂, α_L) for x ≤ q₂, logistic(x; q₂, α_R) for x > q₂.
The weight w = 2α_Lα_R/(α_L+α_R) cancels with normalization C = w.
"""
function asym_logistic_logpdf_norm(x::Float64, q1::Float64, q2::Float64, q3::Float64)
    gap_L = q2 - q1; gap_R = q3 - q2
    (gap_L <= 0 || gap_R <= 0) && return -1e10
    log3 = log(3.0)
    α_L = log3 / gap_L; α_R = log3 / gap_R
    if x <= q2
        logistic_logpdf_direct(x, q2, α_L)
    else
        logistic_logpdf_direct(x, q2, α_R)
    end
end

function asym_logistic_pdf_norm(x::Float64, q1::Float64, q2::Float64, q3::Float64)
    exp(asym_logistic_logpdf_norm(x, q1, q2, q3))
end

"""
Draw from normalized asymmetric logistic with quantiles q₁ < q₂ < q₃.
CDF: F(x) = F_logistic(x; q₂, α_L) for x ≤ q₂  (range [0, 0.5])
     F(x) = F_logistic(x; q₂, α_R) for x > q₂  (range [0.5, 1])
Inverse CDF: x = q₂ + logit(u)/α_L for u ≤ 0.5
             x = q₂ + logit(u)/α_R for u > 0.5
"""
function asym_logistic_draw(rng::AbstractRNG, q1::Float64, q2::Float64, q3::Float64)
    gap_L = q2 - q1; gap_R = q3 - q2
    log3 = log(3.0)
    α_L = log3 / gap_L; α_R = log3 / gap_R
    u = rand(rng)
    u = clamp(u, 1e-15, 1-1e-15)
    if u <= 0.5
        q2 + log(u / (1 - u)) / α_L
    else
        q2 + log(u / (1 - u)) / α_R
    end
end

# ================================================================
#  DIRECT QUANTILE EVALUATION (no gap transform)
# ================================================================

"""
Evaluate conditional quantiles Q₁(η), Q₂(η), Q₃(η) directly from a_Q.
No sorting, no gap transform. Returns (q1, q2, q3).
"""
@inline function direct_quantiles(η::Float64, a_Q::Matrix{Float64},
                                   K::Int, σy::Float64)
    z = η / σy
    hv1 = 1.0; hv2 = z; hv3 = z*z - 1.0
    q1 = a_Q[1,1] + a_Q[2,1]*hv2
    q2 = a_Q[1,2] + a_Q[2,2]*hv2
    q3 = a_Q[1,3] + a_Q[2,3]*hv2
    if K >= 2
        q1 += a_Q[3,1]*hv3
        q2 += a_Q[3,2]*hv3
        q3 += a_Q[3,3]*hv3
    end
    (q1, q2, q3)
end

# ================================================================
#  TRUE PARAMETERS (direct, inside the cone)
# ================================================================

"""
True params for logistic AR(1): η_t | η_{t-1} ~ AsymLogistic(ρη, σ_v).
Equal slopes across τ → inside the non-crossing cone.
"""
function make_true_direct(; ρ=0.8, σ_v=0.5, σ_eps=0.3, σ_η1=1.0, K=2)
    log3 = log(3.0)
    # Logistic scale s = σ√3/π. Gap = 2s×log(3) for symmetric.
    s_v = σ_v * sqrt(3) / π
    s_eps = σ_eps * sqrt(3) / π
    s_η1 = σ_η1 * sqrt(3) / π

    a_Q = zeros(K+1, 3)
    # Intercepts: Q_ℓ(0) = median ± s_v × log(3)
    a_Q[1, 1] = -s_v * log3   # τ=0.25 intercept
    a_Q[1, 2] = 0.0            # τ=0.50 intercept (median at 0)
    a_Q[1, 3] = s_v * log3    # τ=0.75 intercept
    # Slopes: all equal (inside cone)
    a_Q[2, 1] = ρ
    a_Q[2, 2] = ρ
    a_Q[2, 3] = ρ

    a_init = [-s_η1*log3, 0.0, s_η1*log3]
    a_eps = [-s_eps*log3, 0.0, s_eps*log3]

    Params(a_Q, 0.0, 0.0, a_init, 0.0, 0.0, a_eps, 0.0, 0.0)
end

# ================================================================
#  DATA GENERATION
# ================================================================

function generate_data_direct(N::Int, par::Params, K::Int, σy::Float64; seed=42)
    rng = MersenneTwister(seed); T = 3
    η = zeros(N, T); y = zeros(N, T)

    for i in 1:N
        η[i,1] = asym_logistic_draw(rng, par.a_init[1], par.a_init[2], par.a_init[3])
    end
    for t in 2:T, i in 1:N
        q1, q2, q3 = direct_quantiles(η[i,t-1], par.a_Q, K, σy)
        η[i,t] = asym_logistic_draw(rng, q1, q2, q3)
    end
    for t in 1:T, i in 1:N
        y[i,t] = η[i,t] + asym_logistic_draw(rng, par.a_eps[1], par.a_eps[2], par.a_eps[3])
    end
    y, η
end

# ================================================================
#  EXACT LIKELIHOOD (FORWARD FILTER, Simpson's rule)
# ================================================================

function direct_neg_loglik(par::Params, y::Matrix{Float64}, K::Int, σy::Float64;
                           grid_min=-8.0, grid_max=8.0, G=201)
    N, T = size(y)
    G = isodd(G) ? G : G+1
    grid = collect(range(grid_min, grid_max, length=G))
    h = (grid_max - grid_min) / (G-1)
    sw = zeros(G); sw[1]=1.0; sw[G]=1.0
    for i in 2:G-1; sw[i] = iseven(i) ? 4.0 : 2.0; end
    sw .*= h/3

    # Transition matrix — return Inf if ANY grid point has crossing
    T_mat = zeros(G, G)
    @inbounds for g1 in 1:G
        q1, q2, q3 = direct_quantiles(grid[g1], par.a_Q, K, σy)
        gap_L = q2-q1; gap_R = q3-q2
        if gap_L <= 0 || gap_R <= 0
            return Inf  # infeasible θ
        end
        for g2 in 1:G
            T_mat[g1,g2] = asym_logistic_pdf_norm(grid[g2], q1, q2, q3)
        end
    end

    # f_init and f_eps (normalized asymmetric logistic)
    f_init = [asym_logistic_pdf_norm(grid[g], par.a_init[1], par.a_init[2], par.a_init[3])
              for g in 1:G]

    p = zeros(G); p_new = zeros(G); pw = zeros(G)
    total_ll = 0.0

    for i in 1:N
        @inbounds for g in 1:G
            f_eps = asym_logistic_pdf_norm(y[i,1]-grid[g], par.a_eps[1], par.a_eps[2], par.a_eps[3])
            p[g] = f_init[g] * f_eps
        end
        L1 = dot(p, sw)
        L1 < 1e-300 && return Inf
        total_ll += log(L1); p ./= L1

        for t in 2:T
            pw .= p .* sw
            mul!(p_new, transpose(T_mat), pw)
            @inbounds for g in 1:G
                f_eps = asym_logistic_pdf_norm(y[i,t]-grid[g], par.a_eps[1], par.a_eps[2], par.a_eps[3])
                p_new[g] *= f_eps
            end
            Lt = dot(p_new, sw)
            Lt < 1e-300 && return Inf
            total_ll += log(Lt); p_new ./= Lt
            p, p_new = p_new, p
        end
    end
    -total_ll / N
end

# ================================================================
#  MLE ESTIMATION (LBFGS, direct parameterization)
# ================================================================

"""Pack params to vector: a_Q (9) + a_init (3) + a_eps[1],a_eps[3] (2) = 14."""
function pack_direct(par::Params)
    v = Float64[]
    append!(v, vec(par.a_Q))  # 9, column-major
    append!(v, par.a_init)    # 3
    push!(v, par.a_eps[1], par.a_eps[3])  # 2 (a_eps[2]=0 fixed)
    v
end

function unpack_direct!(par::Params, v::Vector{Float64}, K::Int)
    np = (K+1)*3
    par.a_Q .= reshape(view(v, 1:np), K+1, 3)
    par.a_init .= view(v, np+1:np+3)
    par.a_eps[1] = v[np+4]
    par.a_eps[2] = 0.0  # fixed for identification
    par.a_eps[3] = v[np+5]
end

"""
Analytical minimum of the gap Q_{ℓ+1}(η) - Q_ℓ(η) over η ∈ [η_lo, η_hi].
Gap is quadratic in z = η/σ:  Δa₂·z² + Δa₁·z + (Δa₀ - Δa₂).
Minimum on a bounded interval: check vertex (if interior) and both endpoints.
"""
function gap_min_analytical(a_Q::AbstractMatrix, ℓ::Int, σy::Float64;
                             η_lo::Float64=-8.0, η_hi::Float64=8.0)
    Δa0 = a_Q[1, ℓ+1] - a_Q[1, ℓ]
    Δa1 = a_Q[2, ℓ+1] - a_Q[2, ℓ]
    Δa2 = size(a_Q, 1) >= 3 ? (a_Q[3, ℓ+1] - a_Q[3, ℓ]) : 0.0

    # Evaluate gap at z: Δa₂ z² + Δa₁ z + (Δa₀ - Δa₂)
    gap_at(z) = Δa2 * z^2 + Δa1 * z + (Δa0 - Δa2)

    z_lo = η_lo / σy; z_hi = η_hi / σy
    m = min(gap_at(z_lo), gap_at(z_hi))

    # Check vertex if parabola opens upward and vertex is inside [z_lo, z_hi]
    if abs(Δa2) > 1e-12
        z_star = -Δa1 / (2Δa2)
        if z_lo <= z_star <= z_hi
            m = min(m, gap_at(z_star))
        end
    end
    m
end

"""Check if parameter vector v is inside the non-crossing cone."""
function is_feasible(v::Vector{Float64}, K::Int, σy::Float64,
                     η_lo::Float64, η_hi::Float64)
    np = (K+1)*3
    a_Q = reshape(view(v, 1:np), K+1, 3)
    mg1 = gap_min_analytical(a_Q, 1, σy; η_lo=η_lo, η_hi=η_hi)
    mg2 = gap_min_analytical(a_Q, 2, σy; η_lo=η_lo, η_hi=η_hi)
    mg1 <= 0 && return false
    mg2 <= 0 && return false
    # Init ordering
    a_init = view(v, np+1:np+3)
    (a_init[1] >= a_init[2] || a_init[2] >= a_init[3]) && return false
    # Eps ordering (a_eps[2] = 0 fixed)
    (v[np+4] >= 0.0 || v[np+5] <= 0.0) && return false
    true
end

"""
Analytically compute maximum step α_max along direction d from v such that all
feasibility constraints remain satisfied at v + α·d for α ∈ [0, α_max].

Constraints (all enforced simultaneously):
  (a) Q_{ℓ+1}(η) > Q_ℓ(η) for ℓ = 1, 2 and η ∈ [η_lo, η_hi]
  (b) a_init[1] < a_init[2] < a_init[3]
  (c) a_eps[1] < 0 < a_eps[3]   (a_eps[2] = 0 fixed)

Gap along the ray for (a):
  gap_ℓ(α, z) = A(α)·z² + B(α)·z + C(α)   with z = η/σy
  where A, B, C are LINEAR in α. Min over z ∈ [z_lo, z_hi] occurs at an
  endpoint or at the vertex z* = -B/(2A) when A > 0 and z* ∈ (z_lo, z_hi).
  Both endpoint crossings and vertex crossings are roots of linear/quadratic
  equations in α → closed form.
Constraints (b), (c) are linear in α → closed form roots.

α_max is multiplied by `safety` (<1) to stay strictly inside.
"""
function max_feasible_step_analytical(v::Vector{Float64}, d::Vector{Float64},
                                      K::Int, σy::Float64,
                                      η_lo::Float64=-8.0, η_hi::Float64=8.0;
                                      α_cap::Float64=1e6, safety::Float64=0.999)
    np = (K+1)*3
    a_Q  = reshape(view(v, 1:np), K+1, 3)
    da_Q = reshape(view(d, 1:np), K+1, 3)
    z_lo = η_lo / σy; z_hi = η_hi / σy

    α_max = α_cap

    # --- (a) Non-crossing at each ℓ = 1, 2 ---
    for ℓ in 1:2
        # Basis: He_0=1, He_1=z, He_2=z²-1. Hence
        #   Q_{ℓ+1}(η) - Q_ℓ(η) = Δa[3]·z² + Δa[2]·z + (Δa[1] - Δa[3])
        # where Δa_k = a_Q[k, ℓ+1] - a_Q[k, ℓ]. Along the ray v + α d, each
        # Δa_k is LINEAR in α: Δa_k(α) = level_k + α·slope_k.
        a0_2 = (K+1 >= 3) ? (a_Q[3, ℓ+1] - a_Q[3, ℓ]) : 0.0
        a0_1 = a_Q[2, ℓ+1] - a_Q[2, ℓ]
        a0_0 = a_Q[1, ℓ+1] - a_Q[1, ℓ]
        a1_2 = (K+1 >= 3) ? (da_Q[3, ℓ+1] - da_Q[3, ℓ]) : 0.0
        a1_1 = da_Q[2, ℓ+1] - da_Q[2, ℓ]
        a1_0 = da_Q[1, ℓ+1] - da_Q[1, ℓ]

        # gap(α, z) = A(α)·z² + B(α)·z + C(α)
        A₀ = a0_2;         A₁ = a1_2
        B₀ = a0_1;         B₁ = a1_1
        C₀ = a0_0 - a0_2;  C₁ = a1_0 - a1_2

        # Endpoints z = z_lo, z_hi: gap is linear in α ⇒ single root per endpoint.
        for z in (z_lo, z_hi)
            g0 = A₀*z^2 + B₀*z + C₀
            sl = A₁*z^2 + B₁*z + C₁
            if sl < -1e-14
                α_cand = -g0 / sl
                if α_cand > 0 && α_cand < α_max
                    α_max = α_cand
                end
            end
        end

        # Interior vertex (binds only when A(α) > 0 and z* ∈ (z_lo, z_hi)):
        # Vertex value V(α) = C(α) - B(α)²/(4A(α)); V = 0 ⇔ 4A(α)C(α) = B(α)²
        # ⇒ P₂·α² + P₁·α + P₀ = 0, where
        P₀ = 4*A₀*C₀ - B₀^2
        P₁ = 4*A₀*C₁ + 4*A₁*C₀ - 2*B₀*B₁
        P₂ = 4*A₁*C₁ - B₁^2

        r1 = Inf; r2 = Inf
        if abs(P₂) > 1e-14
            disc = P₁^2 - 4*P₂*P₀
            if disc >= 0
                sq = sqrt(disc)
                r1 = (-P₁ + sq) / (2*P₂)
                r2 = (-P₁ - sq) / (2*P₂)
            end
        elseif abs(P₁) > 1e-14
            r1 = -P₀ / P₁
        end
        for α_c in (r1, r2)
            α_c > 0 && isfinite(α_c) || continue
            Aα = A₀ + α_c*A₁
            Aα > 1e-14 || continue              # only a MIN when parabola opens up
            Bα = B₀ + α_c*B₁
            z_star = -Bα / (2*Aα)
            (z_lo < z_star < z_hi) || continue  # vertex binds only inside interval
            if α_c < α_max
                α_max = α_c
            end
        end
    end

    # --- (b) a_init[1] < a_init[2] < a_init[3] ---
    for k in 1:2
        g0 = v[np + k + 1] - v[np + k]
        sl = d[np + k + 1] - d[np + k]
        if sl < -1e-14
            α_cand = -g0 / sl
            if α_cand > 0 && α_cand < α_max
                α_max = α_cand
            end
        end
    end

    # --- (c) a_eps[1] < 0 (at v[np+4]) and a_eps[3] > 0 (at v[np+5]) ---
    if d[np + 4] > 1e-14
        α_cand = -v[np + 4] / d[np + 4]
        if α_cand > 0 && α_cand < α_max
            α_max = α_cand
        end
    end
    if d[np + 5] < -1e-14
        α_cand = -v[np + 5] / d[np + 5]
        if α_cand > 0 && α_cand < α_max
            α_max = α_cand
        end
    end

    α_max * safety
end

# Backwards-compatible alias (old name used by removed code paths)
max_feasible_step = max_feasible_step_analytical

"""
Golden section search for minimum of f on [a, b].
"""
function golden_section(f, a::Float64, b::Float64; tol=1e-6, maxiter=50)
    φ = (sqrt(5) - 1) / 2
    c = b - φ * (b - a); d = a + φ * (b - a)
    fc = f(c); fd = f(d)
    for _ in 1:maxiter
        if fc < fd
            b = d; d = c; fd = fc
            c = b - φ * (b - a); fc = f(c)
        else
            a = c; c = d; fc = fd
            d = a + φ * (b - a); fd = f(d)
        end
        abs(b - a) < tol && break
    end
    (a + b) / 2
end

"""
∂log f(x; q₁,q₂,q₃)/∂(q₁,q₂,q₃) for normalized asymmetric logistic.
"""
function dlogf_dq(x::Float64, q1::Float64, q2::Float64, q3::Float64)
    log3 = log(3.0)
    gL = q2-q1; gR = q3-q2
    αL = log3/gL; αR = log3/gR
    if x <= q2
        z = αL*(x-q2); s = σ_stable(z)
        dlogf_dαL = 1/αL - (x-q2)*(2s-1)
        dαL_dq1 = αL^2/log3; dαL_dq2 = -αL^2/log3
        dlogf_dmu = αL*(2s-1)
        (dlogf_dαL*dαL_dq1, dlogf_dmu + dlogf_dαL*dαL_dq2, 0.0)
    else
        z = αR*(x-q2); s = σ_stable(z)
        dlogf_dαR = 1/αR - (x-q2)*(2s-1)
        dαR_dq2 = αR^2/log3; dαR_dq3 = -αR^2/log3
        dlogf_dmu = αR*(2s-1)
        (0.0, dlogf_dmu + dlogf_dαR*dαR_dq2, dlogf_dαR*dαR_dq3)
    end
end

"""
Compute neg-loglik AND analytical gradient via forward-backward smoothing.
Uses Simpson quadrature weights consistently.

Optimizations versus the original implementation:
  * α/β/L scratch buffers are hoisted out of the observation loop.
  * f_eps and its derivatives are precomputed once per (i, t) on the grid.
  * dlogT is laid out with parameter index FASTEST to keep the inner-loop
    gradient accumulation contiguous in memory.
"""
function negll_and_grad(v::Vector{Float64}, y::Matrix{Float64},
                        K::Int, σy::Float64;
                        G::Int=201,
                        grid_min::Float64=-8.0, grid_max::Float64=8.0)
    np = (K+1)*3
    a_Q    = reshape(view(v, 1:np), K+1, 3)
    a_init = view(v, np+1:np+3)
    a_eps1 = v[np+4]; a_eps3 = v[np+5]
    N, T_max = size(y)
    G = isodd(G) ? G : G+1
    grid = collect(range(grid_min, grid_max, length=G))
    h = (grid_max - grid_min) / (G-1)
    sw = zeros(G); sw[1]=1.0; sw[G]=1.0
    for i in 2:G-1; sw[i] = iseven(i) ? 4.0 : 2.0; end
    sw .*= h/3
    n_params = length(v)

    # ---- Transition density T_mat and ∂log T / ∂v ----
    # Layout dlogT_jgg[j, g2, g1] so the param dimension is contiguous.
    T_mat     = zeros(G, G)
    dlogT_jgg = zeros(np, G, G)
    @inbounds for g1 in 1:G
        z_h = grid[g1] / σy
        h0 = 1.0; h1 = z_h; h2 = z_h*z_h - 1.0
        q1 = a_Q[1,1] + a_Q[2,1]*h1 + a_Q[3,1]*h2
        q2 = a_Q[1,2] + a_Q[2,2]*h1 + a_Q[3,2]*h2
        q3 = a_Q[1,3] + a_Q[2,3]*h1 + a_Q[3,3]*h2
        if q2-q1 <= 0 || q3-q2 <= 0
            return (Inf, zeros(n_params))  # infeasible
        end
        for g2 in 1:G
            T_mat[g1, g2] = asym_logistic_pdf_norm(grid[g2], q1, q2, q3)
            d1, d2, d3 = dlogf_dq(grid[g2], q1, q2, q3)
            # Parameter layout (column-major vec(a_Q)):
            #   j = 1..3 → a_Q[1..3, 1]  (q1 coefs h0, h1, h2)
            #   j = 4..6 → a_Q[1..3, 2]  (q2 coefs)
            #   j = 7..9 → a_Q[1..3, 3]  (q3 coefs)
            dlogT_jgg[1, g2, g1] = d1 * h0
            dlogT_jgg[2, g2, g1] = d1 * h1
            dlogT_jgg[3, g2, g1] = d1 * h2
            dlogT_jgg[4, g2, g1] = d2 * h0
            dlogT_jgg[5, g2, g1] = d2 * h1
            dlogT_jgg[6, g2, g1] = d2 * h2
            dlogT_jgg[7, g2, g1] = d3 * h0
            dlogT_jgg[8, g2, g1] = d3 * h1
            dlogT_jgg[9, g2, g1] = d3 * h2
        end
    end

    # ---- f_init and ∂log f_init / ∂v on grid ----
    f_init     = zeros(G)
    di_init_1  = zeros(G)   # ∂ log f_init / ∂ a_init[1]
    di_init_2  = zeros(G)   # ∂ log f_init / ∂ a_init[2]
    di_init_3  = zeros(G)   # ∂ log f_init / ∂ a_init[3]
    @inbounds for g in 1:G
        f_init[g] = asym_logistic_pdf_norm(grid[g], a_init[1], a_init[2], a_init[3])
        d1, d2, d3 = dlogf_dq(grid[g], a_init[1], a_init[2], a_init[3])
        di_init_1[g] = d1; di_init_2[g] = d2; di_init_3[g] = d3
    end

    # ---- Hoisted per-observation buffers ----
    α_store = zeros(G, T_max)
    β_store = zeros(G, T_max)
    L_store = zeros(T_max)
    fe_vec  = zeros(G, T_max)   # f_eps(y[i,t] - grid[g])
    de1_vec = zeros(G, T_max)   # ∂ log f_eps / ∂ a_eps[1]
    de3_vec = zeros(G, T_max)   # ∂ log f_eps / ∂ a_eps[3]

    total_ll = 0.0
    grad_total = zeros(n_params)

    @inbounds for i in 1:N
        # Precompute f_eps and its derivatives on the grid for every t
        for t in 1:T_max
            for g in 1:G
                x = y[i, t] - grid[g]
                fe_vec[g, t] = asym_logistic_pdf_norm(x, a_eps1, 0.0, a_eps3)
                d1, _, d3 = dlogf_dq(x, a_eps1, 0.0, a_eps3)
                de1_vec[g, t] = d1
                de3_vec[g, t] = d3
            end
        end

        # --- Forward t = 1 ---
        for g in 1:G
            α_store[g, 1] = f_init[g] * fe_vec[g, 1]
        end
        L_store[1] = dot(view(α_store, :, 1), sw)
        if L_store[1] < 1e-300
            return (Inf, fill(NaN, n_params))
        end
        total_ll += log(L_store[1])
        invL1 = 1.0 / L_store[1]
        for g in 1:G; α_store[g, 1] *= invL1; end

        # --- Forward t = 2, ..., T_max ---
        for t in 2:T_max
            for g2 in 1:G
                s = 0.0
                for g1 in 1:G
                    s += T_mat[g1, g2] * α_store[g1, t-1] * sw[g1]
                end
                α_store[g2, t] = s * fe_vec[g2, t]
            end
            L_store[t] = dot(view(α_store, :, t), sw)
            if L_store[t] < 1e-300
                return (Inf, fill(NaN, n_params))
            end
            total_ll += log(L_store[t])
            invLt = 1.0 / L_store[t]
            for g in 1:G; α_store[g, t] *= invLt; end
        end

        # --- Backward: β_T ≡ 1, then recurse down ---
        for g in 1:G; β_store[g, T_max] = 1.0; end
        for t in T_max-1:-1:1
            invLp1 = 1.0 / L_store[t+1]
            for g1 in 1:G
                s = 0.0
                for g2 in 1:G
                    s += T_mat[g1, g2] * fe_vec[g2, t+1] *
                         β_store[g2, t+1] * sw[g2]
                end
                β_store[g1, t] = s * invLp1
            end
        end

        # --- Gradient contribution: f_init + f_eps at t=1 ---
        for g in 1:G
            wg = α_store[g, 1] * β_store[g, 1] * sw[g]
            grad_total[np+1] += wg * di_init_1[g]
            grad_total[np+2] += wg * di_init_2[g]
            grad_total[np+3] += wg * di_init_3[g]
            grad_total[np+4] += wg * de1_vec[g, 1]
            grad_total[np+5] += wg * de3_vec[g, 1]
        end

        # --- Gradient contribution: transitions + f_eps at t ≥ 2 ---
        for t in 2:T_max
            invLt = 1.0 / L_store[t]
            for g1 in 1:G
                a1sw1 = α_store[g1, t-1] * sw[g1]
                for g2 in 1:G
                    ξ = a1sw1 * T_mat[g1, g2] * fe_vec[g2, t] *
                        β_store[g2, t] * sw[g2] * invLt
                    # grad_total += ξ · dlogT_jgg[:, g2, g1]  (contiguous)
                    for j in 1:np
                        grad_total[j] += ξ * dlogT_jgg[j, g2, g1]
                    end
                end
            end
            for g in 1:G
                wg = α_store[g, t] * β_store[g, t] * sw[g]
                grad_total[np+4] += wg * de1_vec[g, t]
                grad_total[np+5] += wg * de3_vec[g, t]
            end
        end
    end

    nll = -total_ll / N
    grad = -grad_total ./ N
    (nll, grad)
end

"""
MLE via Optim.jl LBFGS with analytical gradient (no barrier).
Combined fg! avoids double-computing forward-backward.
Start inside cone (equal slopes); LBFGS stays inside since truth is also inside.
"""
function estimate_direct_ml(y::Matrix{Float64}, K::Int, σy::Float64,
                            par0::Params; G=201, maxiter=100, verbose=false,
                            grid_min=-8.0, grid_max=8.0)
    par = copy_params(par0)

    # Cache: store last (v, nll, grad) to avoid recomputation
    last_v = fill(NaN, (K+1)*3+5)
    last_nll = Ref(NaN)
    last_grad = fill(NaN, (K+1)*3+5)

    function compute!(w)
        if w != last_v
            nll, gr = negll_and_grad(w, y, K, σy; G=G, grid_min=grid_min,
                                      grid_max=grid_max, )
            last_v .= w; last_nll[] = nll; last_grad .= gr
        end
    end

    function obj(w)
        compute!(w)
        last_nll[]
    end

    function grad!(g, w)
        compute!(w)
        g .= last_grad
    end

    function fg!(g, w)
        compute!(w)
        g .= last_grad
        last_nll[]
    end

    v0 = pack_direct(par0)
    od = OnceDifferentiable(obj, grad!, fg!, v0)
    res = Optim.optimize(od, v0, LBFGS(),
                         Optim.Options(iterations=maxiter, g_tol=1e-5,
                                       show_trace=verbose, show_every=5))
    unpack_direct!(par, Optim.minimizer(res), K)
    nll_pure = direct_neg_loglik(par, y, K, σy; G=G)
    par, nll_pure
end

# ================================================================
#  INTERIOR-POINT LBFGS WITH ANALYTICAL FEASIBILITY + GOLDEN SECTION
# ================================================================

"""
Neg-loglik from parameter vector. `par_buf` is a scratch Params overwritten
in place — avoids allocating a new struct on every line-search evaluation.
"""
function direct_neg_loglik_vec!(v::Vector{Float64}, y::Matrix{Float64},
                                par_buf::Params, K::Int, σy::Float64;
                                G::Int=201,
                                grid_min::Float64=-8.0, grid_max::Float64=8.0)
    unpack_direct!(par_buf, v, K)
    direct_neg_loglik(par_buf, y, K, σy; G=G, grid_min=grid_min, grid_max=grid_max)
end

"""
L-BFGS with
  - analytical gradient (`negll_and_grad`, forward-backward smoothing);
  - analytical maximum feasible step (`max_feasible_step_analytical`);
  - golden-section line search on [0, α_max_feas] ensuring every iterate
    stays STRICTLY inside the non-crossing cone.

`par0` must already satisfy `is_feasible`. Returns `(par, nll)`.
"""
function estimate_direct_ml_gs(y::Matrix{Float64}, K::Int, σy::Float64,
                               par0::Params;
                               G::Int=201, maxiter::Int=100,
                               m_lbfgs::Int=8, verbose::Bool=false,
                               grid_min::Float64=-8.0, grid_max::Float64=8.0,
                               g_tol::Float64=1e-5,
                               ls_tol::Float64=1e-5,
                               α_abs_cap::Float64=10.0)
    par = copy_params(par0)
    v   = pack_direct(par0)
    n   = length(v)

    is_feasible(v, K, σy, grid_min, grid_max) ||
        error("estimate_direct_ml_gs: initial params violate quantile ordering")

    # L-BFGS circular buffer (most-recent pair kept at mod(k_stored-1, m)+1)
    S_buf = zeros(n, m_lbfgs)
    Y_buf = zeros(n, m_lbfgs)
    ρs    = zeros(m_lbfgs)
    α_hist = zeros(m_lbfgs)
    k_stored = 0

    # Scratch vectors (avoid per-call allocation in hot loops)
    q       = zeros(n)
    v_new   = zeros(n)
    s_vec   = zeros(n)
    y_vec   = zeros(n)
    ls_buf  = zeros(n)            # scratch for v + α·d inside golden section
    par_ls  = copy_params(par0)   # scratch Params for line-search nll

    nll, g = negll_and_grad(v, y, K, σy; G=G,
                            grid_min=grid_min, grid_max=grid_max)
    verbose && @printf("  iter %3d: nll = %.6f, |g| = %.4e\n", 0, nll, norm(g))

    last_α_max = NaN
    for iter in 1:maxiter
        gnorm = norm(g)
        if gnorm < g_tol
            verbose && @printf("  Converged: |g| = %.2e < %.1e\n", gnorm, g_tol)
            break
        end

        # ---- L-BFGS two-loop recursion: q ← H_k · g, then d = -q ----
        copyto!(q, g)
        bound = min(k_stored, m_lbfgs)
        for i in bound:-1:1
            idx = mod(k_stored - bound + i - 1, m_lbfgs) + 1
            α_hist[i] = ρs[idx] * dot(view(S_buf, :, idx), q)
            @inbounds for j in 1:n
                q[j] -= α_hist[i] * Y_buf[j, idx]
            end
        end
        if k_stored > 0
            idx_last = mod(k_stored - 1, m_lbfgs) + 1
            yy = dot(view(Y_buf, :, idx_last), view(Y_buf, :, idx_last))
            if yy > 1e-14
                γ = dot(view(S_buf, :, idx_last), view(Y_buf, :, idx_last)) / yy
                q .*= γ
            end
        end
        for i in 1:bound
            idx = mod(k_stored - bound + i - 1, m_lbfgs) + 1
            β = ρs[idx] * dot(view(Y_buf, :, idx), q)
            @inbounds for j in 1:n
                q[j] += (α_hist[i] - β) * S_buf[j, idx]
            end
        end
        # q now holds H_k g; descent direction is -q
        @inbounds for j in 1:n
            q[j] = -q[j]
        end
        d = q  # alias

        # Guard: fall back to steepest descent if not a descent direction
        dg = dot(d, g)
        if dg > -1e-12
            @inbounds for j in 1:n
                d[j] = -g[j]
            end
            dg = -dot(g, g)
            verbose && @printf("  iter %3d: non-descent LBFGS dir; fell back to -g\n", iter)
        end

        # ---- Analytical maximum feasible step ----
        α_max_feas = max_feasible_step_analytical(v, d, K, σy, grid_min, grid_max;
                                                  α_cap=1e6, safety=0.999)
        α_cap = min(α_max_feas, α_abs_cap)
        last_α_max = α_max_feas

        if α_cap < 1e-14
            verbose && @printf("  iter %3d: α_cap = %.2e < 1e-14, stopping.\n",
                               iter, α_cap)
            break
        end

        # ---- Golden section on [0, α_cap] ----
        nll_line = function (α)
            @inbounds for j in 1:n
                ls_buf[j] = v[j] + α * d[j]
            end
            direct_neg_loglik_vec!(ls_buf, y, par_ls, K, σy;
                                   G=G, grid_min=grid_min, grid_max=grid_max)
        end
        α_opt = golden_section(nll_line, 0.0, α_cap;
                               tol=ls_tol, maxiter=80)

        # Evaluate nll at α_opt. If no improvement, backtrack (halve α until
        # improvement, or fall back to steepest descent along -g).
        nll_try = nll_line(α_opt)
        if !(nll_try < nll - 1e-12)
            α_bt = α_opt
            found = false
            for _ in 1:30
                α_bt *= 0.5
                α_bt < 1e-16 && break
                if nll_line(α_bt) < nll - 1e-12
                    α_opt  = α_bt
                    nll_try = nll_line(α_bt)   # re-eval to ensure value matches α_opt
                    found  = true
                    break
                end
            end
            if !found
                # Last resort: use steepest descent direction
                @inbounds for j in 1:n; d[j] = -g[j]; end
                α_max_feas = max_feasible_step_analytical(v, d, K, σy,
                                                          grid_min, grid_max;
                                                          α_cap=1e6, safety=0.999)
                α_cap = min(α_max_feas, α_abs_cap)
                if α_cap < 1e-14
                    verbose && @printf("  iter %3d: α_cap=%.2e after SD fallback, stopping.\n",
                                       iter, α_cap)
                    break
                end
                α_opt = golden_section(nll_line, 0.0, α_cap;
                                       tol=ls_tol, maxiter=80)
                nll_try = nll_line(α_opt)
                if !(nll_try < nll - 1e-12)
                    α_bt = α_opt
                    for _ in 1:30
                        α_bt *= 0.5
                        α_bt < 1e-16 && break
                        if nll_line(α_bt) < nll - 1e-12
                            α_opt = α_bt
                            nll_try = nll_line(α_bt)
                            found  = true
                            break
                        end
                    end
                end
                if !found
                    verbose && @printf("  iter %3d: no descent found even with -g (α_cap=%.3e); stopping.\n",
                                       iter, α_cap)
                    break
                end
            end
        end

        @inbounds for j in 1:n
            v_new[j] = v[j] + α_opt * d[j]
        end

        # Feasibility sanity check (should always hold by construction)
        if !is_feasible(v_new, K, σy, grid_min, grid_max)
            verbose && @printf("  iter %3d: feasibility violated (α=%.3e, α_cap=%.3e); stopping.\n",
                               iter, α_opt, α_cap)
            break
        end

        nll_new, g_new = negll_and_grad(v_new, y, K, σy; G=G,
                                        grid_min=grid_min, grid_max=grid_max)

        if !(nll_new < nll - 1e-12)
            verbose && @printf("  iter %3d: negll_and_grad disagrees with nll_line (Δ=%+.2e); stopping.\n",
                               iter, nll_new - nll)
            break
        end

        # Meaningful-progress stop: relative improvement in nll below 1e-9 is
        # numerical noise, not optimisation progress.
        if (nll - nll_new) < 1e-9 * max(abs(nll), 1.0)
            verbose && @printf("  iter %3d: Δnll/|nll| = %.2e < 1e-9; converged.\n",
                               iter, (nll - nll_new) / max(abs(nll), 1.0))
            v   .= v_new; g .= g_new; nll = nll_new
            break
        end

        # ---- Store (s, y) pair if curvature is positive ----
        # Heuristic: if the line search had to backtrack heavily relative to α_cap
        # (i.e., LBFGS suggested a far larger step than was actually good), the
        # implied curvature is unreliable — reset the history so we restart with
        # steepest-descent-quality pairs.
        backtracked_heavily = α_opt < 1e-3 * α_cap
        if backtracked_heavily
            k_stored = 0
            verbose && @printf("  iter %3d: heavy backtrack (α/α_cap=%.1e); resetting L-BFGS memory.\n",
                               iter, α_opt / α_cap)
        else
            @inbounds for j in 1:n
                s_vec[j] = v_new[j] - v[j]
                y_vec[j] = g_new[j] - g[j]
            end
            sy = dot(s_vec, y_vec)
            if sy > 1e-10
                idx = mod(k_stored, m_lbfgs) + 1
                @inbounds for j in 1:n
                    S_buf[j, idx] = s_vec[j]
                    Y_buf[j, idx] = y_vec[j]
                end
                ρs[idx] = 1.0 / sy
                k_stored += 1
            end
        end

        v    .= v_new
        g    .= g_new
        nll   = nll_new

        verbose && @printf("  iter %3d: nll = %.6f, |g| = %.4e, α = %.3e (α_cap = %.3e)\n",
                           iter, nll, norm(g), α_opt, α_cap)
    end

    unpack_direct!(par, v, K)
    (par, nll)
end

# ================================================================
#  QR ESTIMATION (standard ABB M-step, logistic E-step)
# ================================================================

"""Logistic partial log-likelihood for MH E-step (direct parameterization)."""
function logistic_pll_direct(y, eta, t::Int, par::Params, K::Int, σy::Float64, T_max::Int)
    ll = asym_logistic_logpdf_norm(y[t]-eta[t], par.a_eps[1], par.a_eps[2], par.a_eps[3])
    if t == 1
        ll += asym_logistic_logpdf_norm(eta[1], par.a_init[1], par.a_init[2], par.a_init[3])
    end
    if t >= 2
        q1,q2,q3 = direct_quantiles(eta[t-1], par.a_Q, K, σy)
        ll += asym_logistic_logpdf_norm(eta[t], q1, q2, q3)
    end
    if t < T_max
        q1,q2,q3 = direct_quantiles(eta[t], par.a_Q, K, σy)
        ll += asym_logistic_logpdf_norm(eta[t+1], q1, q2, q3)
    end
    ll
end

"""MH E-step using logistic posterior (direct parameterization)."""
function logistic_estep_direct!(eta_all::Array{Float64,3}, y::Matrix{Float64},
                                 par::Params, cfg::Config, K::Int, σy::Float64)
    N,T,M = cfg.N, cfg.T, cfg.M
    n_draws = cfg.n_draws
    save_start = n_draws - M + 1
    eta_cur = eta_all[:,:,M]
    acc_count = zeros(T)
    eta_buf = zeros(T)

    pll = zeros(N, T)
    for i in 1:N, t in 1:T
        pll[i,t] = logistic_pll_direct(view(y,i,:), view(eta_cur,i,:), t, par, K, σy, T)
    end

    save_idx = 0
    for d in 1:n_draws
        for t in 1:T
            vp = cfg.var_prop[t]
            for i in 1:N
                @inbounds for s in 1:T; eta_buf[s] = eta_cur[i,s]; end
                eta_buf[t] = eta_cur[i,t] + sqrt(vp)*randn()
                prop = logistic_pll_direct(view(y,i,:), eta_buf, t, par, K, σy, T)
                if log(rand()) < prop - pll[i,t]
                    eta_cur[i,t] = eta_buf[t]
                    pll[i,t] = prop
                    t>1 && (pll[i,t-1] = logistic_pll_direct(
                        view(y,i,:), view(eta_cur,i,:), t-1, par, K, σy, T))
                    t<T && (pll[i,t+1] = logistic_pll_direct(
                        view(y,i,:), view(eta_cur,i,:), t+1, par, K, σy, T))
                    acc_count[t] += 1
                end
            end
        end
        if d >= save_start
            save_idx += 1
            eta_all[:,:,save_idx] .= eta_cur
        end
    end
    acc_count ./ (N*n_draws)
end

"""Full QR EM with logistic E-step, direct parameterization."""
function estimate_qr_direct(y::Matrix{Float64}, K::Int, σy::Float64,
                             par0::Params; S=50, M=20, n_draws=100)
    N = size(y,1); T = 3; L = 3
    tau = [0.25, 0.50, 0.75]
    cfg = Config(N, T, K, L, tau, σy, S, n_draws, M, fill(0.05, T))
    par = copy_params(par0)
    eta_all = zeros(N, T, M)
    for m in 1:M; eta_all[:,:,m] .= 0.6 .* y; end

    for iter in 1:S
        logistic_estep_direct!(eta_all, y, par, cfg, K, σy)
        m_step_qr!(par, eta_all, y, cfg)
        # Fix ε median at 0
        shift = par.a_eps[2]
        par.a_eps .-= shift
    end
    par
end

# ================================================================
#  NON-CROSSING CHECK
# ================================================================

function check_crossing(par::Params, K::Int, σy::Float64;
                        η_lo=-8.0, η_hi=8.0)
    mg1 = gap_min_analytical(par.a_Q, 1, σy; η_lo=η_lo, η_hi=η_hi)
    mg2 = gap_min_analytical(par.a_Q, 2, σy; η_lo=η_lo, η_hi=η_hi)
    (mg1 < 0 ? 1 : 0) + (mg2 < 0 ? 1 : 0)  # 0 = no crossing
end

# ================================================================
#  TEST
# ================================================================

if abspath(PROGRAM_FILE) == @__FILE__
    K = 2; σy = 1.0; N = 500

    println("="^70)
    println("  DIRECT PARAMETERIZATION: logistic ABB model")
    println("="^70)

    par_true = make_true_direct()
    @printf("\nTrue a_Q:\n")
    @printf("  slopes:     [%.4f, %.4f, %.4f] (all = ρ = 0.8)\n", par_true.a_Q[2,:]...)
    @printf("  intercepts: [%.4f, %.4f, %.4f]\n", par_true.a_Q[1,:]...)
    @printf("  crossing check: %d violations\n", check_crossing(par_true, K, σy))

    y, η = generate_data_direct(N, par_true, K, σy; seed=42)
    @printf("\nGenerated N=%d, η range [%.3f, %.3f]\n", N, extrema(η)...)
    @printf("corr(η₁,η₂) = %.3f (target ~0.8)\n", cor(η[:,1], η[:,2]))

    nll_true = direct_neg_loglik(par_true, y, K, σy; G=201)
    @printf("neg-ll at truth: %.4f\n", nll_true)

    # MLE from equal-slopes start (perturbed)
    println("\n--- MLE (LBFGS from perturbed truth) ---")
    par0 = copy_params(par_true)
    par0.a_Q[2,:] .= 0.5  # perturb slopes from 0.8 to 0.5 (still equal → inside cone)
    t_mle = @elapsed par_mle, nll_mle = estimate_direct_ml(y, K, σy, par0;
                                                           G=201, maxiter=100, verbose=true)
    @printf("\nneg-ll: truth=%.4f, MLE=%.4f (diff=%+.4f, %.1fs)\n",
            nll_true, nll_mle, nll_mle-nll_true, t_mle)
    @printf("MLE slopes:     [%.4f, %.4f, %.4f]\n", par_mle.a_Q[2,:]...)
    @printf("MLE intercepts: [%.4f, %.4f, %.4f]\n", par_mle.a_Q[1,:]...)
    @printf("Crossing check: %d violations\n", check_crossing(par_mle, K, σy))

    # QR
    println("\n--- QR (EM with logistic E-step) ---")
    par0_qr = copy_params(par_true)
    par0_qr.a_Q[2,:] .= 0.5
    t_qr = @elapsed par_qr = estimate_qr_direct(y, K, σy, par0_qr; S=30, M=20)
    @printf("QR slopes:     [%.4f, %.4f, %.4f] (%.1fs)\n", par_qr.a_Q[2,:]..., t_qr)
    @printf("QR intercepts: [%.4f, %.4f, %.4f]\n", par_qr.a_Q[1,:]...)
    @printf("Crossing check: %d violations\n", check_crossing(par_qr, K, σy))

    # Summary
    println("\n", "="^70)
    @printf("  True:  slopes=[%.4f, %.4f, %.4f], intcpts=[%.4f, %.4f, %.4f]\n",
            par_true.a_Q[2,:]..., par_true.a_Q[1,:]...)
    @printf("  MLE:   slopes=[%.4f, %.4f, %.4f], intcpts=[%.4f, %.4f, %.4f]\n",
            par_mle.a_Q[2,:]..., par_mle.a_Q[1,:]...)
    @printf("  QR:    slopes=[%.4f, %.4f, %.4f], intcpts=[%.4f, %.4f, %.4f]\n",
            par_qr.a_Q[2,:]..., par_qr.a_Q[1,:]...)
end

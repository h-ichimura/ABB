#=
test_gradient_compare.jl — Compare three gradient implementations:
  1. Analytical (hand-derived using density derivatives)
  2. ForwardDiff (automatic, needs numerically stable code)
  3. Central finite differences (baseline, known to be noisy)
=#

include("logistic_direct.jl")
using ForwardDiff

σ_stable(z) = z >= 0 ? 1/(1+exp(-z)) : let e=exp(z); e/(1+e) end

# ================================================================
#  ANALYTICAL GRADIENT of asym_logistic_logpdf_norm w.r.t. (q1, q2, q3)
# ================================================================

"""
∂log f(x; q₁,q₂,q₃)/∂(q₁,q₂,q₃) for normalized asymmetric logistic.
Returns (dq1, dq2, dq3).
"""
function dlogf_dq(x::Float64, q1::Float64, q2::Float64, q3::Float64)
    log3 = log(3.0)
    gL = q2-q1; gR = q3-q2
    αL = log3/gL; αR = log3/gR

    if x <= q2
        z = αL*(x-q2)
        s = σ_stable(z)
        # ∂log f/∂α_L:
        dlogf_dαL = 1/αL - (x-q2)*(2s-1)
        # ∂α_L/∂q₁ = α_L²/log3,  ∂α_L/∂q₂ = -α_L²/log3
        dαL_dq1 = αL^2/log3
        dαL_dq2 = -αL^2/log3
        # ∂log f/∂μ where μ=q₂: αL(2σ(z)-1)
        dlogf_dmu = αL*(2s-1)

        dq1 = dlogf_dαL * dαL_dq1
        dq2 = dlogf_dmu + dlogf_dαL * dαL_dq2
        dq3 = 0.0
    else
        z = αR*(x-q2)
        s = σ_stable(z)
        dlogf_dαR = 1/αR - (x-q2)*(2s-1)
        # ∂α_R/∂q₂ = α_R²/log3,  ∂α_R/∂q₃ = -α_R²/log3
        dαR_dq2 = αR^2/log3
        dαR_dq3 = -αR^2/log3
        dlogf_dmu = αR*(2s-1)

        dq1 = 0.0
        dq2 = dlogf_dmu + dlogf_dαR * dαR_dq2
        dq3 = dlogf_dαR * dαR_dq3
    end
    (dq1, dq2, dq3)
end

# ================================================================
#  ANALYTICAL GRADIENT of neg-loglik via forward-backward
# ================================================================

"""
Compute neg-loglik AND its gradient analytically using forward-backward.
Returns (nll, grad_vector).
"""
function negll_and_grad_analytical(v::Vector{Float64}, y::Matrix{Float64},
                                   K::Int, σy::Float64;
                                   G=201, grid_min=-8.0, grid_max=8.0)
    np = (K+1)*3
    a_Q = reshape(view(v, 1:np), K+1, 3)
    a_init = view(v, np+1:np+3)
    a_eps1 = v[np+4]; a_eps3 = v[np+5]
    a_eps = [a_eps1, 0.0, a_eps3]

    N, T_max = size(y)
    G = isodd(G) ? G : G+1
    grid = collect(range(grid_min, grid_max, length=G))
    h = (grid_max - grid_min) / (G-1)
    sw = zeros(G); sw[1]=1.0; sw[G]=1.0
    for i in 2:G-1; sw[i] = iseven(i) ? 4.0 : 2.0; end
    sw .*= h/3

    # Precompute transition density and its derivatives
    T_mat = zeros(G, G)
    # dT_dv[g1, g2, j] = ∂T_mat[g1,g2]/∂v[j]
    n_params = length(v)
    dlogT = zeros(G, G, n_params)  # ∂log T_mat / ∂v

    log3 = log(3.0)
    for g1 in 1:G
        η_lag = grid[g1]
        z_h = η_lag / σy
        hv = [1.0, z_h, z_h^2-1.0]  # Hermite basis (K=2)
        q1 = dot(a_Q[:,1], hv); q2 = dot(a_Q[:,2], hv); q3 = dot(a_Q[:,3], hv)
        gL = q2-q1; gR = q3-q2
        if gL <= 0 || gR <= 0
            T_mat[g1,:] .= 1e-300
            continue
        end
        for g2 in 1:G
            x = grid[g2]
            T_mat[g1,g2] = asym_logistic_pdf_norm(x, q1, q2, q3)
            # Density derivatives w.r.t. q1, q2, q3
            dq1, dq2, dq3 = dlogf_dq(x, q1, q2, q3)
            # Chain rule: ∂q_ℓ/∂a_Q[k,ℓ] = hv[k]
            for k in 1:K+1
                dlogT[g1, g2, (1-1)*(K+1)+k] = dq1 * hv[k]  # ∂/∂a_Q[k,1]
                dlogT[g1, g2, (2-1)*(K+1)+k] = dq2 * hv[k]  # ∂/∂a_Q[k,2]
                dlogT[g1, g2, (3-1)*(K+1)+k] = dq3 * hv[k]  # ∂/∂a_Q[k,3]
            end
        end
    end

    # f_init and its derivatives
    f_init = zeros(G)
    dlogf_init = zeros(G, n_params)
    for g in 1:G
        f_init[g] = asym_logistic_pdf_norm(grid[g], a_init[1], a_init[2], a_init[3])
        di1, di2, di3 = dlogf_dq(grid[g], a_init[1], a_init[2], a_init[3])
        dlogf_init[g, np+1] = di1
        dlogf_init[g, np+2] = di2
        dlogf_init[g, np+3] = di3
    end

    # Forward-backward
    total_ll = 0.0
    grad_total = zeros(n_params)

    for i in 1:N
        # Forward pass — store α_t(g) for each t
        α_store = zeros(G, T_max)  # α_t[g] (unnormalized)
        L_store = zeros(T_max)      # normalizing constants

        # t=1
        for g in 1:G
            fe = asym_logistic_pdf_norm(y[i,1]-grid[g], a_eps[1], a_eps[2], a_eps[3])
            α_store[g, 1] = f_init[g] * fe
        end
        L_store[1] = dot(α_store[:,1], sw)
        L_store[1] < 1e-300 && continue
        total_ll += log(L_store[1])
        α_store[:,1] ./= L_store[1]

        # t=2,3
        for t in 2:T_max
            for g2 in 1:G
                s = 0.0
                for g1 in 1:G; s += T_mat[g1,g2] * α_store[g1,t-1] * sw[g1]; end
                fe = asym_logistic_pdf_norm(y[i,t]-grid[g2], a_eps[1], a_eps[2], a_eps[3])
                α_store[g2, t] = s * fe
            end
            L_store[t] = dot(α_store[:,t], sw)
            L_store[t] < 1e-300 && continue
            total_ll += log(L_store[t])
            α_store[:,t] ./= L_store[t]
        end

        # Backward pass — β_t(g)
        β_store = zeros(G, T_max)
        β_store[:, T_max] .= 1.0

        for t in T_max-1:-1:1
            for g1 in 1:G
                s = 0.0
                for g2 in 1:G
                    fe = asym_logistic_pdf_norm(y[i,t+1]-grid[g2], a_eps[1], a_eps[2], a_eps[3])
                    s += T_mat[g1,g2] * fe * β_store[g2, t+1] * sw[g2]
                end
                β_store[g1, t] = s / L_store[t+1]
            end
        end

        # Accumulate gradient
        # γ_t(g) = α_t(g) β_t(g) (posterior state probability)
        # ξ_t(g1,g2) = α_{t-1}(g1) T(g1,g2) f_eps(y_t-g2) β_t(g2) / L_t

        # Contribution from f_init (need Simpson weight sw[g])
        for g in 1:G
            w_g = α_store[g,1] * β_store[g,1] * sw[g]
            for j in 1:n_params
                grad_total[j] += w_g * dlogf_init[g, j]
            end
        end

        # Contribution from f_eps at t=1 (need Simpson weight sw[g])
        for g in 1:G
            w_g = α_store[g,1] * β_store[g,1] * sw[g]
            de1, de2, de3 = dlogf_dq(y[i,1]-grid[g], a_eps[1], a_eps[2], a_eps[3])
            grad_total[np+4] += w_g * de1  # a_eps[1]
            # a_eps[2] = 0, fixed
            grad_total[np+5] += w_g * de3  # a_eps[3]
        end

        # Contribution from transitions and f_eps at t=2,3
        for t in 2:T_max
            for g1 in 1:G, g2 in 1:G
                fe = asym_logistic_pdf_norm(y[i,t]-grid[g2], a_eps[1], a_eps[2], a_eps[3])
                ξ = α_store[g1,t-1] * sw[g1] * T_mat[g1,g2] * fe * β_store[g2,t] * sw[g2] / L_store[t]
                # Transition derivative
                for j in 1:np  # only a_Q params affect T_mat
                    grad_total[j] += ξ * dlogT[g1, g2, j]
                end
            end
            # f_eps at time t (need Simpson weight sw[g])
            for g in 1:G
                w_g = α_store[g,t] * β_store[g,t] * sw[g]
                de1, de2, de3 = dlogf_dq(y[i,t]-grid[g], a_eps[1], a_eps[2], a_eps[3])
                grad_total[np+4] += w_g * de1
                grad_total[np+5] += w_g * de3
            end
        end
    end

    nll = -total_ll / N
    grad = -grad_total ./ N
    (nll, grad)
end

# ================================================================
#  ForwardDiff version (with numerically stable logistic)
# ================================================================

function stable_logistic_pdf_generic(x::T, μ::T, α::T) where T
    z = α * (x - μ)
    az = abs(z)
    e = exp(-az)
    α * e / (one(T) + e)^2
end

function negll_forwarddiff(v::AbstractVector{T}, y, K, σy, grid, sw) where T
    np = (K+1)*3
    a_Q = reshape(v[1:np], K+1, 3)
    a_init = v[np+1:np+3]
    a_eps = T[v[np+4], zero(T), v[np+5]]
    G = length(grid)
    N = size(y, 1)
    log3 = T(log(3.0))

    # Transition matrix
    T_mat = zeros(T, G, G)
    for g1 in 1:G
        z = T(grid[g1]) / T(σy)
        hv = T[one(T), z, z*z-one(T)]
        q1 = sum(a_Q[k,1]*hv[k] for k in 1:K+1)
        q2 = sum(a_Q[k,2]*hv[k] for k in 1:K+1)
        q3 = sum(a_Q[k,3]*hv[k] for k in 1:K+1)
        gL = q2-q1; gR = q3-q2
        (gL <= zero(T) || gR <= zero(T)) && continue
        αL = log3/gL; αR = log3/gR
        for g2 in 1:G
            x = T(grid[g2])
            T_mat[g1,g2] = x <= q2 ? stable_logistic_pdf_generic(x,q2,αL) :
                                       stable_logistic_pdf_generic(x,q2,αR)
        end
    end

    # f_init
    gLi = a_init[2]-a_init[1]; gRi = a_init[3]-a_init[2]
    αLi = log3/gLi; αRi = log3/gRi
    f_init = [let x=T(grid[g])
                  x <= a_init[2] ? stable_logistic_pdf_generic(x,a_init[2],αLi) :
                                    stable_logistic_pdf_generic(x,a_init[2],αRi)
              end for g in 1:G]

    # f_eps helper
    gLe = a_eps[2]-a_eps[1]; gRe = a_eps[3]-a_eps[2]
    αLe = log3/gLe; αRe = log3/gRe
    feps(ex) = ex <= a_eps[2] ? stable_logistic_pdf_generic(ex,a_eps[2],αLe) :
                                 stable_logistic_pdf_generic(ex,a_eps[2],αRe)

    # Forward filter
    total_ll = zero(T)
    p = zeros(T, G); p_new = zeros(T, G)
    for i in 1:N
        for g in 1:G; p[g] = f_init[g] * feps(T(y[i,1])-T(grid[g])); end
        L = sum(p[g]*T(sw[g]) for g in 1:G)
        L < T(1e-300) && return T(1e10)
        total_ll += log(L); p ./= L

        for t in 2:3
            for g2 in 1:G
                s = sum(T_mat[g1,g2]*p[g1]*T(sw[g1]) for g1 in 1:G)
                p_new[g2] = s * feps(T(y[i,t])-T(grid[g2]))
            end
            Lt = sum(p_new[g]*T(sw[g]) for g in 1:G)
            Lt < T(1e-300) && return T(1e10)
            total_ll += log(Lt); p_new ./= Lt
            p, p_new = p_new, p
        end
    end
    -total_ll / T(N)
end

# ================================================================
#  COMPARISON
# ================================================================

if abspath(PROGRAM_FILE) == @__FILE__
    K = 2; σy = 1.0; N = 50  # small for speed

    par_true = make_true_direct()
    y, η = generate_data_direct(N, par_true, K, σy; seed=42)
    v = pack_direct(par_true)

    G = 101
    grid = collect(range(-8.0, 8.0, length=G))
    h_grid = 16.0/(G-1)
    sw = zeros(G); sw[1]=1.0; sw[G]=1.0
    for i in 2:G-1; sw[i] = iseven(i) ? 4.0 : 2.0; end
    sw .*= h_grid/3

    println("="^70)
    println("  GRADIENT COMPARISON (N=$N, G=$G)")
    println("="^70)

    # 1. Analytical
    println("\n1. Analytical gradient:")
    t1 = @elapsed nll1, g1 = negll_and_grad_analytical(v, y, K, σy; G=G)
    @printf("   neg-ll = %.6f, |g| = %.4f, time = %.3fs\n", nll1, norm(g1), t1)

    # 2. ForwardDiff
    println("\n2. ForwardDiff gradient:")
    t2 = @elapsed begin
        nll2 = negll_forwarddiff(v, y, K, σy, grid, sw)
        g2 = ForwardDiff.gradient(w -> negll_forwarddiff(w, y, K, σy, grid, sw), v)
    end
    @printf("   neg-ll = %.6f, |g| = %.4f, time = %.3fs\n", nll2, norm(g2), t2)

    # 3. Central finite differences
    println("\n3. Central finite differences:")
    par = copy_params(par_true)
    function negll_fd(w)
        unpack_direct!(par, w, K)
        direct_neg_loglik(par, y, K, σy; G=G)
    end
    t3 = @elapsed begin
        nll3 = negll_fd(v)
        g3 = zeros(length(v))
        ε = 1e-5
        for j in 1:length(v)
            v[j] += ε; fp = negll_fd(v)
            v[j] -= 2ε; fm = negll_fd(v)
            g3[j] = (fp - fm) / (2ε)
            v[j] += ε
        end
    end
    @printf("   neg-ll = %.6f, |g| = %.4f, time = %.3fs\n", nll3, norm(g3), t3)

    # Compare
    println("\n--- Comparison ---")
    @printf("   neg-ll:  analytical=%.6f, ForwardDiff=%.6f, FD=%.6f\n", nll1, nll2, nll3)
    @printf("   |g_analytical - g_FD|     = %.2e\n", norm(g1 - g3))
    @printf("   |g_ForwardDiff - g_FD|    = %.2e\n", norm(g2 - g3))
    @printf("   |g_analytical - g_FwdDiff|= %.2e\n", norm(g1 - g2))
    @printf("   Speed: analytical %.3fs, ForwardDiff %.3fs, FD %.3fs\n", t1, t2, t3)
    @printf("   Speedup: analytical vs FD = %.1fx, FwdDiff vs FD = %.1fx\n",
            t3/t1, t3/t2)

    println("\n--- Per-component comparison (first 5) ---")
    @printf("   %5s %12s %12s %12s\n", "param", "analytical", "ForwardDiff", "FD")
    for j in 1:min(5, length(v))
        @printf("   %5d %12.6f %12.6f %12.6f\n", j, g1[j], g2[j], g3[j])
    end
end

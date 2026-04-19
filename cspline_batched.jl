#=
cspline_batched.jl — Batched forward filter for cubic spline ABB model

Key optimization: instead of processing N observations sequentially,
batch all N filtering distributions into a G×N matrix and use a single
matrix multiply for the propagation step.

CPU version (uses BLAS for the batched multiply).
GPU version would use CuArray + cuBLAS.

  P = [p₁ p₂ ... pₙ]   (G × N matrix)
  Propagation: P_new = T' × diag(sw) × P  =  T' × (sw .* P)
  But sw is the same for all observations, so:
    P_sw = P .* sw       (element-wise, G × N)
    P_new = T' * P_sw    (single BLAS matrix multiply)

This replaces N sequential G-dim mat-vecs with ONE (G×G)×(G×N) multiply.
=#

include("cspline_abb.jl")
using Printf

"""
Batched forward filter: compute neg-loglik for ALL N observations simultaneously.
Uses a single matrix multiply per time step instead of N separate mat-vecs.
"""
function cspline_neg_loglik_batched(a_Q::Matrix{Float64}, β_L::Float64, β_R::Float64,
                                    a_init::Vector{Float64}, a_eps1::Float64, a_eps3::Float64,
                                    y::Matrix{Float64}, K::Int, σy::Float64, τ::Vector{Float64};
                                    G::Int=201, grid_min::Float64=-8.0, grid_max::Float64=8.0)
    N, T = size(y)
    G = isodd(G) ? G : G+1
    grid = collect(range(grid_min, grid_max, length=G))
    h = (grid_max - grid_min) / (G-1)
    sw = zeros(G); sw[1]=1.0; sw[G]=1.0
    @inbounds for i in 2:G-1; sw[i] = iseven(i) ? 4.0 : 2.0; end
    sw .*= h/3

    # Build transition matrix (same as before)
    T_mat = zeros(G, G)
    cspline_transition_matrix!(T_mat, grid, a_Q, β_L, β_R, K, σy, τ)

    # Transpose T for the batched multiply: T_t = T'
    T_t = collect(transpose(T_mat))  # G × G, contiguous

    # Init density on grid
    (a_init[2] <= a_init[1] || a_init[3] <= a_init[2]) && return Inf
    s_init = solve_cspline_values(a_init, β_L, β_R, τ)
    masses_init = cspline_masses(a_init, s_init, β_L, β_R)
    C_init = sum(masses_init)
    C_init < 1e-300 && return Inf
    f_init = [exp(cspline_eval(grid[g], a_init, s_init, β_L, β_R)) / C_init for g in 1:G]

    # Eps density
    a_eps = [a_eps1, 0.0, a_eps3]
    (a_eps[2] <= a_eps[1] || a_eps[3] <= a_eps[2]) && return Inf
    s_eps = solve_cspline_values(a_eps, β_L, β_R, τ)
    masses_eps = cspline_masses(a_eps, s_eps, β_L, β_R)
    C_eps = sum(masses_eps)
    C_eps < 1e-300 && return Inf

    # Pre-compute eps density at all (y_it - grid_g) values: F_eps[g, i, t]
    # This is G × N × T — for N=500, G=201, T=3: ~300K entries, ~2.4 MB
    F_eps = zeros(G, N, T)
    @inbounds for t_step in 1:T, i in 1:N, g in 1:G
        F_eps[g, i, t_step] = exp(cspline_eval(y[i,t_step] - grid[g], a_eps, s_eps, β_L, β_R)) / C_eps
    end

    # Batched forward filter
    # P[g, i] = filtering distribution for observation i at grid point g
    P = zeros(G, N)
    P_sw = zeros(G, N)  # Simpson-weighted P
    P_new = zeros(G, N)

    total_ll = 0.0

    # Step 1: P = f_init .* F_eps[:, :, 1]
    @inbounds for i in 1:N, g in 1:G
        P[g, i] = f_init[g] * F_eps[g, i, 1]
    end
    # Normalize: L_i = ∫ P[:, i] × sw
    @inbounds for i in 1:N
        Li = 0.0
        for g in 1:G; Li += P[g, i] * sw[g]; end
        Li < 1e-300 && return Inf
        total_ll += log(Li)
        inv_Li = 1.0 / Li
        for g in 1:G; P[g, i] *= inv_Li; end
    end

    # Steps 2, 3: propagate
    for t_step in 2:T
        # P_sw = P .* sw (broadcast sw across columns)
        @inbounds for i in 1:N, g in 1:G
            P_sw[g, i] = P[g, i] * sw[g]
        end

        # BATCHED MATRIX MULTIPLY: P_new = T' × P_sw
        # This is the key: ONE matrix multiply instead of N mat-vecs
        mul!(P_new, T_t, P_sw)

        # Multiply by eps density and normalize
        @inbounds for i in 1:N, g in 1:G
            P_new[g, i] *= F_eps[g, i, t_step]
        end
        @inbounds for i in 1:N
            Li = 0.0
            for g in 1:G; Li += P_new[g, i] * sw[g]; end
            Li < 1e-300 && return Inf
            total_ll += log(Li)
            inv_Li = 1.0 / Li
            for g in 1:G; P_new[g, i] *= inv_Li; end
        end
        P .= P_new
    end

    -total_ll / N
end

# ================================================================
#  TEST: Compare batched vs sequential
# ================================================================

function test_batched()
    K = 2; σy = 1.0; τ = [0.25, 0.50, 0.75]; N = 500

    println("="^60)
    println("  BATCHED vs SEQUENTIAL FORWARD FILTER")
    println("="^60)

    tp = make_true_cspline()
    y, eta = generate_data_cspline(N, tp.a_Q, tp.β_L, tp.β_R,
                                    tp.a_init, tp.a_eps1, tp.a_eps3,
                                    K, σy, τ; seed=42)

    # Sequential (existing)
    nll_seq = cspline_neg_loglik(tp.a_Q, tp.β_L, tp.β_R, tp.a_init, tp.a_eps1, tp.a_eps3,
                                 y, K, σy, τ; G=201)
    t_seq = @elapsed cspline_neg_loglik(tp.a_Q, tp.β_L, tp.β_R, tp.a_init, tp.a_eps1, tp.a_eps3,
                                         y, K, σy, τ; G=201)

    # Batched
    nll_bat = cspline_neg_loglik_batched(tp.a_Q, tp.β_L, tp.β_R, tp.a_init, tp.a_eps1, tp.a_eps3,
                                          y, K, σy, τ; G=201)
    t_bat = @elapsed cspline_neg_loglik_batched(tp.a_Q, tp.β_L, tp.β_R, tp.a_init, tp.a_eps1, tp.a_eps3,
                                                 y, K, σy, τ; G=201)

    @printf("Sequential: nll=%.8f  time=%.3fs\n", nll_seq, t_seq)
    @printf("Batched:    nll=%.8f  time=%.3fs\n", nll_bat, t_bat)
    @printf("Speedup: %.1fx\n", t_seq / t_bat)
    @printf("Difference: %.2e\n", abs(nll_seq - nll_bat))
end

if abspath(PROGRAM_FILE) == @__FILE__
    test_batched()
end

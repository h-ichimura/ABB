#=
test_feasible_step.jl — sanity tests for max_feasible_step_analytical.

For each test:
  * Build a feasible v and a random direction d.
  * Compute α_max from max_feasible_step_analytical.
  * Verify:
      (i)  v + α_max·d is feasible (may be tight to boundary via `safety`).
      (ii) v + α·d with α slightly larger than α_max / safety is INFEASIBLE.
      (iii) The analytical α_max matches a brute-force bisection to tight tolerance.
=#

include("logistic_direct.jl")
using Random, Printf

const η_lo = -8.0
const η_hi =  8.0
const σy   = 1.0
const K    = 2

"""Bisection fallback to find the max α such that v+αd is feasible."""
function max_feasible_step_bisect(v::Vector{Float64}, d::Vector{Float64};
                                  α_init::Float64=1.0, iters::Int=80)
    α_hi = α_init
    for _ in 1:40
        is_feasible(v .+ α_hi .* d, K, σy, η_lo, η_hi) || break
        α_hi *= 2
    end
    if is_feasible(v .+ α_hi .* d, K, σy, η_lo, η_hi)
        return α_hi  # never infeasible (up to α_hi)
    end
    α_l = 0.0
    for _ in 1:iters
        m = 0.5*(α_l + α_hi)
        if is_feasible(v .+ m .* d, K, σy, η_lo, η_hi)
            α_l = m
        else
            α_hi = m
        end
    end
    α_l
end

"""Compute true gap minimum over η ∈ [η_lo, η_hi] at v+αd for ℓ ∈ {1,2}."""
function gap_min_at(v, d, α)
    w = v .+ α .* d
    np = (K+1)*3
    aQ = reshape(view(w, 1:np), K+1, 3)
    m1 = gap_min_analytical(aQ, 1, σy; η_lo=η_lo, η_hi=η_hi)
    m2 = gap_min_analytical(aQ, 2, σy; η_lo=η_lo, η_hi=η_hi)
    min(m1, m2)
end

function pack_from(aQ, a_init, a_eps)
    v = Float64[]
    append!(v, vec(aQ)); append!(v, a_init)
    push!(v, a_eps[1], a_eps[3])
    v
end

function rand_direction(n::Int, rng::AbstractRNG; scale=0.3)
    d = scale .* randn(rng, n)
    d
end

function run_case(name::String, v::Vector{Float64}, d::Vector{Float64})
    @printf("\n--- %s ---\n", name)
    @assert is_feasible(v, K, σy, η_lo, η_hi) "initial v is infeasible"

    α_an = max_feasible_step_analytical(v, d, K, σy, η_lo, η_hi;
                                         α_cap=1e6, safety=1.0)  # no safety margin
    α_bi = max_feasible_step_bisect(v, d)
    @printf("  analytical α_max = %.8e\n", α_an)
    @printf("  bisection  α_max = %.8e\n", α_bi)

    gap_at_an = gap_min_at(v, d, α_an)
    @printf("  gap_min at α_an  = %+.3e  (expected ≈ 0⁺ if boundary binds here)\n",
            gap_at_an)

    # Just below α_an: should still be feasible
    α_below = α_an * (1 - 1e-6)
    gap_below = gap_min_at(v, d, α_below)
    feas_below = is_feasible(v .+ α_below .* d, K, σy, η_lo, η_hi)
    @printf("  α·(1-1e-6): feasible=%s, gap_min=%+.3e\n", feas_below, gap_below)

    # Just above α_an: should be infeasible (if binding)
    α_above = α_an * (1 + 1e-6)
    gap_above = gap_min_at(v, d, α_above)
    feas_above = is_feasible(v .+ α_above .* d, K, σy, η_lo, η_hi)
    @printf("  α·(1+1e-6): feasible=%s, gap_min=%+.3e\n", feas_above, gap_above)

    # Compare to bisection
    diff = abs(α_an - α_bi) / max(α_bi, 1e-12)
    @printf("  relative diff to bisect = %.2e  %s\n", diff,
            diff < 1e-5 ? "✓" : (α_an >= 1e5 && α_bi >= 1e5 ? "✓ (both >> 1)" : "✗"))
end

# ================================================================
# Case 1: truth with equal slopes + ordered intercepts; direction = random
# ================================================================
println("="^70)
println("  max_feasible_step_analytical sanity tests")
println("="^70)

rng = MersenneTwister(1)
par_true = make_true_direct()
v_true = pack_direct(par_true)
@assert is_feasible(v_true, K, σy, η_lo, η_hi)

for seed in 1:5
    rng = MersenneTwister(seed)
    d = rand_direction(length(v_true), rng; scale=0.3)
    run_case("truth + random d (seed=$seed)", v_true, d)
end

# ================================================================
# Case 2: direction that EXPLICITLY pushes q1 toward q2 (should bind at endpoint)
# ================================================================
println("\n", "="^70)
println("  Case: direction shrinks intercept gap  (should hit endpoint)")
println("="^70)
d = zeros(length(v_true))
d[1] = +0.5   # push a_Q[1,1] (q1 intercept) up — shrinks gap between q1 and q2
run_case("intercept-1 push-up", v_true, d)

# ================================================================
# Case 3: direction that pushes slope asymmetrically (should bind at a grid endpoint)
# ================================================================
d = zeros(length(v_true))
d[2] = +0.2   # increase slope of q1 (a_Q[2,1]) — makes Q_1(η_hi) closer to Q_2(η_hi)
run_case("slope-1 push-up (should bind at z_hi)", v_true, d)

d = zeros(length(v_true))
d[2] = -0.2   # decrease slope of q1 (makes Q_1 at z_lo closer to Q_2 at z_lo)
run_case("slope-1 push-down (should bind at z_lo)", v_true, d)

# ================================================================
# Case 4: direction that changes quadratic coeff — triggers interior vertex
# ================================================================
d = zeros(length(v_true))
d[3]        = +0.5    # increase quadratic of q1 (opens parabola UP in Q1)
d[9]        = -0.5    # decrease quadratic of q3 (opens parabola DOWN in Q3)
# Net: makes Q_1 bigger in middle (vertex binds for ℓ=1)
run_case("quadratic opposite signs (vertex binding)", v_true, d)

# ================================================================
# Case 5: direction in init / eps space
# ================================================================
d = zeros(length(v_true))
d[10] = +0.5  # push a_init[1] up toward a_init[2]
run_case("a_init[1] push-up (init ordering binds)", v_true, d)

d = zeros(length(v_true))
d[13] = +0.5  # push a_eps[1] up toward 0
run_case("a_eps[1] push-up (eps ordering binds)", v_true, d)

# ================================================================
# Case 6: direction that NEVER leaves the cone (should return α_cap unchanged)
# ================================================================
d = zeros(length(v_true))
d[1] = -0.5   # push a_Q[1,1] down (q1 further below q2) — enlarges gap
d[7] = +0.5   # push a_Q[1,3] up (q3 further above q2) — enlarges gap
d[10] = -0.1  # push a_init[1] down
d[12] = +0.1  # push a_init[3] up
run_case("never-leave-cone direction (α should be α_cap)", v_true, d)

println("\n", "="^70)
println("  DONE")
println("="^70)

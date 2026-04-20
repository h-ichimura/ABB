#=
run_misspec_comparison.jl — HPC job: MLE vs C²-QR vs ABB-QR under AR(2) misspecification.

Usage: julia run_misspec_comparison.jl <N> <T> <seed> <rho1> <rho2> <sigma_v>

DGP: y_t = ρ₁ y_{t-1} + ρ₂ y_{t-2} + σ_v v_t,  v_t ~ N(0,1)
All three estimators are misspecified (assume AR(1) latent + meas error).
Metrics: KS and L1 of implied f(y_t|y_{t-1}) vs analytical truth.
=#

using Serialization, Printf, Random, LinearAlgebra, Statistics

if length(ARGS) < 6
    error("Usage: julia run_misspec_comparison.jl <N> <T> <seed> <rho1> <rho2> <sigma_v>")
end

N      = parse(Int, ARGS[1])
T_obs  = parse(Int, ARGS[2])
seed   = parse(Int, ARGS[3])
ρ1     = parse(Float64, ARGS[4])
ρ2     = parse(Float64, ARGS[5])
σ_v    = parse(Float64, ARGS[6])

include("../cspline_abb.jl")
using Optim, LineSearches

K = 2; σy = 1.0; τ = [0.25, 0.50, 0.75]

@printf("Misspec comparison: N=%d, T=%d, seed=%d, ρ₁=%.2f, ρ₂=%.2f, σ_v=%.2f\n",
        N, T_obs, seed, ρ1, ρ2, σ_v); flush(stdout)

# ================================================================
#  DATA GENERATION: AR(2) in y
# ================================================================

function generate_ar2(N::Int, T::Int, ρ1::Float64, ρ2::Float64, σ_v::Float64;
                      seed::Int=42, burnin::Int=200)
    rng = MersenneTwister(seed)
    y = zeros(N, T)
    for i in 1:N
        y_prev2 = 0.0; y_prev1 = 0.0
        for b in 1:burnin
            y_new = ρ1 * y_prev1 + ρ2 * y_prev2 + σ_v * randn(rng)
            y_prev2 = y_prev1; y_prev1 = y_new
        end
        y[i,1] = ρ1 * y_prev1 + ρ2 * y_prev2 + σ_v * randn(rng)
        if T >= 2
            y[i,2] = ρ1 * y[i,1] + ρ2 * y_prev1 + σ_v * randn(rng)
        end
        for t in 3:T
            y[i,t] = ρ1 * y[i,t-1] + ρ2 * y[i,t-2] + σ_v * randn(rng)
        end
    end
    y
end

# ================================================================
#  TRUE CONDITIONAL DENSITY: f(y_t | y_{t-1}) for stationary AR(2)
#  Analytically: integrate out y_{t-2} from joint normal.
# ================================================================

function true_cond_density_ar2(y_grid::Vector{Float64}, y_lag::Float64,
                                ρ1::Float64, ρ2::Float64, σ_v::Float64)
    # Stationary AR(2) autocovariances via Yule-Walker
    denom = (1.0 - ρ2) * ((1.0 + ρ2)^2 - ρ1^2)
    if denom <= 0.0
        # Non-stationary: use wide Gaussian fallback
        σ_wide = 3.0
        return [exp(-0.5*(yv/σ_wide)^2)/(σ_wide*sqrt(2π)) for yv in y_grid]
    end
    γ0 = σ_v^2 / denom
    γ1 = ρ1 * γ0 / (1.0 - ρ2)

    # y_{t-2} | y_{t-1} ~ N(γ1/γ0 × y_{t-1}, γ0 - γ1²/γ0)
    μ_cond = (γ1 / γ0) * y_lag
    σ²_cond = max(γ0 - γ1^2 / γ0, 1e-10)

    # f(y_t | y_{t-1}) = N(ρ₁ y_{t-1} + ρ₂ μ_cond, σ_v² + ρ₂² σ²_cond)
    μ_yt = ρ1 * y_lag + ρ2 * μ_cond
    σ_yt = sqrt(σ_v^2 + ρ2^2 * σ²_cond)

    [exp(-0.5*((yv - μ_yt)/σ_yt)^2) / (σ_yt*sqrt(2π)) for yv in y_grid]
end

# ================================================================
#  ESTIMATED CONDITIONAL DENSITIES
# ================================================================

# C² spline: f̂(y|y_lag) = ∫ f̂_trans(η|y_lag) f̂_ε(y-η) dη
function est_cond_cspline(y_grid::Vector{Float64}, y_lag::Float64,
                           a_Q, M_Q, a_eps1, a_eps3, M_eps; n_eta=500)
    z = y_lag / 1.0
    hv = [1.0, z, z^2 - 1.0]
    t_loc = [dot(view(a_Q,:,l), hv) for l in 1:3]
    (t_loc[2] <= t_loc[1] || t_loc[3] <= t_loc[2]) && return zeros(length(y_grid))

    s = zeros(3); βL=Ref(0.0); βR=Ref(0.0); κ1=Ref(0.0); κ3=Ref(0.0)
    solve_cspline_c2!(s, βL, βR, κ1, κ3, t_loc, τ, M_Q)
    lr = max(s[1], s[2], s[3]); m = zeros(4)
    cspline_masses!(m, t_loc, s, βL[], βR[], κ1[], κ3[], lr)
    C_t = sum(m); C_t < 1e-300 && return zeros(length(y_grid))

    a_eps = [a_eps1, 0.0, a_eps3]
    se = zeros(3); βLe=Ref(0.0); βRe=Ref(0.0); κ1e=Ref(0.0); κ3e=Ref(0.0)
    solve_cspline_c2!(se, βLe, βRe, κ1e, κ3e, a_eps, τ, M_eps)
    lre = max(se[1], se[2], se[3]); me = zeros(4)
    cspline_masses!(me, a_eps, se, βLe[], βRe[], κ1e[], κ3e[], lre)
    C_e = sum(me); C_e < 1e-300 && return zeros(length(y_grid))

    η_lo = t_loc[1] - 5.0/sqrt(max(-M_Q, 0.1))
    η_hi = t_loc[3] + 5.0/sqrt(max(-M_Q, 0.1))
    dη = (η_hi - η_lo) / (n_eta - 1)

    f = zeros(length(y_grid))
    for (iy, yv) in enumerate(y_grid)
        val = 0.0
        for ig in 1:n_eta
            ηv = η_lo + (ig-1)*dη
            f_trans = exp(cspline_eval(ηv, t_loc, s, βL[], βR[], κ1[], κ3[]) - lr) / C_t
            f_eps = exp(cspline_eval(yv - ηv, a_eps, se, βLe[], βRe[], κ1e[], κ3e[]) - lre) / C_e
            val += f_trans * f_eps * dη
        end
        f[iy] = val
    end
    f
end

# ABB uniform: same convolution but with piecewise uniform densities
function est_cond_uniform(y_grid::Vector{Float64}, y_lag::Float64,
                           a_Q, a_eps1, a_eps3; n_eta=500)
    z = y_lag / 1.0
    hv = [1.0, z, z^2 - 1.0]
    t_loc = [dot(view(a_Q,:,l), hv) for l in 1:3]
    (t_loc[2] <= t_loc[1] || t_loc[3] <= t_loc[2]) && return zeros(length(y_grid))

    a_eps = [a_eps1, 0.0, a_eps3]
    (a_eps[2] <= a_eps[1] || a_eps[3] <= a_eps[2]) && return zeros(length(y_grid))

    h1_t = t_loc[2] - t_loc[1]; h2_t = t_loc[3] - t_loc[2]
    η_lo = t_loc[1] - 5.0*h1_t
    η_hi = t_loc[3] + 5.0*h2_t
    dη = (η_hi - η_lo) / (n_eta - 1)

    f = zeros(length(y_grid))
    for (iy, yv) in enumerate(y_grid)
        val = 0.0
        for ig in 1:n_eta
            ηv = η_lo + (ig-1)*dη
            f_trans = exp(abb_uniform_logf(ηv, t_loc))
            f_eps = exp(abb_uniform_logf(yv - ηv, a_eps))
            val += f_trans * f_eps * dη
        end
        f[iy] = val
    end
    f
end

function compute_ks_l1(f_est, f_true, dy)
    fe = f_est ./ max(sum(f_est)*dy, 1e-300)
    ft = f_true ./ max(sum(f_true)*dy, 1e-300)
    ks = maximum(abs.(cumsum(fe)*dy .- cumsum(ft)*dy))
    l1 = sum(abs.(fe .- ft)) * dy
    ks, l1
end

# ================================================================
#  GENERATE DATA
# ================================================================

y = generate_ar2(N, T_obs, ρ1, ρ2, σ_v; seed=seed)
@printf("  Data: y range [%.2f, %.2f]\n", minimum(y), maximum(y)); flush(stdout)

# Initial values for estimation (use generic starting point)
tp = make_true_cspline(rho=0.8)
v_init = pack_profiled(tp.a_Q, tp.a_init, tp.a_eps1, tp.a_eps3)

# ================================================================
#  ESTIMATE: Profiled MLE
# ================================================================

@printf("  Profiled MLE...\n"); flush(stdout)
ml_failed = false
t_ml = @elapsed begin
    try
        global v_ml, nll_ml
        v_ml, nll_ml = estimate_profiled_ml(y, K, σy, v_init, τ; G=201, maxiter=500)
    catch e
        @printf("    MLE failed: %s\n", e); flush(stdout)
        global ml_failed = true
        global v_ml = copy(v_init)
        global nll_ml = NaN
    end
end
if !ml_failed
    aQ_ml, MQ_ml, ai_ml, Mi_ml, ae1_ml, ae3_ml, Me_ml = unpack_profiled(v_ml, K)
    @printf("    ρ=%.4f  ae3=%.4f  time=%.0fs\n", aQ_ml[2,2], ae3_ml, t_ml); flush(stdout)
else
    aQ_ml, MQ_ml, ai_ml, Mi_ml, ae1_ml, ae3_ml, Me_ml = unpack_profiled(v_init, K)
end

# ================================================================
#  ESTIMATE: C² spline QR
# ================================================================

@printf("  C² QR...\n"); flush(stdout)
t_c2 = @elapsed begin
    qr_c2 = estimate_cspline_qr(y, K, σy, tp.a_Q, tp.a_init,
                                  tp.a_eps1, tp.a_eps3, τ;
                                  G=201, S_em=30, M_draws=10,
                                  verbose=false, seed=seed)
end
M_Q_c2 = _M_from_iqr(qr_c2.a_Q[1,3] - qr_c2.a_Q[1,1])
@printf("    ρ=%.4f  ae3=%.4f  time=%.0fs\n", qr_c2.a_Q[2,2], qr_c2.a_eps3, t_c2); flush(stdout)

# ================================================================
#  ESTIMATE: ABB uniform QR
# ================================================================

@printf("  ABB QR...\n"); flush(stdout)
t_abb = @elapsed begin
    qr_abb = estimate_abb_uniform_qr(y, K, σy, tp.a_Q, tp.a_init,
                                      tp.a_eps1, tp.a_eps3, τ;
                                      G=201, S_em=30, M_draws=10,
                                      verbose=false, seed=seed)
end
@printf("    ρ=%.4f  ae3=%.4f  time=%.0fs\n", qr_abb.a_Q[2,2], qr_abb.a_eps3, t_abb); flush(stdout)

# ================================================================
#  DENSITY EVALUATION: KS and L1 at 5 conditioning values
# ================================================================

@printf("  Density evaluation...\n"); flush(stdout)
y_eval = collect(range(-5.0, 5.0, length=1000))
dy = y_eval[2] - y_eval[1]
y_cond_vals = [-1.5, -0.5, 0.0, 0.5, 1.5]
n_cond = length(y_cond_vals)

ks_ml = zeros(n_cond); l1_ml = zeros(n_cond)
ks_c2 = zeros(n_cond); l1_c2 = zeros(n_cond)
ks_abb = zeros(n_cond); l1_abb = zeros(n_cond)

for (j, y_lag) in enumerate(y_cond_vals)
    f_true = true_cond_density_ar2(y_eval, y_lag, ρ1, ρ2, σ_v)

    # MLE
    f_ml = est_cond_cspline(y_eval, y_lag, aQ_ml, MQ_ml, ae1_ml, ae3_ml, Me_ml)
    ks_ml[j], l1_ml[j] = compute_ks_l1(f_ml, f_true, dy)

    # C² QR
    f_c2 = est_cond_cspline(y_eval, y_lag, qr_c2.a_Q, M_Q_c2,
                             qr_c2.a_eps1, qr_c2.a_eps3, qr_c2.M_eps)
    ks_c2[j], l1_c2[j] = compute_ks_l1(f_c2, f_true, dy)

    # ABB QR
    f_abb = est_cond_uniform(y_eval, y_lag, qr_abb.a_Q, qr_abb.a_eps1, qr_abb.a_eps3)
    ks_abb[j], l1_abb[j] = compute_ks_l1(f_abb, f_true, dy)
end

@printf("    KS avg: MLE=%.4f  C²=%.4f  ABB=%.4f\n", mean(ks_ml), mean(ks_c2), mean(ks_abb))
@printf("    L1 avg: MLE=%.4f  C²=%.4f  ABB=%.4f\n", mean(l1_ml), mean(l1_c2), mean(l1_abb))
flush(stdout)

# ================================================================
#  SAVE
# ================================================================

summary = Dict{Symbol, Any}(
    :N => N, :T => T_obs, :seed => seed,
    :rho1 => ρ1, :rho2 => ρ2, :sigma_v => σ_v,
    # MLE
    :ml_v => v_ml, :ml_nll => nll_ml, :ml_time => t_ml, :ml_failed => ml_failed,
    :ml_ks => ks_ml, :ml_l1 => l1_ml,
    # C² QR
    :c2_a_Q => qr_c2.a_Q, :c2_a_init => qr_c2.a_init,
    :c2_a_eps1 => qr_c2.a_eps1, :c2_a_eps3 => qr_c2.a_eps3,
    :c2_M_Q => M_Q_c2, :c2_M_eps => qr_c2.M_eps,
    :c2_time => t_c2,
    :c2_ks => ks_c2, :c2_l1 => l1_c2,
    # ABB QR
    :abb_a_Q => qr_abb.a_Q, :abb_a_init => qr_abb.a_init,
    :abb_a_eps1 => qr_abb.a_eps1, :abb_a_eps3 => qr_abb.a_eps3,
    :abb_time => t_abb,
    :abb_ks => ks_abb, :abb_l1 => l1_abb,
    # Conditioning values
    :y_cond_vals => y_cond_vals,
)

outfile = "misspec_r1$(ρ1)_r2$(ρ2)_N$(N)_T$(T_obs)_seed$(seed).jls"
serialize(outfile, summary)
@printf("Done. Saved to %s\n", outfile)

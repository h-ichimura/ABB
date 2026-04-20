#=
run_profiled_comparison.jl — HPC job: Profiled MLE vs QR on same data.

Usage: julia run_profiled_comparison.jl <N> <T> <seed>

Data: rejection sampling (exact model draws)
MLE: profiled (14 params, M from IQR), analytical gradient
QR: stochastic EM, estimates M from IQR
Louis: missing information at truth (one per job)
=#

using Serialization, Printf

if length(ARGS) < 3
    error("Usage: julia run_profiled_comparison.jl <N> <T> <seed> [rho]")
end

N    = parse(Int, ARGS[1])
T    = parse(Int, ARGS[2])
seed = parse(Int, ARGS[3])
ρ_val = length(ARGS) >= 4 ? parse(Float64, ARGS[4]) : 0.8

include("../cspline_abb.jl")
include("../louis_missing_info.jl")
using Optim, LineSearches

K = 2; σy = 1.0; τ = [0.25, 0.50, 0.75]
tp = make_true_cspline(rho=ρ_val)

@printf("Profiled comparison: N=%d, T=%d, seed=%d, rho=%.2f\n", N, T, seed, ρ_val); flush(stdout)

# Generate data (rejection sampling)
y, eta = generate_data_cspline(N, tp.a_Q, tp.M_Q,
                                tp.a_init, tp.M_init,
                                tp.a_eps1, tp.a_eps3, tp.M_eps,
                                K, σy, τ; seed=seed, T=T)

# ---- Profiled MLE (14 params, analytical gradient) ----
v_true = pack_profiled(tp.a_Q, tp.a_init, tp.a_eps1, tp.a_eps3)

@printf("  Profiled MLE...\n"); flush(stdout)
t_ml = @elapsed begin
    v_ml, nll_ml = estimate_profiled_ml(y, K, σy, v_true, τ;
                                         G=201, maxiter=500)
end
aQ_ml, MQ_ml, ai_ml, Mi_ml, ae1_ml, ae3_ml, Me_ml = unpack_profiled(v_ml, K)
@printf("    ρ=%.4f  ae3=%.4f  time=%.0fs\n", aQ_ml[2,2], ae3_ml, t_ml); flush(stdout)

# ---- QR (estimates M from IQR) ----
@printf("  QR...\n"); flush(stdout)
t_qr = @elapsed begin
    qr_est = estimate_cspline_qr(y, K, σy, tp.a_Q,
                                   tp.a_init,
                                   tp.a_eps1, tp.a_eps3, τ;
                                   G=201, S_em=30, M_draws=10,
                                   verbose=false, seed=seed)
end
@printf("    ρ=%.4f  ae3=%.4f  time=%.0fs\n",
        qr_est.a_Q[2,2], qr_est.a_eps3, t_qr); flush(stdout)

# ---- Louis missing information (at truth, first seed only) ----
louis_result = nothing
if seed == 1
    @printf("  Louis missing info (seed=1 only)...\n"); flush(stdout)
    t_louis = @elapsed begin
        louis_result = compute_missing_info(v_true, y, K, σy, τ; G=201)
    end
    @printf("    done in %.0fs\n", t_louis); flush(stdout)
end

# ---- Save results ----
summary = Dict{Symbol, Any}(
    :N => N, :T => T, :seed => seed, :rho => ρ_val,
    # MLE
    :ml_v => v_ml, :ml_nll => nll_ml, :ml_time => t_ml,
    :ml_a_Q => aQ_ml, :ml_M_Q => MQ_ml,
    :ml_a_init => ai_ml, :ml_M_init => Mi_ml,
    :ml_a_eps1 => ae1_ml, :ml_a_eps3 => ae3_ml, :ml_M_eps => Me_ml,
    # QR
    :qr_a_Q => qr_est.a_Q, :qr_time => t_qr,
    :qr_a_init => qr_est.a_init,
    :qr_a_eps1 => qr_est.a_eps1, :qr_a_eps3 => qr_est.a_eps3,
    :qr_M_Q => qr_est.M_Q, :qr_M_eps => qr_est.M_eps,
    # Louis (seed 1 only)
    :louis_Var_S => louis_result,
    # True values
    :a_Q_true => tp.a_Q, :M_Q_true => tp.M_Q,
    :a_init_true => tp.a_init, :M_init_true => tp.M_init,
    :a_eps1_true => tp.a_eps1, :a_eps3_true => tp.a_eps3, :M_eps_true => tp.M_eps,
)

outfile = "profiled_r$(ρ_val)_N$(N)_T$(T)_seed$(seed).jls"
serialize(outfile, summary)
@printf("Done. Saved to %s\n", outfile)

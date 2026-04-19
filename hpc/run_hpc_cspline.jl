#=
run_hpc_cspline.jl — Single HPC task: fit one dataset with both MLE and QR.

Usage: julia run_hpc_cspline.jl <N> <seed> [G]

C² cubic spline log-density with κ_mean parameterization.
Same data for both methods.
=#

using Serialization, Printf

if length(ARGS) < 2
    error("Usage: julia run_hpc_cspline.jl <N> <seed> [G]")
end

N    = parse(Int, ARGS[1])
seed = parse(Int, ARGS[2])
G    = length(ARGS) >= 3 ? parse(Int, ARGS[3]) : 201

include("../cspline_abb.jl")
using Optim, LineSearches

K = 2; σy = 1.0; τ = [0.25, 0.50, 0.75]
tp = make_true_cspline()

@printf("HPC cspline: N=%d, seed=%d, G=%d\n", N, seed, G); flush(stdout)

# Generate data
y, eta = generate_data_cspline(N, tp.a_Q, tp.M_Q,
                                tp.a_init, tp.M_init,
                                tp.a_eps1, tp.a_eps3, tp.M_eps,
                                K, σy, τ; seed=seed)

# ---- MLE (LBFGS, warm start from true params) ----
v_true = pack_cspline(tp.a_Q, tp.M_Q, tp.a_init, tp.M_init,
                       tp.a_eps1, tp.a_eps3, tp.M_eps)

t_ml = @elapsed begin
    v_opt, nll = estimate_cspline_ml(y, K, σy, v_true, τ;
                                      G=G, maxiter=200, verbose=false)
end
a_Q_ml, MQ_ml, ainit_ml, Mi_ml, ae1_ml, ae3_ml, Me_ml = unpack_cspline(v_opt, K)

@printf("  MLE: nll=%.4f  ρ=%.4f  M_Q=%.4f  time=%.1fs\n",
        nll, a_Q_ml[2,2], MQ_ml, t_ml); flush(stdout)

# ---- QR (stochastic EM with FFBS E-step, warm start from true params) ----
t_qr = @elapsed begin
    qr_est = estimate_cspline_qr(y, K, σy, tp.a_Q, tp.M_Q,
                                   tp.a_init, tp.M_init,
                                   tp.a_eps1, tp.a_eps3, tp.M_eps, τ;
                                   G=G, S_em=30, M_draws=10,
                                   verbose=false, seed=seed)
end

@printf("  QR:  ρ=%.4f  time=%.1fs\n",
        qr_est.a_Q[2,2], t_qr); flush(stdout)

# ---- Save results ----
summary = Dict{Symbol, Any}(
    :N => N, :seed => seed, :G => G,
    # MLE results
    :ml_v_opt => v_opt, :ml_nll => nll, :ml_time => t_ml,
    :ml_a_Q => a_Q_ml, :ml_M_Q => MQ_ml,
    :ml_a_init => ainit_ml, :ml_M_init => Mi_ml,
    :ml_a_eps1 => ae1_ml, :ml_a_eps3 => ae3_ml, :ml_M_eps => Me_ml,
    # QR results
    :qr_a_Q => qr_est.a_Q, :qr_time => t_qr,
    :qr_a_init => qr_est.a_init,
    :qr_a_eps1 => qr_est.a_eps1, :qr_a_eps3 => qr_est.a_eps3,
    # True values
    :a_Q_true => tp.a_Q, :M_Q_true => tp.M_Q,
    :a_init_true => tp.a_init, :M_init_true => tp.M_init,
    :a_eps1_true => tp.a_eps1, :a_eps3_true => tp.a_eps3, :M_eps_true => tp.M_eps,
)

outfile = "cspline_N$(N)_seed$(seed).jls"
serialize(outfile, summary)
@printf("Done. Saved to %s\n", outfile)

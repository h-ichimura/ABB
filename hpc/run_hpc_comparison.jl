#=
run_hpc_comparison.jl — Compare Grid MLE, SML, and QR on same data.

Usage: julia run_hpc_comparison.jl <N> <seed> [G] [R_sml]

Data generated with rejection sampling (exact draws from model).
Grid MLE: Boole's rule, numerical gradient.
SML: importance-sampling-based, numerical gradient.
QR: stochastic EM, estimates M from IQR.
=#

using Serialization, Printf

if length(ARGS) < 2
    error("Usage: julia run_hpc_comparison.jl <N> <seed> [G] [R_sml]")
end

N     = parse(Int, ARGS[1])
seed  = parse(Int, ARGS[2])
G     = length(ARGS) >= 3 ? parse(Int, ARGS[3]) : 201
R_sml = length(ARGS) >= 4 ? parse(Int, ARGS[4]) : 500

include("../cspline_abb.jl")
using Optim, LineSearches

K = 2; σy = 1.0; τ = [0.25, 0.50, 0.75]
tp = make_true_cspline()

@printf("HPC comparison: N=%d, seed=%d, G=%d, R_sml=%d\n", N, seed, G, R_sml); flush(stdout)

# Generate data (rejection sampling — exact draws from model)
y, eta = generate_data_cspline(N, tp.a_Q, tp.M_Q,
                                tp.a_init, tp.M_init,
                                tp.a_eps1, tp.a_eps3, tp.M_eps,
                                K, σy, τ; seed=seed)

v_true = pack_cspline(tp.a_Q, tp.M_Q, tp.a_init, tp.M_init,
                       tp.a_eps1, tp.a_eps3, tp.M_eps)

# ---- Grid MLE ----
@printf("  Grid MLE (G=%d)...\n", G); flush(stdout)
t_ml = @elapsed begin
    v_ml, nll_ml = estimate_cspline_ml(y, K, σy, v_true, τ;
                                        G=G, maxiter=500, verbose=false,
                                        use_analytical_grad=false)
end
aQ_ml, MQ_ml, ainit_ml, Mi_ml, ae1_ml, ae3_ml, Me_ml = unpack_cspline(v_ml, K)
@printf("    nll=%.4f  ρ=%.4f  ae3=%.4f  M_Q=%.1f  M_eps=%.2f  time=%.0fs\n",
        nll_ml, aQ_ml[2,2], ae3_ml, MQ_ml, Me_ml, t_ml); flush(stdout)

# ---- SML ----
@printf("  SML (R=%d)...\n", R_sml); flush(stdout)
np = length(v_true)
function sml_obj(v)
    aQ,MQ,ai,Mi,ae1,ae3,Me = unpack_cspline(v, K)
    val = cspline_neg_loglik_sml(aQ, MQ, ai, Mi, ae1, ae3, Me, y, K, σy, τ; R=R_sml)
    isinf(val) ? 1e10 : val
end
function sml_grad!(g, v)
    h = 1e-3; vt = copy(v)
    @inbounds for j in 1:np
        vt[j]=v[j]+h; fp=sml_obj(vt)
        vt[j]=v[j]-h; fm=sml_obj(vt)
        vt[j]=v[j]; g[j]=(fp-fm)/(2h)
    end
end
t_sml = @elapsed begin
    res_sml = optimize(sml_obj, sml_grad!, v_true,
                        LBFGS(; linesearch=LineSearches.BackTracking()),
                        Optim.Options(iterations=200, g_tol=1e-8))
    v_sml = Optim.minimizer(res_sml)
    nll_sml = Optim.minimum(res_sml)
end
aQ_sml, MQ_sml, ainit_sml, Mi_sml, ae1_sml, ae3_sml, Me_sml = unpack_cspline(v_sml, K)
@printf("    nll=%.4f  ρ=%.4f  ae3=%.4f  M_Q=%.1f  M_eps=%.2f  time=%.0fs  iters=%d\n",
        nll_sml, aQ_sml[2,2], ae3_sml, MQ_sml, Me_sml, t_sml,
        Optim.iterations(res_sml)); flush(stdout)

# ---- QR ----
@printf("  QR (S_em=30, M_draws=10)...\n"); flush(stdout)
t_qr = @elapsed begin
    qr_est = estimate_cspline_qr(y, K, σy, tp.a_Q,
                                   tp.a_init,
                                   tp.a_eps1, tp.a_eps3, τ;
                                   G=G, S_em=30, M_draws=10,
                                   verbose=false, seed=seed)
end
@printf("    ρ=%.4f  ae3=%.4f  M_Q=%.1f  M_eps=%.2f  time=%.0fs\n",
        qr_est.a_Q[2,2], qr_est.a_eps3, qr_est.M_Q, qr_est.M_eps, t_qr); flush(stdout)

# ---- Save results ----
summary = Dict{Symbol, Any}(
    :N => N, :seed => seed, :G => G, :R_sml => R_sml,
    # Grid MLE
    :ml_v => v_ml, :ml_nll => nll_ml, :ml_time => t_ml,
    :ml_a_Q => aQ_ml, :ml_M_Q => MQ_ml,
    :ml_a_init => ainit_ml, :ml_M_init => Mi_ml,
    :ml_a_eps1 => ae1_ml, :ml_a_eps3 => ae3_ml, :ml_M_eps => Me_ml,
    # SML
    :sml_v => v_sml, :sml_nll => nll_sml, :sml_time => t_sml,
    :sml_a_Q => aQ_sml, :sml_M_Q => MQ_sml,
    :sml_a_init => ainit_sml, :sml_M_init => Mi_sml,
    :sml_a_eps1 => ae1_sml, :sml_a_eps3 => ae3_sml, :sml_M_eps => Me_sml,
    # QR
    :qr_a_Q => qr_est.a_Q, :qr_time => t_qr,
    :qr_a_init => qr_est.a_init,
    :qr_a_eps1 => qr_est.a_eps1, :qr_a_eps3 => qr_est.a_eps3,
    :qr_M_Q => qr_est.M_Q, :qr_M_eps => qr_est.M_eps,
    # True values
    :a_Q_true => tp.a_Q, :M_Q_true => tp.M_Q,
    :a_init_true => tp.a_init, :M_init_true => tp.M_init,
    :a_eps1_true => tp.a_eps1, :a_eps3_true => tp.a_eps3, :M_eps_true => tp.M_eps,
)

outfile = "comparison_N$(N)_seed$(seed).jls"
serialize(outfile, summary)
@printf("Done. Saved to %s\n", outfile)

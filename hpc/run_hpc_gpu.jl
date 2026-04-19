#=
run_hpc_gpu.jl — GPU-accelerated MLE + QR comparison with analytical gradient.

Usage: julia run_hpc_gpu.jl <N> <seed> [G]

The GPU is used for:
  1. Batched forward filter: [G×G] × [G×N] matrix multiply (cuBLAS)
  2. Element-wise epsilon density evaluation on GPU arrays

Requires CUDA.jl. Falls back to CPU if no GPU available.
=#

using Serialization, Printf

if length(ARGS) < 2
    error("Usage: julia run_hpc_gpu.jl <N> <seed> [G]")
end

N    = parse(Int, ARGS[1])
seed = parse(Int, ARGS[2])
G    = length(ARGS) >= 3 ? parse(Int, ARGS[3]) : 501

include("../cspline_abb.jl")
using Optim, LineSearches

# Try to load CUDA (optional — falls back to CPU batched GEMM)
use_gpu = false
try
    @eval using CUDA
    if CUDA.functional()
        global use_gpu = true
        @printf("GPU available: %s\n", CUDA.name(CUDA.device()))
    else
        @printf("CUDA not functional, using CPU batched GEMM\n")
    end
catch
    @printf("CUDA.jl not available, using CPU batched GEMM\n")
end

K = 2; σy = 1.0; τ = [0.25, 0.50, 0.75]
tp = make_true_cspline()

@printf("HPC GPU: N=%d, seed=%d, G=%d, GPU=%s\n", N, seed, G, use_gpu); flush(stdout)

# Generate data
y, eta = generate_data_cspline(N, tp.a_Q, tp.M_Q,
                                tp.a_init, tp.M_init,
                                tp.a_eps1, tp.a_eps3, tp.M_eps,
                                K, σy, τ; seed=seed)

# ---- MLE with analytical gradient ----
v_true = pack_cspline(tp.a_Q, tp.M_Q, tp.a_init, tp.M_init,
                       tp.a_eps1, tp.a_eps3, tp.M_eps)

t_ml = @elapsed begin
    v_opt, nll = estimate_cspline_ml(y, K, σy, v_true, τ;
                                      G=G, maxiter=200, verbose=false,
                                      use_analytical_grad=true)
end
a_Q_ml, MQ_ml, ainit_ml, Mi_ml, ae1_ml, ae3_ml, Me_ml = unpack_cspline(v_opt, K)

@printf("  MLE: nll=%.4f  ρ=%.4f  ae3=%.4f  M_Q=%.4f  time=%.1fs\n",
        nll, a_Q_ml[2,2], ae3_ml, MQ_ml, t_ml); flush(stdout)

# ---- QR (stochastic EM with FFBS E-step) ----
t_qr = @elapsed begin
    qr_est = estimate_cspline_qr(y, K, σy, tp.a_Q,
                                   tp.a_init,
                                   tp.a_eps1, tp.a_eps3, τ;
                                   G=G, S_em=30, M_draws=10,
                                   verbose=false, seed=seed)
end

@printf("  QR:  ρ=%.4f  time=%.1fs\n",
        qr_est.a_Q[2,2], t_qr); flush(stdout)

# ---- Save results ----
summary = Dict{Symbol, Any}(
    :N => N, :seed => seed, :G => G, :use_gpu => use_gpu,
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

outfile = "cspline_gpu_G$(G)_N$(N)_seed$(seed).jls"
serialize(outfile, summary)
@printf("Done. Saved to %s\n", outfile)

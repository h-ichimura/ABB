#=
run_abb_pw_gpu.jl — GPU-accelerated per-seed driver for the ABB
piecewise-uniform Monte Carlo.  Drop-in replacement for
run_abb_pw_comparison.jl.

Usage: julia run_abb_pw_gpu.jl <N> <T> <seed> [rho]

Bottlenecks (per-likelihood G×G×N forward filter; per-FFBS the same)
are moved onto a single GPU via cuBLAS GEMM.  Backward sampling stays
on CPU because it's per-panel sequential and not the cost driver.

Output filename matches the CPU driver exactly:
  abbpw_r${rho}_N${N}_T${T}_seed${seed}.jls
so result files are interchangeable with CPU results — important for
mixed-pipeline collection.
=#

using Serialization, Printf

if length(ARGS) < 3
    error("Usage: julia run_abb_pw_gpu.jl <N> <T> <seed> [rho]")
end

N    = parse(Int,     ARGS[1])
T    = parse(Int,     ARGS[2])
seed = parse(Int,     ARGS[3])
ρ_val = length(ARGS) >= 4 ? parse(Float64, ARGS[4]) : 0.8

# CUDA must be loaded BEFORE include of cspline_abb_gpu.jl
using CUDA
if !CUDA.functional()
    error("CUDA is not functional. Check `module load cuda` and the GPU node.")
end
@printf("GPU: %s, free=%.1f GiB / total=%.1f GiB\n",
        CUDA.name(CUDA.device()),
        CUDA.available_memory() / 2^30,
        CUDA.total_memory() / 2^30); flush(stdout)

include("../ABB_three_period.jl")
include("../exact_ml.jl")
include("../cspline_abb.jl")
include("../cspline_abb_gpu.jl")

K = 2; σy = 1.0; τ = [0.25, 0.50, 0.75]; L = length(τ)
par_true = make_true_params_linear(tau = τ, sigma_y = σy, K = K,
                                    rho      = ρ_val,
                                    sigma_v  = 0.5,
                                    sigma_eps = 0.3,
                                    sigma_eta1 = 1.0)

@printf("ABB-PW GPU: N=%d T=%d seed=%d rho=%.2f\n", N, T, seed, ρ_val)
flush(stdout)

# ---- DGP (CPU; rejection-free inverse-CDF sampling) ----
y, eta = generate_data_abb(N, par_true, τ, σy, K; seed = seed)

cfg_ml = Config(N, T, K, L, τ, σy, 1, 1, 1, fill(0.05, T))
par0 = copy_params(par_true)

# ---- Exact MLE on GPU ----
@printf("  Exact MLE (GPU)...\n"); flush(stdout)
t_ml = @elapsed begin
    par_ml = estimate_exact_ml_gpu(y, cfg_ml, par0;
                                    G = 201, grid_min = -8.0, grid_max = 8.0,
                                    maxiter = 500, verbose = false)
end
nll_ml = exact_neg_loglik_gpu(par_ml, y, cfg_ml;
                               grid_min = -8.0, grid_max = 8.0, G = 201)
@printf("    ρ=%.4f  a_eps[3]=%.4f  nll/N=%.4f  time=%.0fs\n",
        par_ml.a_Q[2,2], par_ml.a_eps[3], nll_ml, t_ml); flush(stdout)

# ---- QR with GPU FFBS ----
@printf("  ABB QR (GPU FFBS, S_em=30, M_draws=10)...\n"); flush(stdout)
t_qr = @elapsed begin
    qr_est = estimate_abb_uniform_qr_gpu(y, K, σy,
                                          par_true.a_Q,
                                          par_true.a_init,
                                          par_true.a_eps[1],
                                          par_true.a_eps[3],
                                          τ;
                                          G = 201, S_em = 30, M_draws = 10,
                                          verbose = false, seed = seed)
end
@printf("    ρ=%.4f  a_eps[3]=%.4f  time=%.0fs\n",
        qr_est.a_Q[2,2], qr_est.a_eps3, t_qr); flush(stdout)

# ---- Save (identical schema to run_abb_pw_comparison.jl) ----
summary = Dict{Symbol, Any}(
    :N => N, :T => T, :seed => seed, :rho => ρ_val,
    :ml_a_Q => par_ml.a_Q, :ml_b1_Q => par_ml.b1_Q, :ml_bL_Q => par_ml.bL_Q,
    :ml_a_init => par_ml.a_init,
    :ml_b1_init => par_ml.b1_init, :ml_bL_init => par_ml.bL_init,
    :ml_a_eps => par_ml.a_eps,
    :ml_b1_eps => par_ml.b1_eps, :ml_bL_eps => par_ml.bL_eps,
    :ml_nll => nll_ml, :ml_time => t_ml,
    :qr_a_Q => qr_est.a_Q, :qr_a_init => qr_est.a_init,
    :qr_a_eps1 => qr_est.a_eps1, :qr_a_eps3 => qr_est.a_eps3,
    :qr_M_Q => qr_est.M_Q, :qr_M_eps => qr_est.M_eps,
    :qr_time => t_qr,
    :a_Q_true => par_true.a_Q, :a_init_true => par_true.a_init,
    :a_eps1_true => par_true.a_eps[1], :a_eps3_true => par_true.a_eps[3],
    :b1_Q_true => par_true.b1_Q, :bL_Q_true => par_true.bL_Q,
    :b1_init_true => par_true.b1_init, :bL_init_true => par_true.bL_init,
    :b1_eps_true => par_true.b1_eps, :bL_eps_true => par_true.bL_eps,
)

outfile = "abbpw_r$(ρ_val)_N$(N)_T$(T)_seed$(seed).jls"
serialize(outfile, summary)
@printf("Done. Saved to %s\n", outfile)

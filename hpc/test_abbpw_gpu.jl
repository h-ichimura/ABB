#=
test_abbpw_gpu.jl — STRICT CPU↔GPU agreement check, layered.

Layer 1: forward-filter log-likelihood at a fixed θ.  No optimisation, no
         stochasticity.  Disagreement here can ONLY come from GEMM
         floating-point rounding, so we expect relative diff < 1e-10.

Layer 2: full MLE (LBFGS).  Optimisation is deterministic, but tiny FP
         differences in the gradient can shift the converged minimum
         slightly.  Expect param diff < 1e-4, nll relative diff < 1e-8.

Layer 3: full QR (stochastic EM).  Both versions share MersenneTwister(seed),
         so if the forward filter agrees bit-equally (which it won't,
         because of GEMM rounding), the FFBS draws would be identical.
         In practice tiny CDF differences may flip a single backward
         draw at boundary cases — averaged over S_em*M_draws=300 draws,
         the parameter estimates still agree at ~1e-3 typically.

Pass = all three layers within their tolerance.

Usage:
  julia hpc/test_abbpw_gpu.jl [N] [T] [seed] [rho]

Defaults: N=200, T=3, seed=1, rho=0.8.

Exits non-zero on failure.  Prints all numerical diffs unconditionally so
you can see the actual agreement level even on PASS.
=#

using Printf, Statistics

N    = length(ARGS) >= 1 ? parse(Int,     ARGS[1]) : 200
T    = length(ARGS) >= 2 ? parse(Int,     ARGS[2]) : 3
seed = length(ARGS) >= 3 ? parse(Int,     ARGS[3]) : 1
ρ_val = length(ARGS) >= 4 ? parse(Float64, ARGS[4]) : 0.8

const TOL_FF_REL      = 1e-10  # forward filter at fixed θ, relative
const TOL_MLE_PARAM   = 1e-4
const TOL_MLE_NLL_REL = 1e-8
const TOL_QR_PARAM    = 5e-3   # stochastic EM tolerance

using CUDA
CUDA.functional() || error("CUDA not functional on this machine.")
@printf("GPU device: %s\n", CUDA.name(CUDA.device()))

include("../ABB_three_period.jl")
include("../exact_ml.jl")
include("../cspline_abb.jl")
include("../cspline_abb_gpu.jl")

K = 2; σy = 1.0; τ = [0.25, 0.50, 0.75]; L = length(τ)
par_true = make_true_params_linear(tau = τ, sigma_y = σy, K = K,
                                    rho = ρ_val,
                                    sigma_v = 0.5,
                                    sigma_eps = 0.3,
                                    sigma_eta1 = 1.0)
y, _ = generate_data_abb(N, par_true, τ, σy, K; seed = seed)
cfg = Config(N, T, K, L, τ, σy, 1, 1, 1, fill(0.05, T))

@printf("\nConfig: N=%d T=%d seed=%d ρ=%.2f\n\n", N, T, seed, ρ_val)

fail = false

# ============================================================
# LAYER 1 : forward filter at truth — pure FP comparison
# ============================================================
println("="^60)
println("Layer 1 — forward filter at fixed θ (no optimisation)")
println("="^60)

t1cpu = @elapsed nll_cpu_truth = exact_neg_loglik(par_true, y, cfg;
                                                   grid_min = -8.0,
                                                   grid_max = 8.0, G = 201)
t1gpu = @elapsed nll_gpu_truth = exact_neg_loglik_gpu(par_true, y, cfg;
                                                       grid_min = -8.0,
                                                       grid_max = 8.0, G = 201)
δ_ff = abs(nll_cpu_truth - nll_gpu_truth) / max(abs(nll_cpu_truth), 1e-300)

@printf("  CPU nll(θ_true) = %.16f   (%.2f s)\n", nll_cpu_truth, t1cpu)
@printf("  GPU nll(θ_true) = %.16f   (%.2f s)\n", nll_gpu_truth, t1gpu)
@printf("  |Δnll|/|nll|    = %.3e   tol=%.0e   %s\n",
        δ_ff, TOL_FF_REL, δ_ff < TOL_FF_REL ? "OK" : "FAIL")
δ_ff >= TOL_FF_REL && (fail = true)

# Also a perturbed θ — broader exercise of forward filter
par_pert = copy_params(par_true)
par_pert.a_Q[2, :] .*= 0.9
nll_cpu_pert = exact_neg_loglik(par_pert, y, cfg; G = 201,
                                 grid_min = -8.0, grid_max = 8.0)
nll_gpu_pert = exact_neg_loglik_gpu(par_pert, y, cfg; G = 201,
                                     grid_min = -8.0, grid_max = 8.0)
δ_ff2 = abs(nll_cpu_pert - nll_gpu_pert) / max(abs(nll_cpu_pert), 1e-300)
@printf("  At 0.9× slopes:\n")
@printf("    |Δnll|/|nll|  = %.3e   tol=%.0e   %s\n",
        δ_ff2, TOL_FF_REL, δ_ff2 < TOL_FF_REL ? "OK" : "FAIL")
δ_ff2 >= TOL_FF_REL && (fail = true)

# ============================================================
# LAYER 2 : full MLE
# ============================================================
println()
println("="^60)
println("Layer 2 — full MLE (LBFGS)")
println("="^60)

t2cpu = @elapsed par_ml_cpu = estimate_exact_ml(y, cfg, copy_params(par_true);
                                                 G = 201, grid_min = -8.0,
                                                 grid_max = 8.0,
                                                 maxiter = 500, verbose = false)
nll_cpu_opt = exact_neg_loglik(par_ml_cpu, y, cfg;
                                grid_min = -8.0, grid_max = 8.0, G = 201)

t2gpu = @elapsed par_ml_gpu = estimate_exact_ml_gpu(y, cfg, copy_params(par_true);
                                                     G = 201, grid_min = -8.0,
                                                     grid_max = 8.0,
                                                     maxiter = 500, verbose = false)
nll_gpu_opt = exact_neg_loglik_gpu(par_ml_gpu, y, cfg;
                                    grid_min = -8.0, grid_max = 8.0, G = 201)

δ_aQ      = maximum(abs.(par_ml_cpu.a_Q   .- par_ml_gpu.a_Q))
δ_ainit   = maximum(abs.(par_ml_cpu.a_init .- par_ml_gpu.a_init))
δ_aeps    = maximum(abs.(par_ml_cpu.a_eps  .- par_ml_gpu.a_eps))
δ_b1Q     = abs(par_ml_cpu.b1_Q     - par_ml_gpu.b1_Q)
δ_bLQ     = abs(par_ml_cpu.bL_Q     - par_ml_gpu.bL_Q)
δ_b1init  = abs(par_ml_cpu.b1_init  - par_ml_gpu.b1_init)
δ_bLinit  = abs(par_ml_cpu.bL_init  - par_ml_gpu.bL_init)
δ_b1eps   = abs(par_ml_cpu.b1_eps   - par_ml_gpu.b1_eps)
δ_bLeps   = abs(par_ml_cpu.bL_eps   - par_ml_gpu.bL_eps)
δ_nll_rel = abs(nll_cpu_opt - nll_gpu_opt) / max(abs(nll_cpu_opt), 1e-300)

@printf("  Time CPU=%.1fs  GPU=%.1fs  →  speedup=%.1f×\n",
        t2cpu, t2gpu, t2cpu / max(t2gpu, 1e-9))
@printf("  CPU nll @ opt = %.10f\n", nll_cpu_opt)
@printf("  GPU nll @ opt = %.10f\n", nll_gpu_opt)
@printf("  |Δnll|/|nll|  = %.3e   tol=%.0e   %s\n",
        δ_nll_rel, TOL_MLE_NLL_REL, δ_nll_rel < TOL_MLE_NLL_REL ? "OK" : "FAIL")
@printf("  |Δa_Q|max     = %.3e   tol=%.0e   %s\n",
        δ_aQ,    TOL_MLE_PARAM, δ_aQ    < TOL_MLE_PARAM ? "OK" : "FAIL")
@printf("  |Δa_init|max  = %.3e   tol=%.0e   %s\n",
        δ_ainit, TOL_MLE_PARAM, δ_ainit < TOL_MLE_PARAM ? "OK" : "FAIL")
@printf("  |Δa_eps|max   = %.3e   tol=%.0e   %s\n",
        δ_aeps,  TOL_MLE_PARAM, δ_aeps  < TOL_MLE_PARAM ? "OK" : "FAIL")
@printf("  |Δb1_Q|       = %.3e   tol=%.0e   %s\n",
        δ_b1Q,   TOL_MLE_PARAM, δ_b1Q   < TOL_MLE_PARAM ? "OK" : "FAIL")
@printf("  |ΔbL_Q|       = %.3e   tol=%.0e   %s\n",
        δ_bLQ,   TOL_MLE_PARAM, δ_bLQ   < TOL_MLE_PARAM ? "OK" : "FAIL")
@printf("  |Δb1_init|    = %.3e   tol=%.0e   %s\n",
        δ_b1init,TOL_MLE_PARAM, δ_b1init< TOL_MLE_PARAM ? "OK" : "FAIL")
@printf("  |ΔbL_init|    = %.3e   tol=%.0e   %s\n",
        δ_bLinit,TOL_MLE_PARAM, δ_bLinit< TOL_MLE_PARAM ? "OK" : "FAIL")
@printf("  |Δb1_eps|     = %.3e   tol=%.0e   %s\n",
        δ_b1eps, TOL_MLE_PARAM, δ_b1eps < TOL_MLE_PARAM ? "OK" : "FAIL")
@printf("  |ΔbL_eps|     = %.3e   tol=%.0e   %s\n",
        δ_bLeps, TOL_MLE_PARAM, δ_bLeps < TOL_MLE_PARAM ? "OK" : "FAIL")

for δ in (δ_aQ, δ_ainit, δ_aeps, δ_b1Q, δ_bLQ, δ_b1init, δ_bLinit, δ_b1eps, δ_bLeps)
    δ >= TOL_MLE_PARAM && (fail = true)
end
δ_nll_rel >= TOL_MLE_NLL_REL && (fail = true)

# ============================================================
# LAYER 3 : full QR
# ============================================================
println()
println("="^60)
println("Layer 3 — full QR (stochastic EM)")
println("="^60)

t3cpu = @elapsed qr_cpu = estimate_abb_uniform_qr(y, K, σy,
                                                    par_true.a_Q, par_true.a_init,
                                                    par_true.a_eps[1], par_true.a_eps[3],
                                                    τ;
                                                    G = 201, S_em = 30, M_draws = 10,
                                                    verbose = false, seed = seed)
t3gpu = @elapsed qr_gpu = estimate_abb_uniform_qr_gpu(y, K, σy,
                                                       par_true.a_Q, par_true.a_init,
                                                       par_true.a_eps[1], par_true.a_eps[3],
                                                       τ;
                                                       G = 201, S_em = 30, M_draws = 10,
                                                       verbose = false, seed = seed)

δ_qr_aQ   = maximum(abs.(qr_cpu.a_Q   .- qr_gpu.a_Q))
δ_qr_init = maximum(abs.(qr_cpu.a_init .- qr_gpu.a_init))
δ_qr_e1   = abs(qr_cpu.a_eps1 - qr_gpu.a_eps1)
δ_qr_e3   = abs(qr_cpu.a_eps3 - qr_gpu.a_eps3)

@printf("  Time CPU=%.1fs  GPU=%.1fs  →  speedup=%.1f×\n",
        t3cpu, t3gpu, t3cpu / max(t3gpu, 1e-9))
@printf("  |Δa_Q|max     = %.3e   tol=%.0e   %s\n",
        δ_qr_aQ,   TOL_QR_PARAM, δ_qr_aQ   < TOL_QR_PARAM ? "OK" : "FAIL")
@printf("  |Δa_init|max  = %.3e   tol=%.0e   %s\n",
        δ_qr_init, TOL_QR_PARAM, δ_qr_init < TOL_QR_PARAM ? "OK" : "FAIL")
@printf("  |Δa_eps1|     = %.3e   tol=%.0e   %s\n",
        δ_qr_e1,   TOL_QR_PARAM, δ_qr_e1   < TOL_QR_PARAM ? "OK" : "FAIL")
@printf("  |Δa_eps3|     = %.3e   tol=%.0e   %s\n",
        δ_qr_e3,   TOL_QR_PARAM, δ_qr_e3   < TOL_QR_PARAM ? "OK" : "FAIL")

for δ in (δ_qr_aQ, δ_qr_init, δ_qr_e1, δ_qr_e3)
    δ >= TOL_QR_PARAM && (fail = true)
end

# ============================================================
println()
println("="^60)
if fail
    println("VALIDATION FAILED  —  do not submit GPU jobs.")
    exit(1)
else
    println("VALIDATION PASSED  —  GPU output matches CPU within tolerance.")
end

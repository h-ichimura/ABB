#=
run_hpc.jl — Single HPC task: fit one dataset with QR and Exact ML.

Usage: julia run_hpc.jl <N> <seed>

Skip EM-MLE (too slow, similar results to Exact ML).
Use QR warm start + Exact ML with LBFGS.
=#

using Serialization, Printf, Statistics

if length(ARGS) < 2
    error("Usage: julia run_hpc.jl <N> <seed>")
end

N    = parse(Int, ARGS[1])
seed = parse(Int, ARGS[2])

include("../ABB_three_period.jl")
include("../exact_ml.jl")

K = 2; L = 3; sigma_y = 1.0; T = 3
tau = collect(range(1/(L+1), stop=L/(L+1), length=L))
par_true = make_true_params_linear(tau=tau, sigma_y=sigma_y, K=K)

@printf("HPC run: N=%d, seed=%d\n", N, seed)

# Generate data
y, eta_true = generate_data_abb(N, par_true, tau, sigma_y, K; seed=seed)
vp = fill(0.05, T)

# ---- QR: short EM (S=50 enough for QR to converge to fixed point) ----
S_qr = 50; M_em = 20
cfg_qr = Config(N, T, K, L, tau, sigma_y, S_qr, 100, M_em, vp)
par_qr = init_params(y, cfg_qr)
eta_all = zeros(N, T, M_em)
for m in 1:M_em; eta_all[:, :, m] .= 0.6 .* y; end

# Simple QR loop (avoid run_comparison overhead)
t_qr = @elapsed begin
    for iter in 1:S_qr
        e_step!(eta_all, y, par_qr, cfg_qr)
        m_step_qr!(par_qr, eta_all, y, cfg_qr)
    end
end

# ---- Exact ML (LBFGS, warm-started from QR) ----
cfg_exact = Config(N, T, K, L, tau, sigma_y, 1, 1, 1, vp)
t_exact = @elapsed begin
    par_exact = estimate_exact_ml(y, cfg_exact, par_qr;
                                  G=80, maxiter=100, verbose=false)
end

# ---- Save summary ----
summary = Dict{Symbol, Any}()

for (meth, par) in [(:qr, par_qr), (:exact, par_exact)]
    summary[meth] = (
        slope = [par.a_Q[2, l] / sigma_y for l in 1:L],
        intercept = [par.a_Q[1, l] for l in 1:L],
        a_init = copy(par.a_init),
        a_eps = copy(par.a_eps),
        b1_init = par.b1_init, bL_init = par.bL_init,
        b1_eps = par.b1_eps, bL_eps = par.bL_eps,
        b1_Q = par.b1_Q, bL_Q = par.bL_Q,
    )
end
summary[:time_qr] = t_qr
summary[:time_exact] = t_exact

serialize("summary_N$(N)_seed$(seed).jls", summary)
@printf("N=%d seed=%d: QR %.1fs, Exact %.1fs\n", N, seed, t_qr, t_exact)
@printf("  QR slopes:    [%.4f, %.4f, %.4f]\n", summary[:qr].slope...)
@printf("  Exact slopes: [%.4f, %.4f, %.4f]\n", summary[:exact].slope...)

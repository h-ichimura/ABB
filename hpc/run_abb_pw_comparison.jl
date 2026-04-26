#=
run_abb_pw_comparison.jl — HPC job: MLE and QR on same data, both
                          under the *original ABB piecewise-uniform*
                          density specification.

Usage: julia run_abb_pw_comparison.jl <N> <T> <seed> [rho]

This is the PW-model analog of run_profiled_comparison.jl:
  * DGP              : ABB piecewise-uniform with exponential tails
                        (generate_data_abb → exact ABB draws via inverse CDF)
  * MLE (grid)       : exact MLE via forward-filter, no smoothing
                        (estimate_exact_ml from exact_ml.jl)
  * QR (stochastic EM): ABB-uniform FFBS E-step + QR M-step
                        (estimate_abb_uniform_qr from cspline_abb.jl)

(SML was originally included but removed: it is not used in the smooth
 comparison either, and an N=200 sanity check showed a 66% bias in
 a_eps[3] together with an objective lower than exact MLE — i.e. it was
 not optimising the same likelihood. Re-add only after the SML
 implementation is debugged.)

The MLE optimises the full PW parameter vector (a_Q, b_Q, a_init,
b_init, a_eps with a_eps[2]=0, b_eps). QR estimates a_Q, a_init, and
(a_eps1, a_eps3); tail rates come from the IQR.

Output is a .jls dictionary analogous to profiled_*.jls so it can be
digested by the same collect / plot scripts (see collect_abb_pw.jl and
plot_abb_pw_*.jl).
=#

using Serialization, Printf

if length(ARGS) < 3
    error("Usage: julia run_abb_pw_comparison.jl <N> <T> <seed> [rho]")
end

N    = parse(Int, ARGS[1])
T    = parse(Int, ARGS[2])
seed = parse(Int, ARGS[3])
ρ_val = length(ARGS) >= 4 ? parse(Float64, ARGS[4]) : 0.8

include("../ABB_three_period.jl")   # Params, Config, make_true_params_linear, generate_data_abb, e_step!
include("../exact_ml.jl")           # estimate_exact_ml, exact_neg_loglik
include("../cspline_abb.jl")        # estimate_abb_uniform_qr

K = 2; σy = 1.0; τ = [0.25, 0.50, 0.75]; L = length(τ)
par_true = make_true_params_linear(tau=τ, sigma_y=σy, K=K,
                                    rho=ρ_val, sigma_v=0.5,
                                    sigma_eps=0.3, sigma_eta1=1.0)

@printf("ABB-PW comparison: N=%d, T=%d, seed=%d, rho=%.2f\n",
        N, T, seed, ρ_val); flush(stdout)

# ---- Data: rejection-free inverse-CDF sampling from the ABB PW model ----
y, eta = generate_data_abb(N, par_true, τ, σy, K; seed=seed)

# Config: one EM outer iter for exact MLE (it's a plain L-BFGS on the
# marginal neg-log-lik).
cfg_ml = Config(N, T, K, L, τ, σy, 1, 1, 1, fill(0.05, T))

# ---- Warm start: run a few EM-QR iterations on ABB PW to get a decent
#      parameter vector for the likelihood-based methods.  Same warm-start
#      convention as the cspline version (run_profiled_comparison.jl uses
#      truth as warm start — here we start from truth too, since we want
#      to measure how well each estimator *recovers* the truth without
#      conflating cold-start dynamics).
par0 = copy_params(par_true)

# ---- Exact MLE on ABB PW likelihood (grid-based forward filter) ----
@printf("  Exact MLE...\n"); flush(stdout)
t_ml = @elapsed begin
    par_ml = estimate_exact_ml(y, cfg_ml, par0;
                                G=201, grid_min=-8.0, grid_max=8.0,
                                maxiter=500, verbose=false)
end
nll_ml = exact_neg_loglik(par_ml, y, cfg_ml;
                           grid_min=-8.0, grid_max=8.0, G=201)
@printf("    ρ=%.4f  a_eps[3]=%.4f  nll/N=%.4f  time=%.0fs\n",
        par_ml.a_Q[2,2], par_ml.a_eps[3], nll_ml, t_ml); flush(stdout)

# ---- QR (stochastic EM with ABB-uniform FFBS E-step) ----
@printf("  ABB QR (S_em=30, M_draws=10)...\n"); flush(stdout)
t_qr = @elapsed begin
    qr_est = estimate_abb_uniform_qr(y, K, σy,
                                      par_true.a_Q,
                                      par_true.a_init,
                                      par_true.a_eps[1], par_true.a_eps[3],
                                      τ;
                                      G=201, S_em=30, M_draws=10,
                                      verbose=false, seed=seed)
end
@printf("    ρ=%.4f  a_eps[3]=%.4f  time=%.0fs\n",
        qr_est.a_Q[2,2], qr_est.a_eps3, t_qr); flush(stdout)

# ---- Save results ----
summary = Dict{Symbol, Any}(
    :N => N, :T => T, :seed => seed, :rho => ρ_val,
    # Exact MLE
    :ml_a_Q => par_ml.a_Q, :ml_b1_Q => par_ml.b1_Q, :ml_bL_Q => par_ml.bL_Q,
    :ml_a_init => par_ml.a_init, :ml_b1_init => par_ml.b1_init, :ml_bL_init => par_ml.bL_init,
    :ml_a_eps => par_ml.a_eps, :ml_b1_eps => par_ml.b1_eps, :ml_bL_eps => par_ml.bL_eps,
    :ml_nll => nll_ml, :ml_time => t_ml,
    # QR
    :qr_a_Q => qr_est.a_Q,
    :qr_a_init => qr_est.a_init,
    :qr_a_eps1 => qr_est.a_eps1, :qr_a_eps3 => qr_est.a_eps3,
    :qr_M_Q => qr_est.M_Q, :qr_M_eps => qr_est.M_eps,
    :qr_time => t_qr,
    # Truth
    :a_Q_true => par_true.a_Q,
    :a_init_true => par_true.a_init,
    :a_eps1_true => par_true.a_eps[1], :a_eps3_true => par_true.a_eps[3],
    :b1_Q_true => par_true.b1_Q, :bL_Q_true => par_true.bL_Q,
    :b1_init_true => par_true.b1_init, :bL_init_true => par_true.bL_init,
    :b1_eps_true => par_true.b1_eps, :bL_eps_true => par_true.bL_eps,
)

outfile = "abbpw_r$(ρ_val)_N$(N)_T$(T)_seed$(seed).jls"
serialize(outfile, summary)
@printf("Done. Saved to %s\n", outfile)

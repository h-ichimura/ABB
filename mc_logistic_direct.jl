#=
mc_logistic_direct.jl — Monte Carlo: QR vs MLE with analytical gradient +
golden-section line search + analytical feasibility bounds.

Estimation:
  1. QR  — EM with logistic E-step + ABB m_step_qr (direct parameterization).
  2. MLE-GS — custom LBFGS (analytical gradient, analytical α_max, golden section
     line search) warm-started from QR.
=#

include("logistic_direct.jl")
using Printf, Statistics, Serialization, Random

"""If QR solution has a quantile crossing, shift its intercepts to restore
ordering with a small buffer. Returns a modified par that satisfies
is_feasible(pack_direct(par), K, σy, -8, 8)."""
function project_to_feasible!(par::Params, K::Int, σy::Float64;
                              η_lo=-8.0, η_hi=8.0, buffer=0.05)
    for ℓ in 1:2
        m = gap_min_analytical(par.a_Q, ℓ, σy; η_lo=η_lo, η_hi=η_hi)
        if m <= 0
            shift = -m + buffer
            par.a_Q[1, ℓ+1] += shift
        end
    end
    # Init and eps orderings
    if par.a_init[1] >= par.a_init[2]
        par.a_init[1] = par.a_init[2] - 0.1
    end
    if par.a_init[2] >= par.a_init[3]
        par.a_init[3] = par.a_init[2] + 0.1
    end
    if par.a_eps[1] >= 0
        par.a_eps[1] = -0.1
    end
    if par.a_eps[3] <= 0
        par.a_eps[3] = 0.1
    end
    par.a_eps[2] = 0.0
    par
end

"""Cold initial point: equal slopes, zero quadratic, ordered intercepts.
Guaranteed feasible because non-constant coefficients are identical across ℓ."""
function cold_start_params(par_true::Params; K::Int=2)
    par0 = copy_params(par_true)
    par0.a_Q[1, :] = [-0.3, 0.0, 0.3]
    par0.a_Q[2, :] .= 0.5
    par0.a_Q[3, :] .= 0.0
    par0.a_init .= [-0.8, 0.0, 0.8]
    par0.a_eps  .= [-0.25, 0.0, 0.25]
    par0
end

function run_one(seed::Int, N::Int, par_true::Params,
                 K::Int, L::Int, σy::Float64, tau::Vector{Float64};
                 G::Int=201, S_em::Int=30, M::Int=20, n_draws::Int=100)
    y, eta = generate_data_direct(N, par_true, K, σy; seed=seed)

    # ---- 1. QR: EM with logistic E-step ----
    par0_qr = copy_params(par_true)
    par0_qr.a_Q[2, :] .= 0.5        # perturbed slopes
    par0_qr.a_Q[1, :] = [par_true.a_Q[1,1]*0.8,
                        par_true.a_Q[1,2],
                        par_true.a_Q[1,3]*0.8]
    t_qr = @elapsed par_qr = estimate_qr_direct(y, K, σy, par0_qr;
                                                S=S_em, M=M, n_draws=n_draws)

    # Record QR state BEFORE projection (for crossing diagnostic)
    cross_qr_raw = check_crossing(par_qr, K, σy)
    # Ensure non-crossing for use as MLE warm start
    project_to_feasible!(par_qr, K, σy)

    # ---- 2a. MLE warm-started from QR ----
    t_mle_w = @elapsed par_mle_w, nll_mle_w = estimate_direct_ml_gs(
        y, K, σy, par_qr;
        G=G, maxiter=60, verbose=false, g_tol=1e-5, ls_tol=1e-5)

    # ---- 2b. MLE cold-started (equal coefs except constants) ----
    par0_cold = cold_start_params(par_true; K=K)
    t_mle_c = @elapsed par_mle_c, nll_mle_c = estimate_direct_ml_gs(
        y, K, σy, par0_cold;
        G=G, maxiter=60, verbose=false, g_tol=1e-5, ls_tol=1e-5)

    nll_qr_at_qr = direct_neg_loglik(par_qr,    y, K, σy; G=G)
    nll_true_d   = direct_neg_loglik(par_true,  y, K, σy; G=G)

    (seed     = seed,
     sl_qr    = par_qr.a_Q[2, :] ./ σy,
     sl_mle_w = par_mle_w.a_Q[2, :] ./ σy,
     sl_mle_c = par_mle_c.a_Q[2, :] ./ σy,
     in_qr    = par_qr.a_Q[1, :],
     in_mle_w = par_mle_w.a_Q[1, :],
     in_mle_c = par_mle_c.a_Q[1, :],
     qd_qr    = par_qr.a_Q[3, :],
     qd_mle_w = par_mle_w.a_Q[3, :],
     qd_mle_c = par_mle_c.a_Q[3, :],
     ae_qr    = par_qr.a_eps,
     ae_mle_w = par_mle_w.a_eps,
     ae_mle_c = par_mle_c.a_eps,
     ai_qr    = par_qr.a_init,
     ai_mle_w = par_mle_w.a_init,
     ai_mle_c = par_mle_c.a_init,
     nll_true  = nll_true_d,
     nll_qr    = nll_qr_at_qr,
     nll_mle_w = nll_mle_w,
     nll_mle_c = nll_mle_c,
     cross_qr_raw = cross_qr_raw,
     cross_qr     = check_crossing(par_qr,    K, σy),
     cross_mle_w  = check_crossing(par_mle_w, K, σy),
     cross_mle_c  = check_crossing(par_mle_c, K, σy),
     t_qr     = t_qr,
     t_mle_w  = t_mle_w,
     t_mle_c  = t_mle_c)
end

function mc_run(; N::Int=500, R::Int=20, K::Int=2, σy::Float64=1.0,
                G::Int=201, S_em::Int=30, M::Int=20, n_draws::Int=100)
    L = 3; tau = [0.25, 0.50, 0.75]
    par_true = make_true_direct(K=K)
    true_sl = par_true.a_Q[2, :] ./ σy
    true_in = par_true.a_Q[1, :]

    @printf("========= MC (direct parameterization) =========\n")
    @printf("N=%d, R=%d, G=%d, S_em=%d, M=%d, n_draws=%d\n",
            N, R, G, S_em, M, n_draws)
    @printf("True slopes:  [%.4f, %.4f, %.4f]\n", true_sl...)
    @printf("True intcpts: [%.4f, %.4f, %.4f]\n\n", true_in...)
    flush(stdout)

    results = []
    for r in 1:R
        res = run_one(r, N, par_true, K, L, σy, tau;
                      G=G, S_em=S_em, M=M, n_draws=n_draws)
        push!(results, res)
        @printf("r=%2d | QR=[%.3f,%.3f,%.3f] MLE_w=[%.3f,%.3f,%.3f] MLE_c=[%.3f,%.3f,%.3f] nll=(truth=%.3f, QR=%.3f, MLE_w=%.3f, MLE_c=%.3f) cross(QR_raw=%d, QR=%d, MLE_w=%d, MLE_c=%d) (%.0fs/%.0fs/%.0fs)\n",
                r, res.sl_qr..., res.sl_mle_w..., res.sl_mle_c...,
                res.nll_true, res.nll_qr, res.nll_mle_w, res.nll_mle_c,
                res.cross_qr_raw, res.cross_qr, res.cross_mle_w, res.cross_mle_c,
                res.t_qr, res.t_mle_w, res.t_mle_c)
        flush(stdout)
    end

    # Summaries
    function summarize(name, arr, truth)
        m = mean(arr, dims=1)[:]
        b = m .- truth
        s = std(arr, dims=1)[:]
        rm = sqrt.(mean((arr .- truth').^2, dims=1)[:])
        @printf("  %-8s mean=[%.4f,%.4f,%.4f] bias=[%+.4f,%+.4f,%+.4f] std=[%.4f,%.4f,%.4f] RMSE=[%.4f,%.4f,%.4f]\n",
                name, m..., b..., s..., rm...)
        rm
    end

    sl_qr    = hcat([r.sl_qr    for r in results]...)' |> collect
    sl_mle_w = hcat([r.sl_mle_w for r in results]...)' |> collect
    sl_mle_c = hcat([r.sl_mle_c for r in results]...)' |> collect
    in_qr    = hcat([r.in_qr    for r in results]...)' |> collect
    in_mle_w = hcat([r.in_mle_w for r in results]...)' |> collect
    in_mle_c = hcat([r.in_mle_c for r in results]...)' |> collect

    println("\n", "="^70)
    println("SLOPES (true = [$(join([@sprintf("%.4f",s) for s in true_sl], ", "))])")
    rmse_qr    = summarize("QR",    sl_qr,    true_sl)
    rmse_mle_w = summarize("MLE_w", sl_mle_w, true_sl)
    rmse_mle_c = summarize("MLE_c", sl_mle_c, true_sl)
    @printf("  Eff QR/MLE_w  (RMSE ratio): [%.2f, %.2f, %.2f]\n",
            (rmse_qr ./ max.(rmse_mle_w, 1e-10))...)
    @printf("  Eff QR/MLE_c  (RMSE ratio): [%.2f, %.2f, %.2f]\n",
            (rmse_qr ./ max.(rmse_mle_c, 1e-10))...)

    println("\nINTERCEPTS (true = [$(join([@sprintf("%.4f",s) for s in true_in], ", "))])")
    summarize("QR",    in_qr,    true_in)
    summarize("MLE_w", in_mle_w, true_in)
    summarize("MLE_c", in_mle_c, true_in)

    println("\nNEG-LL")
    @printf("  mean nll @ truth: %.4f\n", mean(r.nll_true  for r in results))
    @printf("  mean nll @ QR:    %.4f\n", mean(r.nll_qr    for r in results))
    @printf("  mean nll @ MLE_w: %.4f\n", mean(r.nll_mle_w for r in results))
    @printf("  mean nll @ MLE_c: %.4f\n", mean(r.nll_mle_c for r in results))

    @printf("\nCrossing violations (across %d reps)\n", R)
    @printf("  QR (raw, before projection): %d\n", sum(r.cross_qr_raw for r in results))
    @printf("  QR (after projection):       %d\n", sum(r.cross_qr     for r in results))
    @printf("  MLE_w: %d\n", sum(r.cross_mle_w for r in results))
    @printf("  MLE_c: %d\n", sum(r.cross_mle_c for r in results))

    @printf("\nAvg time/rep: QR=%.1fs, MLE_w=%.1fs, MLE_c=%.1fs, total=%.1fs\n",
            mean(r.t_qr    for r in results),
            mean(r.t_mle_w for r in results),
            mean(r.t_mle_c for r in results),
            sum(r.t_qr + r.t_mle_w + r.t_mle_c for r in results) / R)

    fname = "mc_logistic_direct_N$(N)_R$(R).jls"
    serialize(fname, results)
    @printf("Saved %s\n", fname)
    results
end

mc_run(N=500, R=20)

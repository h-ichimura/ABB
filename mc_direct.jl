#=
mc_direct.jl — Monte Carlo: QR vs MLE with direct parameterization.

No gap reparameterization. No sorting. No piecewise-uniform.
All densities are asymmetric logistic.

QR estimates each quantile column independently (ignores cross-percentile restrictions).
MLE estimates all 14 params jointly (exploits cross-percentile restrictions).

MLE should have lower variance for slopes because it knows all three
slopes come from the same conditional mean structure.
=#

include("logistic_direct.jl")
using Printf, Statistics, Serialization

function mc_run(; N=500, R=20, K=2, σy=1.0)
    par_true = make_true_direct()
    true_sl = par_true.a_Q[2, :] ./ σy
    true_in = par_true.a_Q[1, :]

    println("="^70)
    @printf("  MC: QR vs MLE (direct param), N=%d, R=%d\n", N, R)
    println("="^70)
    @printf("True slopes:     [%.4f, %.4f, %.4f]\n", true_sl...)
    @printf("True intercepts: [%.4f, %.4f, %.4f]\n\n", true_in...)

    results = []
    for r in 1:R
        y, η = generate_data_direct(N, par_true, K, σy; seed=r)

        # QR: EM with logistic E-step
        par0 = copy_params(par_true)
        par0.a_Q[2, :] .= 0.5  # equal slopes, inside cone
        t_qr = @elapsed par_qr = estimate_qr_direct(y, K, σy, par0; S=30, M=20)

        # MLE: LBFGS warm-started from QR (not cold start)
        t_mle = @elapsed par_mle, _ = estimate_direct_ml(y, K, σy, par_qr;
                                                          G=81, maxiter=30, verbose=false)

        sl_qr = par_qr.a_Q[2, :] ./ σy
        sl_mle = par_mle.a_Q[2, :] ./ σy
        in_qr = par_qr.a_Q[1, :]
        in_mle = par_mle.a_Q[1, :]
        nc_mle = check_crossing(par_mle, K, σy)

        push!(results, (seed=r, sl_qr=sl_qr, sl_mle=sl_mle,
                        in_qr=in_qr, in_mle=in_mle,
                        t_qr=t_qr, t_mle=t_mle, nc_mle=nc_mle))

        @printf("r=%3d | QR=[%.3f,%.3f,%.3f] MLE=[%.3f,%.3f,%.3f] cross=%d (%.0f+%.0fs)\n",
                r, sl_qr..., sl_mle..., nc_mle, t_qr, t_mle)
        flush(stdout)
    end

    # Summary
    println("\n", "="^70)
    println("  RESULTS")
    println("="^70)

    sl_qr_mat = hcat([r.sl_qr for r in results]...)' |> collect
    sl_mle_mat = hcat([r.sl_mle for r in results]...)' |> collect
    in_qr_mat = hcat([r.in_qr for r in results]...)' |> collect
    in_mle_mat = hcat([r.in_mle for r in results]...)' |> collect

    function show_summary(name, est, truth)
        m = mean(est, dims=1)[:]
        s = std(est, dims=1)[:]
        b = m .- truth
        rmse = sqrt.(mean((est .- truth').^2, dims=1)[:])
        @printf("  %-6s mean=[%.4f,%.4f,%.4f] bias=[%+.4f,%+.4f,%+.4f] std=[%.4f,%.4f,%.4f] RMSE=[%.4f,%.4f,%.4f]\n",
                name, m..., b..., s..., rmse...)
    end

    println("\n-- SLOPES (true = [$(join([@sprintf("%.4f",s) for s in true_sl], ","))]) --")
    show_summary("QR", sl_qr_mat, true_sl)
    show_summary("MLE", sl_mle_mat, true_sl)
    eff = sqrt.(mean((sl_qr_mat .- true_sl').^2, dims=1)[:]) ./
          sqrt.(mean((sl_mle_mat .- true_sl').^2, dims=1)[:])
    @printf("  Efficiency (QR RMSE / MLE RMSE): [%.2f, %.2f, %.2f]\n", eff...)

    println("\n-- INTERCEPTS (true = [$(join([@sprintf("%.4f",s) for s in true_in], ","))]) --")
    show_summary("QR", in_qr_mat, true_in)
    show_summary("MLE", in_mle_mat, true_in)

    n_cross = sum(r.nc_mle > 0 for r in results)
    @printf("\nMLE crossing violations: %d/%d replications\n", n_cross, R)
    @printf("Avg time: QR=%.1fs, MLE=%.1fs\n",
            mean(r.t_qr for r in results), mean(r.t_mle for r in results))

    serialize("mc_direct_N$(N)_R$(R).jls", results)
    @printf("Saved mc_direct_N%d_R%d.jls\n", N, R)
    results
end

if abspath(PROGRAM_FILE) == @__FILE__
    mc_run(N=500, R=20)
end

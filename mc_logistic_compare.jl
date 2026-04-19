#=
mc_logistic_compare.jl — Monte Carlo comparison of QR vs Logistic MLE
on the quantile-respecting logistic DGP.

Hypothesis: Logistic MLE exploits cross-percentile restrictions (common slope
across τ, symmetric intercepts, etc.), yielding smaller variance than QR
which estimates each quantile column independently.
=#

include("logistic_ml.jl")
using Printf, Statistics, Serialization

function run_one_replication(seed::Int, N::Int, p_true::LogisticParams,
                              K::Int, σy::Float64, τ::Vector{Float64})
    y, η = generate_data_logistic(N, p_true, K, σy; seed=seed)
    L = length(τ)

    # Logistic MLE
    p0 = copy_params_log(p_true)
    p0.μ_Q[2] = 0.5       # perturb slope from 0.8
    p0.α_Q[1] = 3.0        # perturb scale
    t_mle = @elapsed p_mle = estimate_logistic_ml(y, K, σy, p0; G=80,
                                                    maxiter=100, verbose=false)

    # Translate MLE to implied ABB-style a_Q
    a_Q_mle = logistic_to_aQ(p_mle, τ, K, σy)

    # QR via ABB EM
    t_qr = @elapsed par_qr = estimate_qr_on_logistic(y, K, L, σy, τ;
                                                       S=50, M=20, verbose=false)

    (seed=seed,
     μ_Q_mle=p_mle.μ_Q, α_Q_mle=p_mle.α_Q,
     slopes_mle=a_Q_mle[2, :], intcpts_mle=a_Q_mle[1, :], quad_mle=a_Q_mle[3, :],
     slopes_qr=par_qr.a_Q[2, :], intcpts_qr=par_qr.a_Q[1, :], quad_qr=par_qr.a_Q[3, :],
     t_mle=t_mle, t_qr=t_qr)
end

function mc_comparison(N::Int, R::Int;
                        K::Int=2, σy::Float64=1.0,
                        τ=[0.25, 0.50, 0.75])
    p_true = make_true_logistic_params()
    true_a_Q = logistic_to_aQ(p_true, τ, K, σy)

    println("="^70)
    @printf("  MC: QR vs Logistic MLE, N=%d, R=%d\n", N, R)
    println("="^70)
    @printf("True μ_Q:    [%.4f, %.4f, %.4f]\n", p_true.μ_Q...)
    @printf("True α_Q:    [%.4f, %.4f, %.4f]\n", p_true.α_Q...)
    @printf("Implied a_Q slopes:    [%.4f, %.4f, %.4f]  (all = 0.8000)\n",
            true_a_Q[2, :]...)
    @printf("Implied a_Q intcpts:   [%.4f, %.4f, %.4f]  (±log(3)/α)\n",
            true_a_Q[1, :]...)
    @printf("Implied a_Q quadratic: [%.4f, %.4f, %.4f]  (all = 0)\n",
            true_a_Q[3, :]...)

    # Run replications
    println("\nRunning replications...")
    results = []
    for r in 1:R
        t = @elapsed res = run_one_replication(r, N, p_true, K, σy, τ)
        push!(results, res)
        @printf("  r=%3d | MLE=[%.4f,%.4f,%.4f], QR=[%.4f,%.4f,%.4f] (%.1fs)\n",
                r, res.slopes_mle..., res.slopes_qr..., t)
    end

    # Summarize
    println("\n", "="^70)
    println("  RESULTS (bias, std, RMSE across R replications)")
    println("="^70)

    function summarize(name, vals_R_L, true_vals)
        means = mean(vals_R_L, dims=1)[:]
        stds = std(vals_R_L, dims=1)[:]
        biases = means .- true_vals
        rmses = sqrt.(mean((vals_R_L .- true_vals').^2, dims=1)[:])
        @printf("  %-8s", name)
        @printf("  mean=[%.4f, %.4f, %.4f]", means...)
        @printf("  bias=[%+.4f, %+.4f, %+.4f]", biases...)
        @printf("  std=[%.4f, %.4f, %.4f]", stds...)
        @printf("  RMSE=[%.4f, %.4f, %.4f]\n", rmses...)
    end

    # Stack slopes_mle, slopes_qr, etc. into R × L matrices
    slopes_mle = hcat([r.slopes_mle for r in results]...)' |> collect
    slopes_qr = hcat([r.slopes_qr for r in results]...)' |> collect
    intcpts_mle = hcat([r.intcpts_mle for r in results]...)' |> collect
    intcpts_qr = hcat([r.intcpts_qr for r in results]...)' |> collect
    quad_mle = hcat([r.quad_mle for r in results]...)' |> collect
    quad_qr = hcat([r.quad_qr for r in results]...)' |> collect

    println("\n-- SLOPES (true = [0.8000, 0.8000, 0.8000]) --")
    summarize("MLE", slopes_mle, true_a_Q[2, :])
    summarize("QR",  slopes_qr,  true_a_Q[2, :])
    @printf("  Efficiency gain (QR RMSE / MLE RMSE): [%.2f, %.2f, %.2f]\n",
            [sqrt(mean((slopes_qr[:,l] .- true_a_Q[2,l]).^2)) /
             sqrt(mean((slopes_mle[:,l] .- true_a_Q[2,l]).^2)) for l in 1:length(τ)]...)

    println("\n-- INTERCEPTS (true = [$(round(true_a_Q[1,1], digits=4)), 0.0000, $(round(true_a_Q[1,3], digits=4))]) --")
    summarize("MLE", intcpts_mle, true_a_Q[1, :])
    summarize("QR",  intcpts_qr,  true_a_Q[1, :])

    println("\n-- QUADRATIC (true = [0, 0, 0]) --")
    summarize("MLE", quad_mle, true_a_Q[3, :])
    summarize("QR",  quad_qr,  true_a_Q[3, :])

    mean_t_mle = mean([r.t_mle for r in results])
    mean_t_qr = mean([r.t_qr for r in results])
    @printf("\nAvg time per replication: MLE=%.1fs, QR=%.1fs\n", mean_t_mle, mean_t_qr)

    results
end

if abspath(PROGRAM_FILE) == @__FILE__
    # Quick run: N=500, R=10 reps
    N = 500; R = 10
    results = mc_comparison(N, R)

    # Save
    serialize("mc_logistic_compare_N$(N)_R$(R).jls", results)
    @printf("\nSaved mc_logistic_compare_N%d_R%d.jls\n", N, R)
end

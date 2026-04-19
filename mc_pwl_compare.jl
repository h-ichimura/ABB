#=
mc_pwl_compare.jl — Monte Carlo: QR vs PWL-z MLE on smooth ABB DGP

DGP: PWL-z logistic (smooth extension of ABB)
  η_t | η_{t-1} ~ PWL-logistic(q(η_{t-1}), α_L(η_{t-1}), α_R(η_{t-1}))
  Same quantile knots as ABB, but smooth density.

Estimation:
  QR: ABB stochastic EM with quantile regression M-step
  MLE: LBFGS on exact likelihood (forward filter), warm-started from truth
=#

include("pwl_abb.jl")
using Printf, Statistics, Serialization

function run_one(seed::Int, N::Int, p_true::PWLParams, K::Int, σy::Float64)
    y, η = generate_data_pwl(N, p_true, K, σy; seed=seed)
    L = 3; tau = [0.25, 0.50, 0.75]

    # MLE: LBFGS from truth (best case for MLE)
    p0_mle = copy_pwl(p_true)
    t_mle = @elapsed p_mle, res_mle = estimate_pwl_ml(y, K, σy, p0_mle;
                                                        G=120, maxiter=100,
                                                        verbose=false)
    nll_mle = pwl_neg_loglik(p_mle, y, K, σy; G=120)

    # QR: ABB stochastic EM
    par_qr = init_params(y, Config(N, 3, K, L, tau, σy, 50, 100, 20, fill(0.05, 3)))
    eta_all = zeros(N, 3, 20)
    for m in 1:20; eta_all[:,:,m] .= 0.6 .* y; end
    cfg_qr = Config(N, 3, K, L, tau, σy, 50, 100, 20, fill(0.05, 3))
    t_qr = @elapsed begin
        for iter in 1:50
            e_step!(eta_all, y, par_qr, cfg_qr)
            m_step_qr!(par_qr, eta_all, y, cfg_qr)
        end
    end

    # Extract implied ABB slopes from PWL-z MLE
    # In gap reparam: a_Q[:,1] = median, a_Q[:,2] = log gap_L, a_Q[:,3] = log gap_U
    # Implied quantile slopes:
    #   slope_median = a_Q[2, 1]
    #   slope_q1 = slope_median (gap is constant in η if a_Q[2,2]=0)
    #   slope_q3 = slope_median
    # For comparison with QR slopes, compute implied quantile slopes at η=0
    hv0 = hermite_vec(0.0, K, σy)
    q1_0, q2_0, q3_0 = cond_q(0.0, p_mle.a_Q, K, σy)

    # Numerical derivative dq/dη at η=0
    dη = 0.001
    q1_p, q2_p, q3_p = cond_q(dη, p_mle.a_Q, K, σy)
    q1_m, q2_m, q3_m = cond_q(-dη, p_mle.a_Q, K, σy)
    slopes_mle = [(q1_p-q1_m)/(2dη), (q2_p-q2_m)/(2dη), (q3_p-q3_m)/(2dη)]
    intcpts_mle = [q1_0, q2_0, q3_0]

    slopes_qr = par_qr.a_Q[2, :] ./ σy
    intcpts_qr = par_qr.a_Q[1, :]

    (seed=seed, slopes_mle=slopes_mle, intcpts_mle=intcpts_mle,
     slopes_qr=slopes_qr, intcpts_qr=intcpts_qr,
     nll_mle=nll_mle, t_mle=t_mle, t_qr=t_qr)
end

function mc_pwl(N::Int, R::Int; K::Int=2, σy::Float64=1.0)
    p_true = make_true_pwl_logistic(; ρ=0.8, σ_v=0.5, σ_eps=0.3, σ_η1=1.0, K=K)
    tau = [0.25, 0.50, 0.75]

    # True implied quantile slopes at η=0
    dη = 0.001
    q1_p, q2_p, q3_p = cond_q(dη, p_true.a_Q, K, σy)
    q1_m, q2_m, q3_m = cond_q(-dη, p_true.a_Q, K, σy)
    true_slopes = [(q1_p-q1_m)/(2dη), (q2_p-q2_m)/(2dη), (q3_p-q3_m)/(2dη)]
    q1_0, q2_0, q3_0 = cond_q(0.0, p_true.a_Q, K, σy)
    true_intcpts = [q1_0, q2_0, q3_0]

    println("="^70)
    @printf("  MC: QR vs PWL-z MLE, N=%d, R=%d\n", N, R)
    println("="^70)
    @printf("True slopes at η=0:    [%.4f, %.4f, %.4f]\n", true_slopes...)
    @printf("True intercepts at η=0:[%.4f, %.4f, %.4f]\n", true_intcpts...)
    flush(stdout)

    results = []
    for r in 1:R
        res = run_one(r, N, p_true, K, σy)
        push!(results, res)
        @printf("  r=%3d | MLE=[%.4f,%.4f,%.4f] QR=[%.4f,%.4f,%.4f] (MLE %.0fs, QR %.0fs)\n",
                r, res.slopes_mle..., res.slopes_qr..., res.t_mle, res.t_qr)
        flush(stdout)
    end

    # Summarize
    L = 3
    slopes_mle = hcat([r.slopes_mle for r in results]...)' |> collect
    slopes_qr  = hcat([r.slopes_qr  for r in results]...)' |> collect
    intcpts_mle = hcat([r.intcpts_mle for r in results]...)' |> collect
    intcpts_qr  = hcat([r.intcpts_qr  for r in results]...)' |> collect

    println("\n","="^70)
    println("  RESULTS")
    println("="^70)

    println("\n-- SLOPES (true = [$(join([@sprintf("%.4f",s) for s in true_slopes], ", "))]) --")
    for (name, ests) in [("MLE", slopes_mle), ("QR", slopes_qr)]
        means = mean(ests, dims=1)[:]
        biases = means .- true_slopes
        stds = std(ests, dims=1)[:]
        rmses = sqrt.(mean((ests .- true_slopes').^2, dims=1)[:])
        @printf("  %-4s mean=[%.4f,%.4f,%.4f] bias=[%+.4f,%+.4f,%+.4f] std=[%.4f,%.4f,%.4f] RMSE=[%.4f,%.4f,%.4f]\n",
                name, means..., biases..., stds..., rmses...)
    end
    rmse_mle = sqrt.(mean((slopes_mle .- true_slopes').^2, dims=1)[:])
    rmse_qr  = sqrt.(mean((slopes_qr  .- true_slopes').^2, dims=1)[:])
    @printf("  Efficiency (QR RMSE / MLE RMSE): [%.2f, %.2f, %.2f]\n",
            (rmse_qr ./ rmse_mle)...)

    println("\n-- INTERCEPTS (true = [$(join([@sprintf("%.4f",s) for s in true_intcpts], ", "))]) --")
    for (name, ests) in [("MLE", intcpts_mle), ("QR", intcpts_qr)]
        means = mean(ests, dims=1)[:]
        biases = means .- true_intcpts
        stds = std(ests, dims=1)[:]
        rmses = sqrt.(mean((ests .- true_intcpts').^2, dims=1)[:])
        @printf("  %-4s mean=[%.4f,%.4f,%.4f] bias=[%+.4f,%+.4f,%+.4f] std=[%.4f,%.4f,%.4f] RMSE=[%.4f,%.4f,%.4f]\n",
                name, means..., biases..., stds..., rmses...)
    end

    @printf("\nAvg time: MLE=%.1fs, QR=%.1fs\n",
            mean(r.t_mle for r in results), mean(r.t_qr for r in results))

    serialize("mc_pwl_N$(N)_R$(R).jls", results)
    @printf("Saved mc_pwl_N%d_R%d.jls\n", N, R)
end

mc_pwl(500, 30)

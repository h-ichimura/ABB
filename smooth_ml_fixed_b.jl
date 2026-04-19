#=
smooth_ml_fixed_b.jl — Exact ML with smooth density, b_coef FIXED

Simpler version: hold spline coefficients b_coef at a reasonable value
that makes the density resemble Gaussian. Only estimate the ABB quantile
parameters (a_Q) and marginals (a_init, a_eps + tails).

This tests whether smoothing the density helps MLE recover the transition
parameters better than with piecewise-uniform.
=#

include("smooth_ml.jl")
include("exact_ml.jl")

function estimate_smooth_fixed_b(y::Matrix{Float64}, cfg::Config, par0::Params,
                                  b_coef::Matrix{Float64};
                                  G::Int=80, maxiter::Int=100, verbose::Bool=true)
    K, L = cfg.K, cfg.L
    par = copy_params(par0)

    # Only optimize over Params (a_Q, a_init, a_eps, tail rates)
    np_aQ = (K + 1) * L
    function params_only_vec(par)
        v = Float64[]
        append!(v, vec(par.a_Q))
        append!(v, par.a_init); push!(v, log(par.b1_init), log(par.bL_init))
        append!(v, par.a_eps);  push!(v, log(par.b1_eps),  log(par.bL_eps))
        v
    end
    function vec_only_to_params!(par, v, K, L)
        par.a_Q .= reshape(view(v, 1:np_aQ), K + 1, L)
        i = np_aQ
        par.a_init .= view(v, i+1:i+L); i += L
        par.b1_init = exp(v[i+1]); par.bL_init = exp(v[i+2]); i += 2
        par.a_eps .= view(v, i+1:i+L); i += L
        par.b1_eps = exp(v[i+1]); par.bL_eps = exp(v[i+2])
    end

    function obj(v)
        vec_only_to_params!(par, v, K, L)
        smooth_neg_loglik(par, b_coef, y, cfg; G=G)
    end

    v0 = params_only_vec(par0)
    verbose && @printf("  Smooth ML (fixed b) initial obj: %.6f\n", obj(v0))

    res = optimize(obj, v0, LBFGS(),
                   Optim.Options(iterations=maxiter, g_tol=1e-5,
                                 show_trace=verbose, show_every=5))
    v_opt = Optim.minimizer(res)
    vec_only_to_params!(par, v_opt, K, L)
    @printf("  Smooth ML final obj: %.6f (iters=%d)\n",
            Optim.minimum(res), Optim.iterations(res))
    par
end

function main_fixed_b()
    K = 2; L = 3; sigma_y = 1.0; T = 3; N = 200
    tau = [0.25, 0.50, 0.75]
    par_true = make_true_params_linear(tau=tau, sigma_y=sigma_y, K=K)
    y, eta_true = generate_data_abb(N, par_true, tau, sigma_y, K; seed=42)
    cfg = Config(N, T, K, L, tau, sigma_y, 1, 1, 1, fill(0.05, T))

    # Fixed b_coef: chosen to make density roughly Gaussian-shaped near truth
    b_coef = init_smooth_b_coef(K, L)

    # QR warm start
    println("Getting QR warm start...")
    par0 = init_params(y, cfg)
    eta_all = zeros(N, T, 20)
    for m in 1:20; eta_all[:,:,m] .= 0.6 .* y; end
    cfg_em = Config(N, T, K, L, tau, sigma_y, 5, 100, 20, fill(0.05, T))
    for _ in 1:5
        e_step!(eta_all, y, par0, cfg_em)
        m_step_qr!(par0, eta_all, y, cfg_em)
    end

    println("\nSmooth ML with b_coef FIXED:")
    t = @elapsed par_sml = estimate_smooth_fixed_b(y, cfg, par0, b_coef;
                                                    G=80, maxiter=50, verbose=true)

    # Also run piecewise-uniform exact ML for comparison
    println("\nPiecewise-uniform exact ML (for comparison):")
    t_pw = @elapsed par_pw = estimate_exact_ml(y, cfg, par0; G=80,
                                                maxiter=50, verbose=false)

    println("\n==== SLOPES ====")
    @printf("True:     [%.4f, %.4f, %.4f]\n", par_true.a_Q[2,:]...)
    @printf("QR:       [%.4f, %.4f, %.4f]\n", par0.a_Q[2,:]...)
    @printf("PW ML:    [%.4f, %.4f, %.4f]\n", par_pw.a_Q[2,:]...)
    @printf("Smooth ML:[%.4f, %.4f, %.4f]\n", par_sml.a_Q[2,:]...)

    println("\n==== INTERCEPTS ====")
    @printf("True:     [%.4f, %.4f, %.4f]\n", par_true.a_Q[1,:]...)
    @printf("QR:       [%.4f, %.4f, %.4f]\n", par0.a_Q[1,:]...)
    @printf("PW ML:    [%.4f, %.4f, %.4f]\n", par_pw.a_Q[1,:]...)
    @printf("Smooth ML:[%.4f, %.4f, %.4f]\n", par_sml.a_Q[1,:]...)

    @printf("\nTime: Smooth %.1fs, PW %.1fs\n", t, t_pw)
end

main_fixed_b()

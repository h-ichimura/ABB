#=
mle_gap.jl — MLE M-step using gap reparameterization

Replace linear Q_l(η) = h(η)'·a_Q[:,l] with:
  Q_2(η)  = h(η)' · a_med               (median, free)
  Q_1(η)  = Q_2(η) - exp(h(η)' · δ_L)  (< Q_2 by construction)
  Q_3(η)  = Q_2(η) + exp(h(η)' · δ_U)  (> Q_2 by construction)

Non-crossing is automatic for all η. Unconstrained optimization.
=#

include("ABB_three_period.jl")
using Optim, LinearAlgebra, Printf

"""
Convert linear a_Q parameterization to gap (a_med, δ_L, δ_U) parameterization.
Uses regression of log(gap) on Hermite basis at observed η_lag points.
"""
function aQ_to_gap(a_Q::Matrix{Float64}, H::Matrix{Float64})
    # a_Q: (K+1) × 3, H: n_obs × (K+1)
    P = size(a_Q, 1)
    a_med = copy(a_Q[:, 2])

    Q_all = H * a_Q                 # n_obs × 3
    gap_L = Q_all[:, 2] .- Q_all[:, 1]  # Q_2 - Q_1
    gap_U = Q_all[:, 3] .- Q_all[:, 2]  # Q_3 - Q_2
    # If QR has crossings, clip to positive
    gap_L = max.(gap_L, 1e-4)
    gap_U = max.(gap_U, 1e-4)
    δ_L = H \ log.(gap_L)
    δ_U = H \ log.(gap_U)
    a_med, δ_L, δ_U
end

"""Compute Q_1, Q_2, Q_3 at all η_lag from gap parameterization."""
function gap_to_Q(a_med, δ_L, δ_U, H::Matrix{Float64})
    Q_med = H * a_med
    Q_low = Q_med .- exp.(H * δ_L)
    Q_high = Q_med .+ exp.(H * δ_U)
    Q_low, Q_med, Q_high
end

"""MLE M-step with gap parameterization."""
function m_step_mle_gap!(par::Params, eta_all::Array{Float64,3},
                          y::Matrix{Float64}, cfg::Config;
                          verbose::Bool=false)
    K, L = cfg.K, cfg.L
    @assert L == 3 "Gap parameterization requires L=3 (τ=0.25, 0.50, 0.75)"
    P = K + 1

    # Marginals (same as QR/MLE: sample quantiles)
    eta1_all = stack_initial(eta_all, cfg)
    for l in 1:L; par.a_init[l] = quantile(eta1_all, cfg.tau[l]); end
    par.b1_init, par.bL_init = update_tails(eta1_all, par.a_init[1], par.a_init[L])

    eps_all = stack_eps(eta_all, y, cfg)
    for l in 1:L; par.a_eps[l] = quantile(eps_all, cfg.tau[l]); end
    par.a_eps .-= mean(par.a_eps)
    par.b1_eps, par.bL_eps = update_tails(eps_all, par.a_eps[1], par.a_eps[L])

    # Transition via gap reparameterization
    eta_t, eta_lag = stack_transition(eta_all, cfg)
    H = hermite_basis(eta_lag, K, cfg.sigma_y)
    n_obs = length(eta_t)

    # Warm start: QR → gap parameterization
    m_step_qr!(par, eta_all, y, cfg)
    a_med0, δ_L0, δ_U0 = aQ_to_gap(par.a_Q, H)
    theta0 = vcat(a_med0, δ_L0, δ_U0)

    # Negative profiled CDLL in gap parameterization
    q_tmp = Vector{Float64}(undef, 3)
    function neg_ll(theta)
        am = view(theta, 1:P)
        dL = view(theta, P+1:2P)
        dU = view(theta, 2P+1:3P)
        Q_low, Q_med, Q_high = gap_to_Q(am, dL, dU, H)

        # Profile tail rates
        r1 = eta_t .- Q_low
        rH = eta_t .- Q_high
        ml = r1 .<= 0; mh = rH .>= 0
        s1 = sum(r1[ml]); sL = sum(rH[mh])
        b1v = s1 < -1e-10 ? -count(ml)/s1 : par.b1_Q
        bLv = sL >  1e-10 ?  count(mh)/sL : par.bL_Q

        # Log-likelihood
        ll = 0.0
        @inbounds for j in 1:n_obs
            q_tmp[1] = Q_low[j]; q_tmp[2] = Q_med[j]; q_tmp[3] = Q_high[j]
            ll += pw_logdens(eta_t[j], q_tmp, cfg.tau, b1v, bLv)
        end
        -ll / n_obs
    end

    # Coordinate descent with 1D Brent line search (no constraints to worry about)
    theta = copy(theta0)
    n_params = length(theta)
    n_cycles = 8
    for cycle in 1:n_cycles
        max_step = 0.0
        # Search widths shrink over cycles; different widths for (a_med, δ_L, δ_U)
        for k in 1:n_params
            cur = theta[k]
            # a_med params (k=1..P), δ parameters (k=P+1..3P) get different widths
            is_med = k <= P
            w = cycle <= 2 ? (is_med ? 0.5 : 0.8) : (cycle <= 5 ? 0.2 : 0.05)
            lo = cur - w; hi = cur + w

            function f1d(v)
                theta[k] = v
                neg_ll(theta)
            end

            res1d = optimize(f1d, lo, hi)
            new_val = Optim.minimizer(res1d)
            max_step = max(max_step, abs(new_val - cur))
            theta[k] = new_val
        end
        verbose && @printf("  gap cycle %d: max step = %.6f\n", cycle, max_step)
        max_step < 1e-6 && break
    end
    theta_opt = theta
    am_opt = theta_opt[1:P]
    dL_opt = theta_opt[P+1:2P]
    dU_opt = theta_opt[2P+1:3P]

    # Compute final Q values and store approximation in par.a_Q
    # Note: Q_1 and Q_3 are NOT linear in h(η) in this parameterization,
    # so storing as a_Q (linear coeffs) is an APPROXIMATION.
    # For downstream code (E-step, full_loglik) to work, we need a consistent
    # representation. Option: project Q onto linear span via regression.
    Q_low, Q_med, Q_high = gap_to_Q(am_opt, dL_opt, dU_opt, H)

    # Linear projection of each Q onto Hermite basis
    par.a_Q[:, 1] .= H \ Q_low
    par.a_Q[:, 2] .= am_opt
    par.a_Q[:, 3] .= H \ Q_high

    # Final tail rates
    Q_recomputed = H * par.a_Q
    r1 = eta_t .- view(Q_recomputed, :, 1); rH = eta_t .- view(Q_recomputed, :, L)
    ml = r1 .<= 0; mh = rH .>= 0
    s1 = sum(r1[ml]); sL = sum(rH[mh])
    s1 < -1e-10 && (par.b1_Q = -count(ml)/s1)
    sL >  1e-10 && (par.bL_Q =  count(mh)/sL)

    neg_ll(theta_opt)
end

# ================================================================
#  QUICK TEST: observed η
# ================================================================
if abspath(PROGRAM_FILE) == @__FILE__
    K = 2; L = 3; sigma_y = 1.0; T = 3; N = 2000
    tau = [0.25, 0.50, 0.75]
    par_true = make_true_params_linear(tau=tau, sigma_y=sigma_y, K=K)

    println("="^70)
    println("  Gap-reparameterized MLE: observed η test (N=$N)")
    println("="^70)
    println("\nTrue: slopes=0.80, intercepts=(-0.337, 0, 0.337)")

    results = []
    for seed in 1:5
        y, eta_true = generate_data_abb(N, par_true, tau, sigma_y, K; seed=seed)
        y_clean = copy(eta_true)
        M = 1
        cfg = Config(N, T, K, L, tau, sigma_y, 1, 1, M, fill(0.05, T))
        eta_all = zeros(N, T, M); eta_all[:, :, 1] .= eta_true

        # QR baseline
        par_qr = copy_params(par_true); par_qr.a_Q .= par_true.a_Q .* 0.5
        par_qr.b1_Q = 3.0; par_qr.bL_Q = 3.0
        t_qr = @elapsed m_step_qr!(par_qr, eta_all, y_clean, cfg)

        # Gap MLE
        par_gap = copy_params(par_true); par_gap.a_Q .= par_true.a_Q .* 0.5
        par_gap.b1_Q = 3.0; par_gap.bL_Q = 3.0
        t_gap = @elapsed m_step_mle_gap!(par_gap, eta_all, y_clean, cfg)

        # Standard MLE (coord descent with feasible intervals)
        par_std = copy_params(par_true); par_std.a_Q .= par_true.a_Q .* 0.5
        par_std.b1_Q = 3.0; par_std.bL_Q = 3.0
        t_std = @elapsed m_step_mle!(par_std, eta_all, y_clean, cfg)

        push!(results, (
            seed=seed,
            qr_slopes=par_qr.a_Q[2,:],
            gap_slopes=par_gap.a_Q[2,:],
            std_slopes=par_std.a_Q[2,:],
            t_qr=t_qr, t_gap=t_gap, t_std=t_std))

        @printf("\nseed=%d:\n", seed)
        @printf("  QR slopes:  [%.4f, %.4f, %.4f]  (t=%.2fs)\n",
                par_qr.a_Q[2,:]..., t_qr)
        @printf("  std MLE:    [%.4f, %.4f, %.4f]  (t=%.2fs)\n",
                par_std.a_Q[2,:]..., t_std)
        @printf("  gap MLE:    [%.4f, %.4f, %.4f]  (t=%.2fs)\n",
                par_gap.a_Q[2,:]..., t_gap)
    end

    # Averages
    println("\n", "="^70)
    println("Averages across 5 seeds:")
    avg_qr = mean(mean(r.qr_slopes) for r in results)
    avg_gap = mean(mean(r.gap_slopes) for r in results)
    avg_std = mean(mean(r.std_slopes) for r in results)
    @printf("  QR avg slope: %.4f  (time %.2fs)\n",
            avg_qr, mean(r.t_qr for r in results))
    @printf("  std avg:      %.4f  (time %.2fs)\n",
            avg_std, mean(r.t_std for r in results))
    @printf("  gap avg:      %.4f  (time %.2fs)\n",
            avg_gap, mean(r.t_gap for r in results))
    @printf("  true: 0.8000\n")
end

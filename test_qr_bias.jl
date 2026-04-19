#=
test_qr_bias.jl — Does QR on posterior draws recover true transition parameters?

Case A: QR on transition draws (η_t, η_{t-1}) — must recover truth.
Case B: QR on posterior draws from E-step with true θ — tests whether
        conditioning on y distorts the quantile regression target.

Vary σ_ε to see how signal-to-noise affects the bias.
=#

include("ABB_three_period.jl")

# ================================================================
#  QR on arbitrary (Y, X) pairs — same as M-step but standalone
# ================================================================
function run_qr(eta_t::Vector{Float64}, eta_lag::Vector{Float64},
                K::Int, sigma_y::Float64, tau::Vector{Float64})
    L = length(tau)
    H = hermite_basis(eta_lag, K, sigma_y)
    a_Q = zeros(K+1, L)
    for l in 1:L
        tau_l = tau[l]
        obj(a) = let r = eta_t .- H*a; mean(r .* (tau_l .- (r.<0))); end
        res = optimize(obj, zeros(K+1), LBFGS(),
                       Optim.Options(iterations=200, g_tol=1e-10, show_trace=false))
        a_Q[:,l] .= Optim.minimizer(res)
    end
    # Tail rates
    r1 = eta_t .- H*a_Q[:,1]; rL = eta_t .- H*a_Q[:,L]
    ml = r1 .<= 0; mh = rL .>= 0
    s1 = sum(r1[ml]); sL = sum(rL[mh])
    b1 = s1 < -1e-10 ? -count(ml)/s1 : 2.0
    bL = sL >  1e-10 ?  count(mh)/sL : 2.0
    a_Q, b1, bL
end

# ================================================================
#  EXPERIMENT
# ================================================================
K = 2; L = 3; sigma_y = 1.0; T = 3
tau = [0.25, 0.50, 0.75]
N = 5000
M = 200        # posterior draws to keep
n_draws = 500  # total MH steps (keep last M)

sigma_eps_values = [0.1, 0.3, 0.6, 1.0]

# Open output file
io = open("results_qr_bias.txt", "w")

function printboth(args...)
    print(stdout, args...); print(io, args...)
end
function printlnboth(args...)
    println(stdout, args...); println(io, args...)
end

printlnboth("="^75)
printlnboth("  QR BIAS TEST: Transition Draws vs Posterior Draws")
printlnboth("  N=$N, M=$M, n_draws=$n_draws, K=$K, L=$L")
printlnboth("="^75)

for sigma_eps in sigma_eps_values
    printlnboth("\n", "-"^75)
    printlnboth("  σ_ε = $sigma_eps")
    printlnboth("-"^75)

    # True parameters for this σ_ε
    par_true = make_true_params_linear(tau=tau, sigma_y=sigma_y, K=K,
                                       sigma_eps=sigma_eps)

    # Generate data
    y, eta_true = generate_data_abb(N, par_true, tau, sigma_y, K; seed=42)

    # Print true parameters
    printboth(@sprintf("  True slopes:     [%.4f, %.4f, %.4f]\n",
                       par_true.a_Q[2,:]...))
    printboth(@sprintf("  True intercepts: [%.4f, %.4f, %.4f]\n",
                       par_true.a_Q[1,:]...))
    printboth(@sprintf("  True b1_Q=%.4f  bL_Q=%.4f\n",
                       par_true.b1_Q, par_true.bL_Q))

    # ============================================================
    #  CASE A: QR on transition draws
    #  Draw η_{t-1} from marginal, then η_t from transition.
    #  No y, no posterior.
    # ============================================================
    rng_A = MersenneTwister(123)
    n_trans = N * (T-1)  # same size as one draw's worth of transition pairs
    # Use eta_true as source of η_{t-1} values (from the DGP marginal)
    eta_lag_A = Float64[]
    eta_t_A   = Float64[]
    q_buf = zeros(L)
    for t in 2:T, i in 1:N
        push!(eta_lag_A, eta_true[i, t-1])
        transition_quantiles!(q_buf, eta_true[i, t-1], par_true.a_Q, K, sigma_y)
        push!(eta_t_A, pw_draw(rng_A, q_buf, tau, par_true.b1_Q, par_true.bL_Q))
    end

    a_Q_A, b1_A, bL_A = run_qr(eta_t_A, eta_lag_A, K, sigma_y, tau)

    printlnboth("\n  Case A: QR on transition draws (no y)")
    printboth(@sprintf("    slopes:     [%.4f, %.4f, %.4f]\n", a_Q_A[2,:]...))
    printboth(@sprintf("    intercepts: [%.4f, %.4f, %.4f]\n", a_Q_A[1,:]...))
    printboth(@sprintf("    b1=%.4f  bL=%.4f\n", b1_A, bL_A))

    # ============================================================
    #  CASE B: QR on posterior draws (E-step with true θ)
    # ============================================================
    vp = fill(0.05, T)
    cfg = Config(N, T, K, L, tau, sigma_y, 1, n_draws, M, vp)

    # Initialize eta_all at 0.6*y
    eta_all = zeros(N, T, M)
    for m in 1:M; eta_all[:,:,m] .= 0.6 .* y; end

    # Run E-step with TRUE parameters
    acc = e_step!(eta_all, y, par_true, cfg)

    # Stack posterior draws
    eta_t_B, eta_lag_B = stack_transition(eta_all, cfg)

    a_Q_B, b1_B, bL_B = run_qr(eta_t_B, eta_lag_B, K, sigma_y, tau)

    printlnboth("\n  Case B: QR on posterior draws (E-step with true θ)")
    printboth(@sprintf("    slopes:     [%.4f, %.4f, %.4f]\n", a_Q_B[2,:]...))
    printboth(@sprintf("    intercepts: [%.4f, %.4f, %.4f]\n", a_Q_B[1,:]...))
    printboth(@sprintf("    b1=%.4f  bL=%.4f\n", b1_B, bL_B))
    printboth(@sprintf("    E-step acceptance: %.2f/%.2f/%.2f\n", acc...))

    # ============================================================
    #  COMPARISON
    # ============================================================
    printlnboth("\n  Bias (Case B - True):")
    printboth(@sprintf("    slope bias:     [%+.4f, %+.4f, %+.4f]\n",
              (a_Q_B[2,:] .- par_true.a_Q[2,:])...))
    printboth(@sprintf("    intercept bias: [%+.4f, %+.4f, %+.4f]\n",
              (a_Q_B[1,:] .- par_true.a_Q[1,:])...))

    printlnboth("\n  Bias (Case A - True):")
    printboth(@sprintf("    slope bias:     [%+.4f, %+.4f, %+.4f]\n",
              (a_Q_A[2,:] .- par_true.a_Q[2,:])...))
    printboth(@sprintf("    intercept bias: [%+.4f, %+.4f, %+.4f]\n",
              (a_Q_A[1,:] .- par_true.a_Q[1,:])...))

    flush(io)
end

printlnboth("\n", "="^75)
printlnboth("  END OF QR BIAS TEST")
printlnboth("="^75)

close(io)
println("\nResults saved to results_qr_bias.txt")

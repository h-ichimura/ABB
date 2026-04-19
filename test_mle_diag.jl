#=
test_mle_diag.jl — Diagnose why MLE has higher ll but worse slopes in full EM

Plan:
1. Generate data with known truth
2. Do one E-step from truth → get η_draws
3. Compare QR vs MLE M-step on those same η_draws
4. Show: parameters, full loglik at each, whether MLE's "better CDLL" means better params
5. Also: compare MLE M-step with and without non-crossing constraint
=#

include("ABB_three_period.jl")
using Printf, Statistics, Random

K = 2; L = 3; sigma_y = 1.0; T = 3
tau = [0.25, 0.50, 0.75]
par_true = make_true_params_linear(tau=tau, sigma_y=sigma_y, K=K)

N = 500; M = 50
cfg = Config(N, T, K, L, tau, sigma_y, 1, 200, M, fill(0.05, T))
y, eta_true = generate_data_abb(N, par_true, tau, sigma_y, K; seed=42)

println("="^70)
println("  MLE DIAGNOSTIC: 1 E-step + 1 M-step (each method)")
println("="^70)

# ── Initialize at truth, do 1 E-step ──────────────────────────────
par = copy_params(par_true)
eta_all = zeros(N, T, M)
for m in 1:M; eta_all[:,:,m] .= 0.6 .* y; end

Random.seed!(123)
acc = e_step!(eta_all, y, par, cfg)
@printf("E-step acc rates: [%s]\n", join([@sprintf("%.3f", a) for a in acc], ", "))

# Stack transition pairs
eta_t, eta_lag = stack_transition(eta_all, cfg)
H = hermite_basis(eta_lag, K, cfg.sigma_y)
n_obs = length(eta_t)
@printf("Transition pairs: n_obs=%d\n", n_obs)

# Helper: CDLL at given a_Q using given eta_t, eta_lag (no sorting)
function cdll_at(a_Q, b1, bL)
    Q = H * a_Q
    ll = 0.0
    for j in 1:n_obs
        ll += pw_logdens(eta_t[j], view(Q, j, :), tau, b1, bL)
    end
    ll / n_obs
end

# Helper: closed-form tail rates given a_Q
function profile_tails(a_Q)
    Q = H * a_Q
    r1 = eta_t .- view(Q, :, 1); rL = eta_t .- view(Q, :, L)
    ml = r1 .<= 0; mh = rL .>= 0
    s1 = sum(r1[ml]); sL = sum(rL[mh])
    b1 = s1 < -1e-10 ? -count(ml)/s1 : 3.0
    bL = sL >  1e-10 ?  count(mh)/sL : 3.0
    b1, bL
end

# True parameters
ll_true = cdll_at(par_true.a_Q, par_true.b1_Q, par_true.bL_Q)
b1t, bLt = profile_tails(par_true.a_Q)
ll_true_prof = cdll_at(par_true.a_Q, b1t, bLt)
println("\n=== Truth ===")
@printf("  slopes: [%.4f, %.4f, %.4f]\n", par_true.a_Q[2,:]...)
@printf("  intercepts: [%.4f, %.4f, %.4f]\n", par_true.a_Q[1,:]...)
@printf("  b1_Q=%.4f bL_Q=%.4f (true tails)\n", par_true.b1_Q, par_true.bL_Q)
@printf("  profiled b1=%.4f bL=%.4f\n", b1t, bLt)
@printf("  CDLL at truth: %.4f (profiled: %.4f)\n", ll_true, ll_true_prof)

# ── QR ────────────────────────────────────────────────────────────
par_qr = copy_params(par_true)  # start from truth for fair comparison
par_qr.a_Q .= par_true.a_Q .* 0.5  # perturb
par_qr.b1_Q = 3.0; par_qr.bL_Q = 3.0
t_qr = @elapsed m_step_qr!(par_qr, eta_all, y, cfg)
b1q, bLq = profile_tails(par_qr.a_Q)
ll_qr_actual = cdll_at(par_qr.a_Q, par_qr.b1_Q, par_qr.bL_Q)
ll_qr_prof = cdll_at(par_qr.a_Q, b1q, bLq)

println("\n=== QR M-step ===")
@printf("  slopes: [%.4f, %.4f, %.4f]\n", par_qr.a_Q[2,:]...)
@printf("  intercepts: [%.4f, %.4f, %.4f]\n", par_qr.a_Q[1,:]...)
@printf("  quadratic: [%.4f, %.4f, %.4f]\n", par_qr.a_Q[3,:]...)
@printf("  b1_Q=%.4f bL_Q=%.4f (QR tails)\n", par_qr.b1_Q, par_qr.bL_Q)
@printf("  profiled b1=%.4f bL=%.4f\n", b1q, bLq)
@printf("  CDLL at QR: %.4f (profiled: %.4f)\n", ll_qr_actual, ll_qr_prof)
@printf("  Time: %.2fs\n", t_qr)

# Count crossings at observed eta_lag
Q_qr = H * par_qr.a_Q
n_cross_qr = count(j -> any(l -> Q_qr[j,l+1] < Q_qr[j,l], 1:L-1), 1:n_obs)
@printf("  Crossings at observed points: %d/%d\n", n_cross_qr, n_obs)

# ── MLE ───────────────────────────────────────────────────────────
par_mle = copy_params(par_true)
par_mle.a_Q .= par_true.a_Q .* 0.5
par_mle.b1_Q = 3.0; par_mle.bL_Q = 3.0
t_mle = @elapsed m_step_mle!(par_mle, eta_all, y, cfg)
b1m, bLm = profile_tails(par_mle.a_Q)
ll_mle_actual = cdll_at(par_mle.a_Q, par_mle.b1_Q, par_mle.bL_Q)
ll_mle_prof = cdll_at(par_mle.a_Q, b1m, bLm)

println("\n=== MLE M-step ===")
@printf("  slopes: [%.4f, %.4f, %.4f]\n", par_mle.a_Q[2,:]...)
@printf("  intercepts: [%.4f, %.4f, %.4f]\n", par_mle.a_Q[1,:]...)
@printf("  quadratic: [%.4f, %.4f, %.4f]\n", par_mle.a_Q[3,:]...)
@printf("  b1_Q=%.4f bL_Q=%.4f (MLE tails)\n", par_mle.b1_Q, par_mle.bL_Q)
@printf("  profiled b1=%.4f bL=%.4f\n", b1m, bLm)
@printf("  CDLL at MLE: %.4f (profiled: %.4f)\n", ll_mle_actual, ll_mle_prof)
@printf("  Time: %.2fs\n", t_mle)

Q_mle = H * par_mle.a_Q
n_cross_mle = count(j -> any(l -> Q_mle[j,l+1] < Q_mle[j,l], 1:L-1), 1:n_obs)
@printf("  Crossings at observed points: %d/%d\n", n_cross_mle, n_obs)

# ── Key question: if we take QR's params and evaluate CDLL, does MLE still beat it?
println("\n=== CDLL comparison (on the SAME eta_draws) ===")
@printf("  CDLL at truth:  %.6f\n", ll_true_prof)
@printf("  CDLL at QR:     %.6f (diff vs truth: %+.6f)\n", ll_qr_prof, ll_qr_prof - ll_true_prof)
@printf("  CDLL at MLE:    %.6f (diff vs truth: %+.6f)\n", ll_mle_prof, ll_mle_prof - ll_true_prof)
@printf("  CDLL: MLE vs QR: %+.6f\n", ll_mle_prof - ll_qr_prof)

# ── Check: is QR feasible for MLE's constraint?
println("\n=== Is QR solution feasible for MLE's non-crossing constraint? ===")
if n_cross_qr > 0
    println("  QR has $n_cross_qr crossings — MLE constraint rejects this")
    println("  MLE must find a non-crossing optimum — possibly lower ll than QR's")
else
    println("  QR has no crossings — MLE should be able to find it if optimal")
end

println("\nDone.")

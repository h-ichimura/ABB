include("ABB_three_period.jl")

K=2; L=3; sigma_y=1.0; T=3; N=200; M=10
tau = [0.25, 0.50, 0.75]
par_true = make_true_params_linear(tau=tau, sigma_y=sigma_y, K=K)
y, eta_true = generate_data_abb(N, par_true, tau, sigma_y, K; seed=42)
vp = fill(0.05, T)
cfg = Config(N, T, K, L, tau, sigma_y, 1, 200, M, vp)

eta_all = zeros(N, T, M)
for m in 1:M; eta_all[:,:,m] .= 0.6 .* y; end

global par = init_params(y, cfg)

# One E-step
acc = e_step!(eta_all, y, par, cfg)

# Stack transition data
eta_t, eta_lag = stack_transition(eta_all, cfg)
H = hermite_basis(eta_lag, K, sigma_y)
n_obs = length(eta_t)
Q_mat = zeros(n_obs, L)
q_sorted = zeros(L)
q_low = zeros(n_obs)
q_high = zeros(n_obs)

function neg_prof(a_Q)
    mul!(Q_mat, H, a_Q)
    for j in 1:n_obs
        for l in 1:L; q_sorted[l] = Q_mat[j,l]; end
        sort!(q_sorted)
        q_low[j] = q_sorted[1]; q_high[j] = q_sorted[L]
    end
    r1 = eta_t .- q_low; rL = eta_t .- q_high
    ml = r1 .<= 0; mh = rL .>= 0
    s1 = sum(r1[ml]); sL = sum(rL[mh])
    b1v = s1 < -1e-10 ? -count(ml)/s1 : 2.0
    bLv = sL >  1e-10 ?  count(mh)/sL : 2.0
    ll = 0.0
    for j in 1:n_obs
        for l in 1:L; q_sorted[l] = Q_mat[j,l]; end
        sort!(q_sorted)
        ll += pw_logdens(eta_t[j], q_sorted, tau, b1v, bLv)
    end
    -ll / n_obs
end

# QR M-step
m_step_qr!(par, eta_all, y, cfg)
a_qr = copy(par.a_Q)
obj_qr = neg_prof(a_qr)
@printf("After QR:  neg_prof = %.6f  slopes = [%.4f, %.4f, %.4f]\n",
        obj_qr, a_qr[2,:]...)

# Now run coordinate descent step by step
a_cur = copy(a_qr)
for cyc in 1:3
    for l in 1:L
        for k in 1:K+1
            if k == 1;     lo, hi = -2.0, 2.0
            elseif k == 2; lo, hi =  0.0, 2.0
            else;          lo, hi = -1.0, 1.0
            end

            old_val = a_cur[k,l]
            obj_before = neg_prof(a_cur)

            f1d(v) = (a_cur[k,l] = v; neg_prof(a_cur))
            res = optimize(f1d, lo, hi)
            a_cur[k,l] = Optim.minimizer(res)
            obj_after = neg_prof(a_cur)

            pname = k==1 ? "intcpt" : k==2 ? "slope" : "quad"
            @printf("  cyc%d %s(l=%d): %.4f -> %.4f  obj: %.6f -> %.6f  Δ=%+.6f\n",
                    cyc, pname, l, old_val, a_cur[k,l], obj_before, obj_after,
                    obj_after - obj_before)
        end
    end
end

obj_final = neg_prof(a_cur)
@printf("\nFinal:     neg_prof = %.6f  slopes = [%.4f, %.4f, %.4f]\n",
        obj_final, a_cur[2,:]...)
@printf("QR was:    neg_prof = %.6f\n", obj_qr)
@printf("Δ(MLE-QR) = %+.6f\n", obj_final - obj_qr)

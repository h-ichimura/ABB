#=
ABB_three_period.jl  --  Three-Period Nonlinear Earnings Process
================================================================
Model (simplified ABB 2017 Ecta, no age, T=3):
  y_{it} = η_{it} + ε_{it}
  η_{i1} ~ F_init,   η_{it} = Q(η_{i,t-1}, u_{it}),  u~U(0,1)
  ε_{it} ~ F_eps  (iid)
All distributions: piecewise-uniform + exponential tails.
Data generated from this model (correctly specified).
M draws from posterior used to approximate E-step integral.
=#

using LinearAlgebra, Statistics, Random, Printf, Optim

# ================================================================
#  HERMITE POLYNOMIALS
# ================================================================
@inline function He(n::Int, x::Float64)
    n == 0 && return 1.0
    n == 1 && return x
    hm2, hm1 = 1.0, x
    for k in 2:n; hm2, hm1 = hm1, x*hm1 - (k-1)*hm2; end
    hm1
end

function hermite_basis(x::AbstractVector{Float64}, K::Int, sigma::Float64)
    n = length(x)
    H = Matrix{Float64}(undef, n, K+1)
    @inbounds for i in 1:n
        z = x[i]/sigma; H[i,1] = 1.0
        K >= 1 && (H[i,2] = z)
        for k in 2:K; H[i,k+1] = z*H[i,k] - (k-1)*H[i,k-1]; end
    end
    H
end

# ================================================================
#  PIECEWISE-UNIFORM DENSITY + EXPONENTIAL TAILS
# ================================================================
function pw_logdens(x::Float64, q::AbstractVector{Float64},
                    tau::Vector{Float64}, b1::Float64, bL::Float64)
    L = length(q)
    x <= q[1] && return log(max(tau[1]*b1, 1e-300)) + b1*(x - q[1])
    x > q[L]  && return log(max((1.0-tau[L])*bL, 1e-300)) - bL*(x - q[L])
    @inbounds for l in 1:L-1
        if x <= q[l+1]
            dq = q[l+1] - q[l]
            return dq > 1e-12 ? log(tau[l+1]-tau[l]) - log(dq) : -700.0
        end
    end
    -700.0
end

function pw_draw(rng::AbstractRNG, q::AbstractVector{Float64},
                 tau::Vector{Float64}, b1::Float64, bL::Float64)
    L = length(q); u = rand(rng)
    u < tau[1]  && return q[1] + log(u/tau[1])/b1
    u >= tau[L] && return q[L] - log((1.0-u)/(1.0-tau[L]))/bL
    @inbounds for l in 1:L-1
        u < tau[l+1] && return q[l] + (q[l+1]-q[l])*(u-tau[l])/(tau[l+1]-tau[l])
    end
    q[L]
end

# ================================================================
#  MODEL PARAMETERS AND CONFIGURATION
# ================================================================
mutable struct Params
    a_Q::Matrix{Float64}; b1_Q::Float64; bL_Q::Float64
    a_init::Vector{Float64}; b1_init::Float64; bL_init::Float64
    a_eps::Vector{Float64}; b1_eps::Float64; bL_eps::Float64
end
function copy_params(p::Params)
    Params(copy(p.a_Q),p.b1_Q,p.bL_Q, copy(p.a_init),p.b1_init,p.bL_init,
           copy(p.a_eps),p.b1_eps,p.bL_eps)
end

struct Config
    N::Int; T::Int; K::Int; L::Int
    tau::Vector{Float64}; sigma_y::Float64
    maxiter::Int; n_draws::Int; M::Int   # M = draws kept for M-step
    var_prop::Vector{Float64}
end

# ================================================================
#  TRANSITION QUANTILE KNOTS
# ================================================================
function transition_quantiles!(q::Vector{Float64}, eta_lag::Float64,
                                a_Q::Matrix{Float64}, K::Int, sigma::Float64)
    z = eta_lag/sigma; L = size(a_Q,2)
    hv = Vector{Float64}(undef, K+1); hv[1]=1.0
    K>=1 && (hv[2]=z)
    for k in 2:K; hv[k+1] = z*hv[k]-(k-1)*hv[k-1]; end
    @inbounds for l in 1:L
        s=0.0; for k in 1:K+1; s += a_Q[k,l]*hv[k]; end; q[l]=s
    end
end

# ================================================================
#  LOG-LIKELIHOODS
# ================================================================
function partial_loglik(y, eta, t::Int, par::Params, cfg::Config, q::Vector{Float64})
    ll = pw_logdens(y[t]-eta[t], par.a_eps, cfg.tau, par.b1_eps, par.bL_eps)
    t==1 && (ll += pw_logdens(eta[1], par.a_init, cfg.tau, par.b1_init, par.bL_init))
    if t>=2; transition_quantiles!(q,eta[t-1],par.a_Q,cfg.K,cfg.sigma_y)
        ll += pw_logdens(eta[t], q, cfg.tau, par.b1_Q, par.bL_Q); end
    if t<cfg.T; transition_quantiles!(q,eta[t],par.a_Q,cfg.K,cfg.sigma_y)
        ll += pw_logdens(eta[t+1], q, cfg.tau, par.b1_Q, par.bL_Q); end
    ll
end

function full_loglik(y, eta, par::Params, cfg::Config, q::Vector{Float64})
    ll = pw_logdens(eta[1], par.a_init, cfg.tau, par.b1_init, par.bL_init)
    for t in 1:cfg.T; ll += pw_logdens(y[t]-eta[t], par.a_eps, cfg.tau,
                                         par.b1_eps, par.bL_eps); end
    for t in 2:cfg.T; transition_quantiles!(q,eta[t-1],par.a_Q,cfg.K,cfg.sigma_y)
        ll += pw_logdens(eta[t], q, cfg.tau, par.b1_Q, par.bL_Q); end
    ll
end

# ================================================================
#  DATA GENERATION
# ================================================================
function generate_data_abb(N, par::Params, tau, sigma_y, K; seed=42)
    rng = MersenneTwister(seed); T=3; L=length(tau)
    eta=zeros(N,T); y=zeros(N,T); q=zeros(L)
    for i in 1:N; eta[i,1]=pw_draw(rng,par.a_init,tau,par.b1_init,par.bL_init); end
    for t in 2:T, i in 1:N
        transition_quantiles!(q,eta[i,t-1],par.a_Q,K,sigma_y)
        eta[i,t]=pw_draw(rng,q,tau,par.b1_Q,par.bL_Q)
    end
    for t in 1:T, i in 1:N
        y[i,t]=eta[i,t]+pw_draw(rng,par.a_eps,tau,par.b1_eps,par.bL_eps)
    end
    y, eta
end

function norminv(p::Float64)
    p<=0.0 && return -Inf; p>=1.0 && return Inf
    t = p<0.5 ? sqrt(-2.0*log(p)) : sqrt(-2.0*log(1.0-p))
    x = t-(2.515517+0.802853t+0.010328t^2)/(1.0+1.432788t+0.189269t^2+0.001308t^3)
    p<0.5 ? -x : x
end

function make_true_params_linear(; rho=0.8, sigma_v=0.5, sigma_eps=0.3,
                                   sigma_eta1=1.0, tau, sigma_y, K=2)
    L=length(tau); a_Q=zeros(K+1,L)
    for l in 1:L; a_Q[1,l]=sigma_v*norminv(tau[l]); end
    a_Q[2,:] .= rho*sigma_y
    Params(a_Q, 1/sigma_v, 1/sigma_v,
           [sigma_eta1*norminv(tau[l]) for l in 1:L], 1/sigma_eta1, 1/sigma_eta1,
           [sigma_eps*norminv(tau[l]) for l in 1:L], 1/sigma_eps, 1/sigma_eps)
end

function make_true_params_nonlinear(; sigma_v_low=0.6, sigma_v_high=0.3,
                                      rho_low=0.5, rho_high=0.9,
                                      sigma_eps=0.3, sigma_eta1=1.0,
                                      tau, sigma_y, K=2)
    L=length(tau); a_Q=zeros(K+1,L)
    for l in 1:L
        w=tau[l]; rho_l=rho_low+(rho_high-rho_low)*w
        svl=sigma_v_low+(sigma_v_high-sigma_v_low)*w
        a_Q[1,l]=svl*norminv(tau[l]); a_Q[2,l]=rho_l*sigma_y
    end
    for l in 1:L; a_Q[3,l]=0.05*sigma_y*(tau[l]-0.5); end
    Params(a_Q, 2.0, 2.0,
           [sigma_eta1*norminv(tau[l]) for l in 1:L], 1/sigma_eta1, 1/sigma_eta1,
           [sigma_eps*norminv(tau[l]) for l in 1:L], 1/sigma_eps, 1/sigma_eps)
end

# ================================================================
#  E-STEP: MH chain, keep last M draws
#
#  eta_all[i, t, m] for m = 1,...,M
#  Run n_draws MH steps total; save the last M states.
#  Requires n_draws >= M.
# ================================================================
function e_step!(eta_all::Array{Float64,3}, y::Matrix{Float64},
                 par::Params, cfg::Config)
    N,T,M = cfg.N, cfg.T, cfg.M
    n_draws = cfg.n_draws
    @assert n_draws >= M
    save_start = n_draws - M + 1  # start saving from this MH step

    # Current state: initialize from last draw (m = M)
    eta_cur = eta_all[:, :, M]  # N × T, copy
    acc_count = zeros(T)
    q_buf = zeros(cfg.L)
    eta_buf = zeros(T)

    # Pre-compute partial log-likelihoods for current state
    pll = zeros(N, T)
    for i in 1:N, t in 1:T
        pll[i,t] = partial_loglik(view(y,i,:), view(eta_cur,i,:), t, par, cfg, q_buf)
    end

    save_idx = 0  # which draw slot to fill next

    for d in 1:n_draws
        for t in 1:T
            vp = cfg.var_prop[t]
            for i in 1:N
                @inbounds for s in 1:T; eta_buf[s]=eta_cur[i,s]; end
                eta_buf[t] = eta_cur[i,t] + sqrt(vp)*randn()
                prop = partial_loglik(view(y,i,:), eta_buf, t, par, cfg, q_buf)
                if log(rand()) < prop - pll[i,t]
                    eta_cur[i,t] = eta_buf[t]
                    pll[i,t] = prop
                    t>1 && (pll[i,t-1] = partial_loglik(
                        view(y,i,:), view(eta_cur,i,:), t-1, par, cfg, q_buf))
                    t<T && (pll[i,t+1] = partial_loglik(
                        view(y,i,:), view(eta_cur,i,:), t+1, par, cfg, q_buf))
                    acc_count[t] += 1
                end
            end
        end
        # Save this state if we're in the saving window
        if d >= save_start
            save_idx += 1
            eta_all[:, :, save_idx] .= eta_cur
        end
    end

    acc_count ./ (N * n_draws)
end

# ================================================================
#  TAIL UPDATE
# ================================================================
function update_tails(data::Vector{Float64}, q_low::Float64, q_high::Float64)
    r1=data.-q_low; rL=data.-q_high
    ml=r1.<=0; mh=rL.>=0; s1=sum(r1[ml]); sL=sum(rL[mh])
    b1 = s1<-1e-10 ? -count(ml)/s1 : 5.0
    bL = sL> 1e-10 ?  count(mh)/sL : 5.0
    b1, bL
end

# ================================================================
#  STACK ALL M DRAWS INTO VECTORS FOR M-STEP
#
#  For transition: stack N*(T-1)*M pairs (η_t, η_{t-1})
#  For initial: stack N*M values of η_1
#  For eps: stack N*T*M values of y-η
# ================================================================
function stack_transition(eta_all::Array{Float64,3}, cfg::Config)
    N,T,M = cfg.N, cfg.T, cfg.M
    n = N*(T-1)*M
    eta_t   = Vector{Float64}(undef, n)
    eta_lag = Vector{Float64}(undef, n)
    idx = 0
    for m in 1:M, t in 2:T, i in 1:N
        idx += 1
        eta_t[idx]   = eta_all[i, t, m]
        eta_lag[idx] = eta_all[i, t-1, m]
    end
    eta_t, eta_lag
end

function stack_initial(eta_all::Array{Float64,3}, cfg::Config)
    N,M = cfg.N, cfg.M
    v = Vector{Float64}(undef, N*M)
    idx = 0
    for m in 1:M, i in 1:N
        idx += 1; v[idx] = eta_all[i, 1, m]
    end
    v
end

function stack_eps(eta_all::Array{Float64,3}, y::Matrix{Float64}, cfg::Config)
    N,T,M = cfg.N, cfg.T, cfg.M
    v = Vector{Float64}(undef, N*T*M)
    idx = 0
    for m in 1:M, t in 1:T, i in 1:N
        idx += 1; v[idx] = y[i,t] - eta_all[i,t,m]
    end
    v
end

# ================================================================
#  M-STEP (QR)
# ================================================================
function m_step_qr!(par::Params, eta_all::Array{Float64,3},
                     y::Matrix{Float64}, cfg::Config)
    K, L = cfg.K, cfg.L

    # Transition
    eta_t, eta_lag = stack_transition(eta_all, cfg)
    H = hermite_basis(eta_lag, K, cfg.sigma_y)
    for l in 1:L
        tau_l = cfg.tau[l]
        obj(a) = let r = eta_t .- H*a; mean(r .* (tau_l .- (r.<0))); end
        res = optimize(obj, par.a_Q[:,l], LBFGS(),
                       Optim.Options(iterations=100, g_tol=1e-8, show_trace=false))
        par.a_Q[:,l] .= Optim.minimizer(res)
    end
    # Tail rates: QR estimates each column at the correct τ level,
    # so column 1 = τ[1] quantile, column L = τ[L] quantile
    r1=eta_t.-H*par.a_Q[:,1]; rL=eta_t.-H*par.a_Q[:,L]
    ml=r1.<=0; mh=rL.>=0; s1=sum(r1[ml]); sL=sum(rL[mh])
    s1<-1e-10 && (par.b1_Q = -count(ml)/s1)
    sL> 1e-10 && (par.bL_Q =  count(mh)/sL)

    # Initial η₁
    eta1_all = stack_initial(eta_all, cfg)
    for l in 1:L; par.a_init[l] = quantile(eta1_all, cfg.tau[l]); end
    par.b1_init, par.bL_init = update_tails(eta1_all, par.a_init[1], par.a_init[L])

    # Transitory ε
    eps_all = stack_eps(eta_all, y, cfg)
    for l in 1:L; par.a_eps[l] = quantile(eps_all, cfg.tau[l]); end
    par.a_eps .-= mean(par.a_eps)
    par.b1_eps, par.bL_eps = update_tails(eps_all, par.a_eps[1], par.a_eps[L])
    nothing
end

# ================================================================
#  M-STEP (MLE): COORDINATE DESCENT ON PROFILED LOG-LIKELIHOOD
#
#  For each knot parameter a_Q[k,l] in turn:
#    1. Profile out tail rates b1_Q, bL_Q via closed-form MLE
#    2. Minimize the neg profiled CDLL over a_Q[k,l] by golden section
#    3. Cycle through all (K+1)*L = 9 knot parameters
#    4. Repeat for n_cycles
#
#  Each 1D profile is unimodal and steep (verified by plot_profiled.jl),
#  so golden section search converges quickly.
# ================================================================


function m_step_mle!(par::Params, eta_all::Array{Float64,3},
                      y::Matrix{Float64}, cfg::Config)
    K, L = cfg.K, cfg.L

    # Marginals: sample quantiles (MLE for unconditional piecewise-uniform)
    eta1_all = stack_initial(eta_all, cfg)
    for l in 1:L; par.a_init[l] = quantile(eta1_all, cfg.tau[l]); end
    par.b1_init, par.bL_init = update_tails(eta1_all, par.a_init[1], par.a_init[L])

    eps_all = stack_eps(eta_all, y, cfg)
    for l in 1:L; par.a_eps[l] = quantile(eps_all, cfg.tau[l]); end
    par.a_eps .-= mean(par.a_eps)
    par.b1_eps, par.bL_eps = update_tails(eps_all, par.a_eps[1], par.a_eps[L])

    # Transition: coordinate descent with analytical feasible intervals
    # enforcing non-crossing Q_{l+1}(η_j) ≥ Q_l(η_j) at all observed η_j.
    eta_t, eta_lag = stack_transition(eta_all, cfg)
    H = hermite_basis(eta_lag, K, cfg.sigma_y)
    n_obs = length(eta_t)

    # QR warm start
    m_step_qr!(par, eta_all, y, cfg)
    a_cur = copy(par.a_Q)

    # Profiled neg-CDLL with sorted knots (handles any parameter values)
    q_sorted = Vector{Float64}(undef, L)
    function neg_profiled_cdll(a_Q_mat)
        Q_mat = H * a_Q_mat
        # Compute tail rates from sorted lowest/highest knots
        ll = 0.0; n_left = 0; sum_left = 0.0; n_right = 0; sum_right = 0.0
        @inbounds for j in 1:n_obs
            for l in 1:L; q_sorted[l] = Q_mat[j, l]; end
            sort!(q_sorted)
            r1 = eta_t[j] - q_sorted[1]
            rL = eta_t[j] - q_sorted[L]
            if r1 <= 0; n_left += 1; sum_left += r1; end
            if rL >= 0; n_right += 1; sum_right += rL; end
        end
        b1v = sum_left < -1e-10 ? -n_left / sum_left : par.b1_Q
        bLv = sum_right > 1e-10 ? n_right / sum_right : par.bL_Q
        @inbounds for j in 1:n_obs
            for l in 1:L; q_sorted[l] = Q_mat[j, l]; end
            sort!(q_sorted)
            ll += pw_logdens(eta_t[j], q_sorted, cfg.tau, b1v, bLv)
        end
        -ll / n_obs
    end

    # Coordinate descent with moderate bounds around current value.
    # The sorting in neg_profiled_cdll handles crossing, but moderate
    # bounds avoid Brent jumping across sort boundaries.
    n_cycles = 8
    for cyc in 1:n_cycles
        max_step = 0.0
        w = cyc <= 3 ? 0.3 : (cyc <= 6 ? 0.1 : 0.03)  # shrinking window
        for l in 1:L, k in 1:K+1
            cur = a_cur[k, l]
            if k == 2
                lo = max(cur - w, 0.0)  # slope non-negative
                hi = cur + w
            else
                lo = cur - w
                hi = cur + w
            end

            function f1d(v)
                a_cur[k, l] = v
                neg_profiled_cdll(a_cur)
            end

            res1d = optimize(f1d, lo, hi)
            new_val = Optim.minimizer(res1d)
            max_step = max(max_step, abs(new_val - cur))
            a_cur[k, l] = new_val
        end
        max_step < 1e-6 && break
    end

    par.a_Q .= a_cur

    # Final tail rates from fitted a_Q
    Q_mat = H * par.a_Q
    r1 = eta_t .- view(Q_mat, :, 1); rL = eta_t .- view(Q_mat, :, L)
    ml = r1 .<= 0; mh = rL .>= 0
    s1 = sum(r1[ml]); sL = sum(rL[mh])
    s1 < -1e-10 && (par.b1_Q = -count(ml)/s1)
    sL >  1e-10 && (par.bL_Q =  count(mh)/sL)
    nothing
end

# ================================================================
#  M-STEP (OLS): SEGMENT-WISE OLS
#
#  Since Y_j is uniform on (q_ell, q_{ell+1}) within segment ell,
#  the conditional mean equals the conditional median:
#    E[Y_j | X_j, segment ell] = (q_ell(X_j) + q_{ell+1}(X_j)) / 2
#                               = h_j' (a_ell + a_{ell+1}) / 2
#
#  For the tails (exponential):
#    E[Y_j | X_j, left tail]  = q_1(X_j) - 1/lambda^Q
#    E[Y_j | X_j, right tail] = q_3(X_j) + 1/lambda_+^Q
#
#  Procedure:
#    1. Assign each observation to a segment using current parameters
#    2. OLS on left tail  -> a_1 (slope/quadratic; intercept shifted by 1/lambda)
#    3. OLS on right tail -> a_3 (slope/quadratic; intercept shifted by 1/lambda_+)
#    4. OLS on segment 1  -> (a_1 + a_2)/2, solve for a_2
#    5. OLS on segment 2  -> (a_2 + a_3)/2, cross-check
#    6. Update tail rates from residuals (closed form)
#
#  All steps are closed-form OLS — no iterative optimization.
# ================================================================

function m_step_ols!(par::Params, eta_all::Array{Float64,3},
                      y::Matrix{Float64}, cfg::Config)
    K, L = cfg.K, cfg.L
    @assert L == 3 "OLS M-step currently requires L=3"

    # Marginals: same as QR
    eta1_all = stack_initial(eta_all, cfg)
    for l in 1:L; par.a_init[l] = quantile(eta1_all, cfg.tau[l]); end
    par.b1_init, par.bL_init = update_tails(eta1_all, par.a_init[1], par.a_init[L])

    eps_all = stack_eps(eta_all, y, cfg)
    for l in 1:L; par.a_eps[l] = quantile(eps_all, cfg.tau[l]); end
    par.a_eps .-= mean(par.a_eps)
    par.b1_eps, par.bL_eps = update_tails(eps_all, par.a_eps[1], par.a_eps[L])

    # Transition: segment-wise OLS
    eta_t, eta_lag = stack_transition(eta_all, cfg)
    H = hermite_basis(eta_lag, K, cfg.sigma_y)
    n_obs = length(eta_t)
    Q_mat = Matrix{Float64}(undef, n_obs, L)

    # Step 1: Compute knots and assign segments using current parameters
    mul!(Q_mat, H, par.a_Q)

    seg = zeros(Int, n_obs)  # 0=left, 1=seg1, 2=seg2, 3=right
    @inbounds for j in 1:n_obs
        x = eta_t[j]
        if x <= Q_mat[j, 1]
            seg[j] = 0
        elseif x > Q_mat[j, L]
            seg[j] = 3
        elseif x <= Q_mat[j, 2]
            seg[j] = 1
        else
            seg[j] = 2
        end
    end

    idx_left  = findall(seg .== 0)
    idx_seg1  = findall(seg .== 1)
    idx_seg2  = findall(seg .== 2)
    idx_right = findall(seg .== 3)

    # Step 2: OLS on left tail
    # E[Y_j | X_j, left] = h_j' a_1 - 1/lambda^Q
    # So Y_j + 1/lambda^Q = h_j' a_1
    # First estimate a_1 slope and quadratic from OLS (intercept absorbs 1/lambda)
    if length(idx_left) > K + 2
        H_left = H[idx_left, :]
        Y_left = eta_t[idx_left]
        a1_ols = H_left \ Y_left   # OLS: includes the -1/lambda shift in intercept
        # a1_ols estimates a_1 with intercept biased by -1/lambda
        # We get lambda from the tail MLE below, then correct
        par.a_Q[:, 1] .= a1_ols
    end

    # Step 3: OLS on right tail
    # E[Y_j | X_j, right] = h_j' a_3 + 1/lambda_+^Q
    if length(idx_right) > K + 2
        H_right = H[idx_right, :]
        Y_right = eta_t[idx_right]
        a3_ols = H_right \ Y_right  # intercept biased by +1/lambda_+
        par.a_Q[:, 3] .= a3_ols
    end

    # Step 4: Update tail rates from residuals (closed form)
    # This also lets us correct the intercept bias
    mul!(Q_mat, H, par.a_Q)
    r1 = eta_t .- view(Q_mat, :, 1)
    rL = eta_t .- view(Q_mat, :, L)
    ml = r1 .<= 0; mh = rL .>= 0
    s1 = sum(r1[ml]); sL = sum(rL[mh])
    b1_new = s1 < -1e-10 ? -count(ml) / s1 : par.b1_Q
    bL_new = sL >  1e-10 ?  count(mh) / sL : par.bL_Q

    # Correct intercepts: OLS on left tail estimated a_1[1] - 1/lambda,
    # so true a_1[1] = a1_ols[1] + 1/lambda.  Similarly for right tail.
    if length(idx_left) > K + 2
        par.a_Q[1, 1] += 1.0 / b1_new
    end
    if length(idx_right) > K + 2
        par.a_Q[1, 3] -= 1.0 / bL_new
    end
    par.b1_Q = b1_new
    par.bL_Q = bL_new

    # Step 5: OLS on segment 1
    # E[Y_j | X_j, seg 1] = h_j' (a_1 + a_2) / 2
    # So h_j' a_2 = 2 * E[Y_j | X_j, seg 1] - h_j' a_1
    if length(idx_seg1) > K + 2
        H_s1 = H[idx_seg1, :]
        Y_s1 = eta_t[idx_seg1]
        # OLS estimates (a_1 + a_2)/2
        mid1_ols = H_s1 \ Y_s1
        par.a_Q[:, 2] .= 2.0 .* mid1_ols .- par.a_Q[:, 1]
    end

    # Step 6: OLS on segment 2 as cross-check / alternative
    # E[Y_j | X_j, seg 2] = h_j' (a_2 + a_3) / 2
    # Could use this to improve a_2 or a_3, but with a_1 and a_3 from tails
    # and a_2 from segment 1, we already have all parameters.
    # Use segment 2 to refine a_2 by averaging the two estimates:
    if length(idx_seg2) > K + 2
        H_s2 = H[idx_seg2, :]
        Y_s2 = eta_t[idx_seg2]
        mid2_ols = H_s2 \ Y_s2
        a2_from_seg2 = 2.0 .* mid2_ols .- par.a_Q[:, 3]
        # Average the two estimates of a_2
        par.a_Q[:, 2] .= 0.5 .* (par.a_Q[:, 2] .+ a2_from_seg2)
    end

    # Final tail rate update with corrected knots
    mul!(Q_mat, H, par.a_Q)
    r1 = eta_t .- view(Q_mat, :, 1); rL = eta_t .- view(Q_mat, :, L)
    ml = r1 .<= 0; mh = rL .>= 0
    s1 = sum(r1[ml]); sL = sum(rL[mh])
    s1 < -1e-10 && (par.b1_Q = -count(ml) / s1)
    sL >  1e-10 && (par.bL_Q =  count(mh) / sL)
    nothing
end

# ================================================================
#  INITIALIZATION
# ================================================================
function init_params(y::Matrix{Float64}, cfg::Config)
    L,K = cfg.L, cfg.K; y_all=vec(y)
    a_init = [quantile(y[:,1], cfg.tau[l]) for l in 1:L]
    a_eps  = [quantile(y_all, cfg.tau[l])*0.2 for l in 1:L]
    a_eps .-= mean(a_eps)
    a_Q = zeros(K+1, L)
    a_Q[1,:] .= [quantile(y_all, cfg.tau[l])*0.1 for l in 1:L]
    K>=1 && (a_Q[2,:] .= 0.5*cfg.sigma_y)  # deliberately away from true 0.8
    Params(a_Q, 3.0,3.0, a_init, 3.0,3.0, a_eps, 3.0,3.0)
end

# ================================================================
#  ESTIMATION LOOP
# ================================================================
struct ParamHistory
    a_Q::Array{Float64,3}; b1_Q::Vector{Float64}; bL_Q::Vector{Float64}
    a_init::Matrix{Float64}; b1_init::Vector{Float64}; bL_init::Vector{Float64}
    a_eps::Matrix{Float64};  b1_eps::Vector{Float64};  bL_eps::Vector{Float64}
end

function estimate(y::Matrix{Float64}, cfg::Config;
                  method::Symbol=:qr, verbose::Bool=true)
    N,T,M = cfg.N, cfg.T, cfg.M
    S = cfg.maxiter; S2 = div(S,2); K,L = cfg.K, cfg.L
    par = init_params(y, cfg)

    # Initialize M draws: all start at 0.6*y
    eta_all = zeros(N, T, M)
    for m in 1:M; eta_all[:,:,m] .= 0.6 .* y; end

    ll_hist = zeros(S); q_buf = zeros(L)
    hist = ParamHistory(
        zeros(K+1,L,S), zeros(S), zeros(S),
        zeros(L,S), zeros(S), zeros(S),
        zeros(L,S), zeros(S), zeros(S))

    for iter in 1:S
        acc = e_step!(eta_all, y, par, cfg)
        if method==:qr; m_step_qr!(par,eta_all,y,cfg)
        elseif method==:mle; m_step_mle!(par,eta_all,y,cfg)
        elseif method==:ols; m_step_ols!(par,eta_all,y,cfg)
        end

        hist.a_Q[:,:,iter].=par.a_Q; hist.b1_Q[iter]=par.b1_Q; hist.bL_Q[iter]=par.bL_Q
        hist.a_init[:,iter].=par.a_init; hist.b1_init[iter]=par.b1_init; hist.bL_init[iter]=par.bL_init
        hist.a_eps[:,iter].=par.a_eps; hist.b1_eps[iter]=par.b1_eps; hist.bL_eps[iter]=par.bL_eps

        # Monitor: avg complete-data loglik over M draws
        ll = 0.0
        for m in 1:M, i in 1:N
            ll += full_loglik(view(y,i,:), view(eta_all,i,:,m), par, cfg, q_buf)
        end
        ll_hist[iter] = ll / (N*M)

        if verbose && (iter%25==0 || iter<=3)
            @printf("  [%-3s] %3d/%d | ll %8.4f | acc %s\n",
                    uppercase(string(method)), iter, S, ll_hist[iter],
                    join([@sprintf("%.2f",a) for a in acc], "/"))
        end
    end

    rng = (S-S2+1):S
    par_avg = Params(
        dropdims(mean(hist.a_Q[:,:,rng],dims=3),dims=3),
        mean(hist.b1_Q[rng]), mean(hist.bL_Q[rng]),
        vec(mean(hist.a_init[:,rng],dims=2)),
        mean(hist.b1_init[rng]), mean(hist.bL_init[rng]),
        vec(mean(hist.a_eps[:,rng],dims=2)),
        mean(hist.b1_eps[rng]), mean(hist.bL_eps[rng]))
    @printf("  Averaged last S̃=%d of S=%d  (M=%d draws per iter)\n", S2, S, M)
    par_avg, par, eta_all, ll_hist, hist
end

# ================================================================
#  PERCENTILE TABLE
# ================================================================
function print_percentiles(hist::ParamHistory, par_true::Params,
                           cfg::Config; label::String="")
    S=size(hist.a_Q,3); S2=div(S,2); rng=(S-S2+1):S; L=cfg.L
    pcts=[0.10, 0.25, 0.50, 0.75, 0.90]
    println("\n  Percentiles over last S̃=$S2 iterations  $label")
    println("  "*"-"^78)
    @printf("  %-32s %8s | %8s %8s %8s %8s %8s\n",
            "Parameter","True","p10","p25","p50","p75","p90")
    println("  "*"-"^78)
    for l in 1:L
        v=vec(hist.a_Q[2,l,rng])./cfg.sigma_y; pv=quantile(v,pcts)
        @printf("  %-32s %8.4f | %8.4f %8.4f %8.4f %8.4f %8.4f\n",
                "persistence (τ=$(cfg.tau[l]))", par_true.a_Q[2,l]/cfg.sigma_y, pv...)
    end
    for l in 1:L
        v=vec(hist.a_Q[1,l,rng]); pv=quantile(v,pcts)
        @printf("  %-32s %8.4f | %8.4f %8.4f %8.4f %8.4f %8.4f\n",
                "intercept (τ=$(cfg.tau[l]))", par_true.a_Q[1,l], pv...)
    end
    for l in 1:L
        v=vec(hist.a_eps[l,rng]); pv=quantile(v,pcts)
        @printf("  %-32s %8.4f | %8.4f %8.4f %8.4f %8.4f %8.4f\n",
                "ε quantile (τ=$(cfg.tau[l]))", par_true.a_eps[l], pv...)
    end
    for (nm,hv,tv) in [("b1_Q",hist.b1_Q,par_true.b1_Q),("bL_Q",hist.bL_Q,par_true.bL_Q),
                        ("b1_eps",hist.b1_eps,par_true.b1_eps),("bL_eps",hist.bL_eps,par_true.bL_eps)]
        pv=quantile(vec(hv[rng]),pcts)
        @printf("  %-32s %8.4f | %8.4f %8.4f %8.4f %8.4f %8.4f\n", nm, tv, pv...)
    end
    println("  "*"-"^78)
end

# ================================================================
#  COMPARISON
# ================================================================
function run_comparison(; N=300, K=2, L=3, maxiter=100, n_draws=200,
                         M=50, var_prop=0.05, seed=42, nonlinear=false,
                         methods=[:qr, :ols, :mle])
    T=3; sigma_y=1.0
    tau = collect(range(1/(L+1), stop=L/(L+1), length=L))
    if nonlinear
        par_true = make_true_params_nonlinear(tau=tau, sigma_y=sigma_y, K=K)
        dgp_str = "Nonlinear ABB"
    else
        par_true = make_true_params_linear(tau=tau, sigma_y=sigma_y, K=K)
        dgp_str = "Linear ABB (ρ=0.8, σ_v=0.5, σ_ε=0.3)"
    end
    y, eta_true = generate_data_abb(N, par_true, tau, sigma_y, K; seed=seed)
    vp = fill(var_prop, T)
    cfg = Config(N,T,K,L, tau, sigma_y, maxiter, n_draws, M, vp)

    println("="^70)
    println("  ABB T=3: QR vs OLS vs MLE  (M=$M draws per E-step)")
    println("="^70)
    println("DGP: $dgp_str")
    @printf("N=%d T=%d K=%d L=%d S=%d n_draws=%d M=%d\n", N,T,K,L,maxiter,n_draws,M)
    println("τ: ", join([@sprintf("%.3f",t) for t in tau],", "))
    println()

    results = Dict{Symbol, Any}()
    for meth in methods
        println("-"^70,"\n  $(uppercase(string(meth)))\n","-"^70)
        t_el = @elapsed p_avg,_,eta_est,ll,hist = estimate(y,cfg; method=meth)
        @printf("  %.1f s\n", t_el)
        results[meth] = (par=p_avg, eta=eta_est, ll=ll, hist=hist, time=t_el)
    end

    S2=div(maxiter,2)
    println("\n","="^70)
    @printf("  RESULTS (S=%d, avg last %d, M=%d)\n", maxiter, S2, M)
    println("="^70)

    # Log-likelihood
    print("Avg ll (last S̃):")
    for meth in methods
        ll = results[meth].ll
        @printf("  %s=%.4f", uppercase(string(meth)), mean(ll[end-S2+1:end]))
    end
    println()

    # Monotonicity
    print("Mono violations: ")
    for meth in methods
        dll = diff(results[meth].ll)
        @printf(" %s=%d/%d", uppercase(string(meth)), count(dll.<-0.01), maxiter-1)
    end
    println()

    # Eta recovery
    println("\nη recovery (corr, last draw m=$M):")
    for t in 1:T
        @printf("  t=%d:", t)
        for meth in methods
            c = cor(results[meth].eta[:,t,M], eta_true[:,t])
            @printf("  %s=%.4f", uppercase(string(meth)), c)
        end
        println()
    end

    # Timing
    print("\nTime (s):")
    for meth in methods
        @printf("  %s=%.1f", uppercase(string(meth)), results[meth].time)
    end
    println()

    # Percentile tables
    for meth in methods
        print_percentiles(results[meth].hist, par_true, cfg;
                          label="[$(uppercase(string(meth)))]")
    end
    println()
    results
end

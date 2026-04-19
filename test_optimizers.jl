#=
test_optimizers.jl — Compare coordinate descent vs steepest descent
on the profiled CDLL for transition knot parameters.
Both profile out tail rates via closed-form MLE.
=#

include("ABB_three_period.jl")

K=2; L=3; sigma_y=1.0; T=3; N=2000; M=1
tau = [0.25, 0.50, 0.75]
par_true = make_true_params_linear(tau=tau, sigma_y=sigma_y, K=K)
y, eta_true = generate_data_abb(N, par_true, tau, sigma_y, K; seed=42)
y_clean = copy(eta_true)

cfg = Config(N, T, K, L, tau, sigma_y, 1, 1, M, fill(0.05, T))
eta_all = zeros(N, T, M)
eta_all[:,:,1] .= eta_true

# Stack transition data
eta_t, eta_lag = stack_transition(eta_all, cfg)
H = hermite_basis(eta_lag, K, sigma_y)
n_obs = length(eta_t)
Q_mat = zeros(n_obs, L)
q_sorted = zeros(L)

"""Profiled neg-CDLL as function of vectorized knot params."""
function neg_prof_cdll(theta::Vector{Float64})
    a = reshape(theta, K+1, L)
    mul!(Q_mat, H, a)
    r1 = eta_t .- view(Q_mat,:,1); rL = eta_t .- view(Q_mat,:,L)
    ml = r1 .<= 0; mh = rL .>= 0
    s1 = sum(r1[ml]); sL = sum(rL[mh])
    b1v = s1 < -1e-10 ? -count(ml)/s1 : 2.0
    bLv = sL >  1e-10 ?  count(mh)/sL : 2.0
    ll = 0.0
    @inbounds for j in 1:n_obs
        for l in 1:L; q_sorted[l] = Q_mat[j,l]; end
        sort!(q_sorted)
        ll += pw_logdens(eta_t[j], q_sorted, tau, b1v, bLv)
    end
    -ll / n_obs
end

"""Numerical gradient by central differences."""
function num_grad(f, x; h=1e-5)
    g = zeros(length(x))
    for i in eachindex(x)
        xp = copy(x); xp[i] += h
        xm = copy(x); xm[i] -= h
        g[i] = (f(xp) - f(xm)) / (2h)
    end
    g
end

# Starting point: 0.5x truth
a_start = copy(par_true.a_Q) .* 0.5
theta_start = vec(copy(a_start))

println("True params (vectorized):")
println("  ", round.(vec(par_true.a_Q), digits=4))
println("Start (0.5x truth):")
println("  ", round.(theta_start, digits=4))
@printf("neg-CDLL at truth: %.6f\n", neg_prof_cdll(vec(copy(par_true.a_Q))))
@printf("neg-CDLL at start: %.6f\n\n", neg_prof_cdll(theta_start))

# ================================================================
println("="^60)
println("METHOD 1: Coordinate Descent (golden section)")
println("="^60)

function run_coord_descent(theta0; n_cycles=5, tol=1e-5)
    theta = copy(theta0)
    np = length(theta)
    neval = 0
    for cycle in 1:n_cycles
        for i in 1:np
            cur = theta[i]
            # Determine search bounds
            k = mod(i-1, K+1) + 1  # which Hermite order
            if k == 1;     lo, hi = cur-0.5, cur+0.5  # intercept
            elseif k == 2; lo, hi = max(cur-0.3, 0.0), cur+0.3  # slope
            else;          lo, hi = cur-0.2, cur+0.2  # quadratic
            end
            function f1d(v)
                theta[i] = v
                neval += 1
                neg_prof_cdll(theta)
            end
            theta[i] = golden_section(f1d, lo, hi; tol=tol, maxiter=50)
        end
    end
    theta, neval
end

t1 = @elapsed theta_cd, neval_cd = run_coord_descent(copy(theta_start))
obj_cd = neg_prof_cdll(theta_cd)
@printf("  Final neg-CDLL: %.6f  (evals: %d, time: %.2f s)\n", obj_cd, neval_cd, t1)
a_cd = reshape(theta_cd, K+1, L)
@printf("  slopes:     [%.4f, %.4f, %.4f]\n", a_cd[2,:]...)
@printf("  intercepts: [%.4f, %.4f, %.4f]\n", a_cd[1,:]...)

# ================================================================
println("\n","="^60)
println("METHOD 2: Steepest Descent (backtracking line search)")
println("="^60)

function run_steepest_descent(theta0; maxiter=200, tol=1e-8, alpha0=0.01)
    theta = copy(theta0)
    neval = 0
    obj = neg_prof_cdll(theta); neval += 1
    for iter in 1:maxiter
        g = num_grad(neg_prof_cdll, theta); neval += 2*length(theta)
        gnorm = norm(g)
        gnorm < tol && break

        # Backtracking line search (Armijo)
        alpha = alpha0
        d = -g / gnorm  # normalized descent direction
        obj_new = neg_prof_cdll(theta .+ alpha .* d); neval += 1
        while obj_new > obj - 1e-4 * alpha * gnorm && alpha > 1e-10
            alpha *= 0.5
            obj_new = neg_prof_cdll(theta .+ alpha .* d); neval += 1
        end
        if alpha > 1e-10
            theta .+= alpha .* d
            obj = obj_new
        else
            break
        end
    end
    theta, neval
end

t2 = @elapsed theta_sd, neval_sd = run_steepest_descent(copy(theta_start))
obj_sd = neg_prof_cdll(theta_sd)
@printf("  Final neg-CDLL: %.6f  (evals: %d, time: %.2f s)\n", obj_sd, neval_sd, t2)
a_sd = reshape(theta_sd, K+1, L)
@printf("  slopes:     [%.4f, %.4f, %.4f]\n", a_sd[2,:]...)
@printf("  intercepts: [%.4f, %.4f, %.4f]\n", a_sd[1,:]...)

# ================================================================
println("\n","="^60)
println("METHOD 3: Steepest Descent (larger step, unnormalized)")
println("="^60)

function run_steepest_descent2(theta0; maxiter=500, tol=1e-8, alpha0=1.0)
    theta = copy(theta0)
    neval = 0
    obj = neg_prof_cdll(theta); neval += 1
    for iter in 1:maxiter
        g = num_grad(neg_prof_cdll, theta); neval += 2*length(theta)
        gnorm = norm(g)
        gnorm < tol && break

        # Backtracking line search (Armijo), unnormalized direction
        alpha = alpha0
        d = -g
        obj_new = neg_prof_cdll(theta .+ alpha .* d); neval += 1
        while obj_new > obj - 1e-4 * alpha * dot(g,g) && alpha > 1e-10
            alpha *= 0.5
            obj_new = neg_prof_cdll(theta .+ alpha .* d); neval += 1
        end
        if alpha > 1e-10
            theta .+= alpha .* d
            obj = obj_new
        else
            break
        end
    end
    theta, neval
end

t3 = @elapsed theta_sd2, neval_sd2 = run_steepest_descent2(copy(theta_start))
obj_sd2 = neg_prof_cdll(theta_sd2)
@printf("  Final neg-CDLL: %.6f  (evals: %d, time: %.2f s)\n", obj_sd2, neval_sd2, t3)
a_sd2 = reshape(theta_sd2, K+1, L)
@printf("  slopes:     [%.4f, %.4f, %.4f]\n", a_sd2[2,:]...)
@printf("  intercepts: [%.4f, %.4f, %.4f]\n", a_sd2[1,:]...)

# ================================================================
println("\n","="^60)
println("SUMMARY")
println("="^60)
obj_true = neg_prof_cdll(vec(copy(par_true.a_Q)))
@printf("  %-30s  neg-CDLL  evals   time\n", "Method")
@printf("  %-30s  %.6f      -      -\n", "Truth", obj_true)
@printf("  %-30s  %.6f      -      -\n", "Start (0.5x)", neg_prof_cdll(theta_start))
@printf("  %-30s  %.6f  %5d  %.2f s\n", "Coord descent (golden)", obj_cd, neval_cd, t1)
@printf("  %-30s  %.6f  %5d  %.2f s\n", "Steepest (normalized)", obj_sd, neval_sd, t2)
@printf("  %-30s  %.6f  %5d  %.2f s\n", "Steepest (unnormalized)", obj_sd2, neval_sd2, t3)

#=
compare_optimizers.jl — Compare 4 optimization methods for logistic-ABB MLE

Data generated from the asymmetric logistic model with ABB-style quantile knots.
All methods use the same numerical gradient (central differences).

Methods:
  1. LBFGS (Optim.jl)
  2. Gradient descent with backtracking line search
  3. Adam (adaptive learning rate)
  4. Conjugate gradient (Fletcher-Reeves)
=#

include("logistic_abb.jl")
using Printf

# ================================================================
#  NUMERICAL GRADIENT
# ================================================================

function num_grad!(g::Vector{Float64}, f, v::Vector{Float64}; h::Float64=1e-5)
    for i in eachindex(v)
        vi = v[i]
        v[i] = vi + h; fp = f(v)
        v[i] = vi - h; fm = f(v)
        v[i] = vi
        g[i] = (fp - fm) / (2h)
    end
end

# ================================================================
#  METHOD 1: LBFGS (Optim.jl)
# ================================================================

function opt_lbfgs(f, v0; maxiter=100, verbose=true)
    res = optimize(f, v0, LBFGS(),
                   Optim.Options(iterations=maxiter, g_tol=1e-5,
                                 show_trace=false))
    Optim.minimizer(res), Optim.minimum(res), Optim.iterations(res)
end

# ================================================================
#  METHOD 2: Gradient descent with backtracking
# ================================================================

function opt_gd(f, v0; maxiter=200, lr=0.01, verbose=true)
    v = copy(v0); np = length(v); g = zeros(np)
    obj = f(v)
    for iter in 1:maxiter
        num_grad!(g, f, v)
        gnorm = sqrt(sum(g.^2))
        gnorm < 1e-5 && break
        # Backtracking line search
        step = lr
        for _ in 1:20
            v_new = v .- step .* g
            obj_new = f(v_new)
            if obj_new < obj - 1e-4 * step * gnorm^2
                v .= v_new; obj = obj_new; break
            end
            step *= 0.5
        end
        if verbose && (iter <= 5 || iter % 20 == 0)
            @printf("    GD iter %3d: obj=%.6f |g|=%.4e step=%.4e\n", iter, obj, gnorm, step)
        end
    end
    v, obj, 0
end

# ================================================================
#  METHOD 3: Adam
# ================================================================

function opt_adam(f, v0; maxiter=500, lr=0.001, β1=0.9, β2=0.999, ε=1e-8, verbose=true)
    v = copy(v0); np = length(v); g = zeros(np)
    m = zeros(np); s = zeros(np)
    obj = f(v)
    best_v = copy(v); best_obj = obj
    for iter in 1:maxiter
        num_grad!(g, f, v)
        m .= β1 .* m .+ (1 - β1) .* g
        s .= β2 .* s .+ (1 - β2) .* g.^2
        m_hat = m ./ (1 - β1^iter)
        s_hat = s ./ (1 - β2^iter)
        v .-= lr .* m_hat ./ (sqrt.(s_hat) .+ ε)
        obj = f(v)
        if obj < best_obj; best_v .= v; best_obj = obj; end
        if verbose && (iter <= 5 || iter % 50 == 0)
            @printf("    Adam iter %3d: obj=%.6f |g|=%.4e\n",
                    iter, obj, sqrt(sum(g.^2)))
        end
    end
    best_v, best_obj, 0
end

# ================================================================
#  METHOD 4: Conjugate gradient (Fletcher-Reeves)
# ================================================================

function opt_cg(f, v0; maxiter=200, verbose=true)
    v = copy(v0); np = length(v)
    g = zeros(np); g_old = zeros(np); d = zeros(np)
    num_grad!(g, f, v)
    d .= -g; obj = f(v)
    for iter in 1:maxiter
        gnorm = sqrt(sum(g.^2))
        gnorm < 1e-5 && break
        # Line search along d
        step = 0.1
        for _ in 1:30
            v_new = v .+ step .* d
            obj_new = f(v_new)
            if obj_new < obj - 1e-4 * step * abs(dot(g, d))
                v .= v_new; obj = obj_new; break
            end
            step *= 0.5
        end
        g_old .= g
        num_grad!(g, f, v)
        # Fletcher-Reeves β
        β_fr = max(0.0, dot(g, g) / max(dot(g_old, g_old), 1e-30))
        d .= -g .+ β_fr .* d
        if verbose && (iter <= 5 || iter % 20 == 0)
            @printf("    CG iter %3d: obj=%.6f |g|=%.4e β=%.4f\n",
                    iter, obj, gnorm, β_fr)
        end
    end
    v, obj, 0
end

# ================================================================
#  COMPARISON
# ================================================================

function main()
    K = 2; L = 3; sigma_y = 1.0; T = 3; N = 300
    tau = [0.25, 0.50, 0.75]
    par_true = make_true_params_linear(tau=tau, sigma_y=sigma_y, K=K)
    y, eta = generate_data_logistic_abb(N, par_true, tau, sigma_y, K; seed=42)
    cfg = Config(N, T, K, L, tau, sigma_y, 1, 1, 1, fill(0.05, T))

    nll_true = logistic_neg_loglik(par_true, y, cfg; G=200)
    @printf("neg-ll at truth: %.6f\n\n", nll_true); flush(stdout)

    # Starting point: 10% perturbed
    par0 = copy_params(par_true)
    par0.a_Q .*= 0.9; par0.a_init .*= 1.1; par0.a_eps .*= 0.9
    v0 = params_to_vec_la(par0)

    obj(v) = (vec_to_params_la!(par0, v, K, L);
              logistic_neg_loglik(par0, y, cfg; G=200))

    nll0 = obj(v0)
    @printf("neg-ll at start: %.6f\n\n", nll0); flush(stdout)

    methods = [
        ("LBFGS",     () -> opt_lbfgs(obj, copy(v0); maxiter=50)),
        ("GD",        () -> opt_gd(obj, copy(v0); maxiter=100, lr=0.01)),
        ("Adam",      () -> opt_adam(obj, copy(v0); maxiter=300, lr=0.003)),
        ("CG",        () -> opt_cg(obj, copy(v0); maxiter=100)),
    ]

    println("="^60); flush(stdout)
    for (name, run_fn) in methods
        @printf("--- %s ---\n", name); flush(stdout)
        t = @elapsed v_opt, nll_opt, _ = run_fn()
        par_opt = copy_params(par_true)
        vec_to_params_la!(par_opt, v_opt, K, L)
        @printf("  neg-ll: %.6f (diff from truth: %+.6f)\n", nll_opt, nll_opt - nll_true)
        @printf("  slopes: [%.4f, %.4f, %.4f]\n", par_opt.a_Q[2,:]...)
        @printf("  intcpts:[%.4f, %.4f, %.4f]\n", par_opt.a_Q[1,:]...)
        @printf("  time: %.1fs\n\n", t); flush(stdout)
    end

    println("True slopes:  [0.8000, 0.8000, 0.8000]")
    println("True intcpts: ", round.(par_true.a_Q[1,:], digits=4))
end

main()

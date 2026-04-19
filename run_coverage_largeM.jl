using Serialization, Printf, Statistics
using Plots; gr()

include("ABB_three_period.jl")

# ── Run with larger M to reduce E-step MC noise ──────────────────
# Compare M=50 (baseline) vs M=200 vs M=500
# Focus on N=500 to keep runtime reasonable

L = 3; K = 2; sigma_y = 1.0
tau = collect(range(1/(L+1), stop=L/(L+1), length=L))
par_true = make_true_params_linear(tau=tau, sigma_y=sigma_y, K=K)

all_results = Dict{Int, Any}()

for M in [50, 200, 500]
    println("\n>>> N=500, M=$M <<<\n")
    res = run_comparison(N=500, K=2, L=3, maxiter=100, n_draws=200, M=M,
                         nonlinear=false, seed=42, methods=[:qr, :mle])
    serialize("results_M$(M)_N500.jls", res)
    all_results[M] = res
end

# ── Report monotonicity violations ────────────────────────────────
open("results_monotonicity.txt", "w") do io
    header = "\nMonotonicity violations (|Δll| > 0.01) by M and method:"
    println(header); println(io, header)
    println(io, "="^60)
    @printf(io, "%6s %8s %12s %12s\n", "M", "Method", "Violations", "Out of")
    println(io, "-"^60)
    for M in [50, 200, 500]
        res = all_results[M]
        for meth in [:qr, :mle]
            haskey(res, meth) || continue
            dll = diff(res[meth].ll)
            nv = count(dll .< -0.01)
            @printf(io, "%6d %8s %12d %12d\n", M, uppercase(string(meth)), nv, length(dll))
            @printf("%6d %8s %12d %12d\n", M, uppercase(string(meth)), nv, length(dll))
        end
    end

    # ── Coverage computation ──────────────────────────────────────
    deltas = range(0.0, 0.15, length=200)

    println(io, "\n" * "="^60)
    println(io, "  Coverage P(|slope_est - slope_true| ≤ δ) for N=500")
    println(io, "="^60)

    plot_data = Dict{Tuple{Int,Symbol}, Vector{Float64}}()

    for M in [50, 200, 500]
        res = all_results[M]
        for meth in [:qr, :mle]
            haskey(res, meth) || continue
            hist = res[meth].hist
            S = size(hist.a_Q, 3)
            S2 = div(S, 2)
            rng = (S - S2 + 1):S

            cov_avg = zeros(length(deltas))
            for l in 1:L
                slopes = vec(hist.a_Q[2, l, rng]) ./ sigma_y
                true_slope = par_true.a_Q[2, l] / sigma_y
                errors = abs.(slopes .- true_slope)
                cov_l = [mean(errors .≤ d) for d in deltas]
                cov_avg .+= cov_l
            end
            cov_avg ./= L
            plot_data[(M, meth)] = cov_avg
        end
    end

    key_deltas = [0.01, 0.02, 0.03, 0.05, 0.07, 0.10, 0.15]
    for M in [50, 200, 500]
        println(io, "\nM = $M:")
        @printf(io, "  %8s", "δ")
        for d in key_deltas; @printf(io, " %6.3f", d); end
        println(io)
        for meth in [:qr, :mle]
            haskey(all_results[M], meth) || continue
            cov = plot_data[(M, meth)]
            @printf(io, "  %8s", uppercase(string(meth)))
            for d in key_deltas
                idx = argmin(abs.(collect(deltas) .- d))
                @printf(io, " %6.3f", cov[idx])
            end
            println(io)
        end
    end
end

println("\nResults written to results_monotonicity.txt")

# ── Plots ─────────────────────────────────────────────────────────
deltas_vec = collect(range(0.0, 0.15, length=200))

# Plot: Coverage by M
p1 = plot(size=(700, 500), xlabel="δ", ylabel="Coverage P(|slope - true| ≤ δ)",
          title="N=500: Effect of M on Coverage", legend=:bottomright)
styles = Dict(50 => :dot, 200 => :dash, 500 => :solid)
for M in [50, 200, 500]
    for (meth, col) in [(:qr, :blue), (:mle, :red)]
        haskey(all_results[M], meth) || continue
        plot!(p1, deltas_vec, plot_data[(M, meth)],
              label="$(uppercase(string(meth))) M=$M",
              color=col, linestyle=styles[M], lw=2)
    end
end
savefig(p1, "coverage_by_M.png")
println("Saved coverage_by_M.png")

# Plot: Log-likelihood paths for each M
p2 = plot(layout=(1,3), size=(1200, 400),
          title=["M=50" "M=200" "M=500"],
          xlabel="Iteration", ylabel="Log-likelihood")
for (j, M) in enumerate([50, 200, 500])
    for (meth, col, ls) in [(:qr, :blue, :solid), (:mle, :red, :dash)]
        haskey(all_results[M], meth) || continue
        plot!(p2[j], all_results[M][meth].ll,
              label=uppercase(string(meth)), color=col, linestyle=ls, lw=1.5)
    end
end
savefig(p2, "loglik_paths_by_M.png")
println("Saved loglik_paths_by_M.png")

println("\nDone!")

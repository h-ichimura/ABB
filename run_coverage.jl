using Serialization, Printf, Statistics
using Plots; gr()

include("ABB_three_period.jl")

# ── Run all sample sizes ─────────────────────────────────────────
all_results = Dict{Int, Any}()

for N in [200, 500, 1000]
    println("\n>>> Running N=$N <<<\n")
    res = run_comparison(N=N, K=2, L=3, maxiter=100, n_draws=200, M=50,
                         nonlinear=false, seed=42, methods=[:qr, :mle])
    serialize("results_coverage_N$(N).jls", res)
    println("  Saved results_coverage_N$(N).jls")
    all_results[N] = res
end

# ── Setup true params ─────────────────────────────────────────────
L = 3; K = 2; sigma_y = 1.0
tau = collect(range(1/(L+1), stop=L/(L+1), length=L))
par_true = make_true_params_linear(tau=tau, sigma_y=sigma_y, K=K)

# ── Report number of estimates ────────────────────────────────────
open("results_coverage.txt", "w") do io
    for N in [200, 500, 1000]
        res = all_results[N]
        for meth in [:qr, :mle]
            haskey(res, meth) || continue
            S = size(res[meth].hist.a_Q, 3)
            S2 = div(S, 2)
            msg = "N=$N, $(uppercase(string(meth))): S=$S iterations, using last S̃=$S2 as estimates"
            println(msg); println(io, msg)
        end
    end

    # ── Coverage computation ──────────────────────────────────────
    deltas = range(0.0, 0.15, length=200)

    println(io, "\n" * "="^70)
    println(io, "  Coverage probability: P(|slope_est - slope_true| ≤ δ)")
    println(io, "="^70)

    plot_data = Dict{Tuple{Int,Symbol}, Vector{Float64}}()
    plot_data_byq = Dict{Tuple{Int,Symbol,Int}, Vector{Float64}}()

    for N in [200, 500, 1000]
        res = all_results[N]
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
                plot_data_byq[(N, meth, l)] = cov_l
                cov_avg .+= cov_l
            end
            cov_avg ./= L
            plot_data[(N, meth)] = cov_avg
        end
    end

    # Print key delta values
    key_deltas = [0.01, 0.02, 0.03, 0.05, 0.07, 0.10, 0.15]
    for N in [200, 500, 1000]
        println(io, "\nN = $N:")
        @printf(io, "  %8s", "δ")
        for d in key_deltas; @printf(io, " %6.3f", d); end
        println(io)
        for meth in [:qr, :mle]
            haskey(all_results[N], meth) || continue
            cov = plot_data[(N, meth)]
            @printf(io, "  %8s", uppercase(string(meth)))
            for d in key_deltas
                idx = argmin(abs.(collect(deltas) .- d))
                @printf(io, " %6.3f", cov[idx])
            end
            println(io)
        end
    end
end

println("\nResults written to results_coverage.txt")

# ── Plots ─────────────────────────────────────────────────────────
deltas_vec = collect(range(0.0, 0.15, length=200))

# Plot 1: Coverage averaged over quantile levels, one panel per N
p1 = plot(layout=(1,3), size=(1200, 400),
          title=["N=200" "N=500" "N=1000"],
          xlabel="δ", ylabel="Coverage P(|slope - true| ≤ δ)")

for (j, N) in enumerate([200, 500, 1000])
    for (meth, col, ls) in [(:qr, :blue, :solid), (:mle, :red, :dash)]
        haskey(all_results[N], meth) || continue
        plot!(p1[j], deltas_vec, plot_data[(N, meth)],
              label=uppercase(string(meth)), color=col, linestyle=ls, lw=2)
    end
end
savefig(p1, "coverage_slope_byN.png")
println("Saved coverage_slope_byN.png")

# Plot 2: Coverage by quantile level for N=1000
p2 = plot(layout=(1,3), size=(1200, 400),
          title=["τ=0.25" "τ=0.50" "τ=0.75"],
          xlabel="δ", ylabel="Coverage")

for (j, l) in enumerate(1:L)
    for (meth, col, ls) in [(:qr, :blue, :solid), (:mle, :red, :dash)]
        haskey(plot_data_byq, (1000, meth, l)) || continue
        plot!(p2[j], deltas_vec, plot_data_byq[(1000, meth, l)],
              label=uppercase(string(meth)), color=col, linestyle=ls, lw=2)
    end
end
savefig(p2, "coverage_slope_N1000_byquantile.png")
println("Saved coverage_slope_N1000_byquantile.png")

# Plot 3: All N on one plot (averaged over quantile levels)
p3 = plot(size=(700, 500), xlabel="δ", ylabel="Coverage P(|slope - true| ≤ δ)",
          title="Slope Coverage: QR vs MLE", legend=:bottomright)
styles = Dict(200 => :dot, 500 => :dash, 1000 => :solid)
for N in [200, 500, 1000]
    for (meth, col) in [(:qr, :blue), (:mle, :red)]
        haskey(all_results[N], meth) || continue
        plot!(p3, deltas_vec, plot_data[(N, meth)],
              label="$(uppercase(string(meth))) N=$N",
              color=col, linestyle=styles[N], lw=2)
    end
end
savefig(p3, "coverage_slope_all.png")
println("Saved coverage_slope_all.png")

println("\nDone!")

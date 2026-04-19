#=
collect_results.jl — Aggregate Monte Carlo summaries across seeds.

Run after all HPC jobs complete. Computes bias, std dev, RMSE, and coverage
for each sample size N and each method (QR, Exact ML).

Usage: julia collect_results.jl
=#

using Serialization, Printf, Statistics

include("../ABB_three_period.jl")

L = 3; K = 2; sigma_y = 1.0
tau = collect(range(1/(L+1), stop=L/(L+1), length=L))
par_true = make_true_params_linear(tau=tau, sigma_y=sigma_y, K=K)

true_slope = [par_true.a_Q[2, l] / sigma_y for l in 1:L]
true_intercept = [par_true.a_Q[1, l] for l in 1:L]
true_a_init = copy(par_true.a_init)
true_a_eps = copy(par_true.a_eps)

# Load all summaries
files = filter(f -> startswith(f, "summary_") && endswith(f, ".jls"), readdir("."))
println("Found $(length(files)) summary files")

function parse_filename(f)
    m = match(r"summary_N(\d+)_seed(\d+)\.jls", f)
    isnothing(m) && return nothing
    (parse(Int, m[1]), parse(Int, m[2]))
end

grouped = Dict{Int, Vector{Any}}()
for f in files
    key = parse_filename(f)
    isnothing(key) && continue
    N, seed = key
    s = deserialize(f)
    if !haskey(grouped, N); grouped[N] = []; end
    push!(grouped[N], s)
end

Ns = sort(collect(keys(grouped)))

open("results_hpc_summary.txt", "w") do io
    println(io, "ABB Monte Carlo: QR vs Exact ML")
    println(io, "True slopes: ", join([@sprintf("%.4f", s) for s in true_slope], ", "))
    println(io, "True intercepts: ", join([@sprintf("%.4f", s) for s in true_intercept], ", "))
    println(io, "True a_init: ", join([@sprintf("%.4f", s) for s in true_a_init], ", "))
    println(io, "True a_eps: ", join([@sprintf("%.4f", s) for s in true_a_eps], ", "))
    println(io, "="^70)

    for N in Ns
        runs = grouped[N]
        R = length(runs)
        println(io, "\n>>> N=$N, R=$R replications <<<")

        for meth in [:qr, :exact]
            slope_ests = zeros(R, L)
            intercept_ests = zeros(R, L)
            a_init_ests = zeros(R, L)
            a_eps_ests = zeros(R, L)
            b1_init_ests = zeros(R)
            bL_init_ests = zeros(R)
            b1_eps_ests = zeros(R)
            bL_eps_ests = zeros(R)
            times = zeros(R)

            valid = 0
            for (r, run) in enumerate(runs)
                haskey(run, meth) || continue
                valid += 1
                slope_ests[valid, :] = run[meth].slope
                intercept_ests[valid, :] = run[meth].intercept
                a_init_ests[valid, :] = run[meth].a_init
                a_eps_ests[valid, :] = run[meth].a_eps
                b1_init_ests[valid] = run[meth].b1_init
                bL_init_ests[valid] = run[meth].bL_init
                b1_eps_ests[valid] = run[meth].b1_eps
                bL_eps_ests[valid] = run[meth].bL_eps
                times[valid] = (meth == :qr) ? run[:time_qr] : run[:time_exact]
            end
            valid == 0 && continue
            slope_ests = slope_ests[1:valid, :]
            intercept_ests = intercept_ests[1:valid, :]
            a_init_ests = a_init_ests[1:valid, :]
            a_eps_ests = a_eps_ests[1:valid, :]
            b1_init_ests = b1_init_ests[1:valid]
            bL_init_ests = bL_init_ests[1:valid]
            b1_eps_ests = b1_eps_ests[1:valid]
            bL_eps_ests = bL_eps_ests[1:valid]

            println(io, "\n  $(uppercase(string(meth))) ($valid replications, avg $(round(mean(times), digits=1))s):")

            function show_param(name, ests, true_val)
                bias = mean(ests) - true_val
                sd = std(ests)
                rmse = sqrt(mean((ests .- true_val).^2))
                @printf(io, "    %-22s true=%+8.4f  mean=%+8.4f  bias=%+8.4f  std=%.4f  RMSE=%.4f\n",
                        name, true_val, mean(ests), bias, sd, rmse)
            end

            println(io, "  -- Transition slopes --")
            for l in 1:L
                show_param(@sprintf("slope τ=%.2f", tau[l]), slope_ests[:, l], true_slope[l])
            end
            println(io, "  -- Transition intercepts --")
            for l in 1:L
                show_param(@sprintf("intcpt τ=%.2f", tau[l]), intercept_ests[:, l], true_intercept[l])
            end
            println(io, "  -- η_1 quantiles --")
            for l in 1:L
                show_param(@sprintf("a_init τ=%.2f", tau[l]), a_init_ests[:, l], true_a_init[l])
            end
            show_param("b1_init", b1_init_ests, par_true.b1_init)
            show_param("bL_init", bL_init_ests, par_true.bL_init)
            println(io, "  -- ε quantiles --")
            for l in 1:L
                show_param(@sprintf("a_eps τ=%.2f", tau[l]), a_eps_ests[:, l], true_a_eps[l])
            end
            show_param("b1_eps", b1_eps_ests, par_true.b1_eps)
            show_param("bL_eps", bL_eps_ests, par_true.bL_eps)
        end
    end
end

println("Written results_hpc_summary.txt")

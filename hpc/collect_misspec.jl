#=
collect_misspec.jl — Collect and analyze misspecified AR(2) comparison results.

Reports KS and L1 distances for MLE, C²-QR, ABB-QR at each conditioning value,
averaged over seeds.
=#

using Serialization, Printf, Statistics

results_dir = length(ARGS) >= 1 ? ARGS[1] : "results"

# Discover all misspec result files
files = filter(f -> startswith(f, "misspec_") && endswith(f, ".jls"),
               readdir(results_dir))
println("Found $(length(files)) result files in $results_dir")

# Group by configuration
configs = Dict{String, Vector{Dict{Symbol,Any}}}()
for f in files
    d = deserialize(joinpath(results_dir, f))
    key = @sprintf("r1%.1f_r2%.1f_N%d_T%d", d[:rho1], d[:rho2], d[:N], d[:T])
    if !haskey(configs, key)
        configs[key] = Dict{Symbol,Any}[]
    end
    push!(configs[key], d)
end

println("\nConfigurations found: $(length(configs))")
for (key, dicts) in sort(collect(configs), by=x->x[1])
    println("  $key: $(length(dicts)) seeds")
end

# Report for each configuration
for (key, dicts) in sort(collect(configs), by=x->x[1])
    d1 = dicts[1]
    S = length(dicts)
    n_cond = length(d1[:y_cond_vals])
    y_conds = d1[:y_cond_vals]

    println("\n", "="^80)
    @printf("  DGP: y_t = %.1f y_{t-1} + (%.1f) y_{t-2} + %.1f v_t\n",
            d1[:rho1], d1[:rho2], d1[:sigma_v])
    @printf("  N=%d, T=%d, S=%d seeds\n", d1[:N], d1[:T], S)
    println("="^80)

    # Count MLE failures
    n_fail = count(d -> d[:ml_failed], dicts)
    n_fail > 0 && @printf("  MLE failures: %d/%d (%.0f%%)\n", n_fail, S, 100*n_fail/S)

    # Collect KS and L1
    ks_ml = zeros(S, n_cond); l1_ml = zeros(S, n_cond)
    ks_c2 = zeros(S, n_cond); l1_c2 = zeros(S, n_cond)
    ks_abb = zeros(S, n_cond); l1_abb = zeros(S, n_cond)

    for (s, d) in enumerate(dicts)
        ks_ml[s,:] = d[:ml_ks]; l1_ml[s,:] = d[:ml_l1]
        ks_c2[s,:] = d[:c2_ks]; l1_c2[s,:] = d[:c2_l1]
        ks_abb[s,:] = d[:abb_ks]; l1_abb[s,:] = d[:abb_l1]
    end

    # Exclude MLE failures
    good = [!d[:ml_failed] for d in dicts]

    # Per conditioning value
    @printf("\n%-8s  %10s  %10s  %10s  %10s  %10s  %10s  %8s\n",
            "y_{t-1}", "KS_MLE", "KS_C²", "KS_ABB", "L1_MLE", "L1_C²", "L1_ABB", "Best_L1")
    println("-"^85)
    for j in 1:n_cond
        km = mean(ks_ml[good,j]); kc = mean(ks_c2[:,j]); ka = mean(ks_abb[:,j])
        lm = mean(l1_ml[good,j]); lc = mean(l1_c2[:,j]); la = mean(l1_abb[:,j])
        best = lm < lc && lm < la ? "MLE" : (lc < la ? "C²" : "ABB")
        @printf("%-8.1f  %10.4f  %10.4f  %10.4f  %10.4f  %10.4f  %10.4f  %8s\n",
                y_conds[j], km, kc, ka, lm, lc, la, best)
    end

    # Averages
    km = mean(ks_ml[good,:]); kc = mean(ks_c2); ka = mean(ks_abb)
    lm = mean(l1_ml[good,:]); lc = mean(l1_c2); la = mean(l1_abb)
    println()
    @printf("Average KS:  MLE=%.4f  C²=%.4f  ABB=%.4f\n", km, kc, ka)
    @printf("Average L1:  MLE=%.4f  C²=%.4f  ABB=%.4f\n", lm, lc, la)
    @printf("Ratios:  C²/MLE=(%.2f,%.2f)  ABB/MLE=(%.2f,%.2f)  ABB/C²=(%.2f,%.2f)\n",
            kc/km, lc/lm, ka/km, la/lm, ka/kc, la/lc)

    # Timing
    t_ml = mean(d[:ml_time] for d in dicts)
    t_c2 = mean(d[:c2_time] for d in dicts)
    t_abb = mean(d[:abb_time] for d in dicts)
    @printf("\nAvg time: MLE=%.0fs  C²-QR=%.0fs  ABB-QR=%.0fs\n", t_ml, t_c2, t_abb)
end

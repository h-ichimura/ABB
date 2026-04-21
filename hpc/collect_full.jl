#=
collect_full.jl — Collect profiled MLE vs QR results.
Outputs: full LaTeX tables (all parameters) and coverage probability data.

Usage: julia collect_full.jl [results_dir]
=#

using Serialization, Printf, Statistics

results_dir = length(ARGS) >= 1 ? ARGS[1] : "results"

files = filter(f -> startswith(f, "profiled_") && endswith(f, ".jls"), readdir(results_dir))
isempty(files) && error("No profiled results found in $results_dir")

# Group by (rho, N, T)
results = Dict{Tuple{Float64,Int,Int}, Vector{Dict}}()
for f in files
    d = deserialize(joinpath(results_dir, f))
    key = (d[:rho], d[:N], d[:T])
    haskey(results, key) || (results[key] = Dict[])
    push!(results[key], d)
end

@printf("Loaded %d files across %d configurations.\n", length(files), length(results))
for key in sort(collect(keys(results)))
    @printf("  ρ=%.2f N=%d T=%d: %d seeds\n", key..., length(results[key]))
end

# All parameters: name, ML extractor, QR extractor, true extractor
params = [
    # Transition median quantile (3 Hermite coefficients)
    ("q_{50,0}",  d->d[:ml_a_Q][1,2], d->d[:qr_a_Q][1,2], d->d[:a_Q_true][1,2]),
    ("q_{50,1}(\\rho)", d->d[:ml_a_Q][2,2], d->d[:qr_a_Q][2,2], d->d[:a_Q_true][2,2]),
    ("q_{50,2}",  d->d[:ml_a_Q][3,2], d->d[:qr_a_Q][3,2], d->d[:a_Q_true][3,2]),
    # Transition 25th quantile
    ("q_{25,0}",  d->d[:ml_a_Q][1,1], d->d[:qr_a_Q][1,1], d->d[:a_Q_true][1,1]),
    ("q_{25,1}",  d->d[:ml_a_Q][2,1], d->d[:qr_a_Q][2,1], d->d[:a_Q_true][2,1]),
    ("q_{25,2}",  d->d[:ml_a_Q][3,1], d->d[:qr_a_Q][3,1], d->d[:a_Q_true][3,1]),
    # Transition 75th quantile
    ("q_{75,0}",  d->d[:ml_a_Q][1,3], d->d[:qr_a_Q][1,3], d->d[:a_Q_true][1,3]),
    ("q_{75,1}",  d->d[:ml_a_Q][2,3], d->d[:qr_a_Q][2,3], d->d[:a_Q_true][2,3]),
    ("q_{75,2}",  d->d[:ml_a_Q][3,3], d->d[:qr_a_Q][3,3], d->d[:a_Q_true][3,3]),
    # Initial distribution
    ("a_{init,1}", d->d[:ml_a_init][1], d->d[:qr_a_init][1], d->d[:a_init_true][1]),
    ("a_{init,2}", d->d[:ml_a_init][2], d->d[:qr_a_init][2], d->d[:a_init_true][2]),
    ("a_{init,3}", d->d[:ml_a_init][3], d->d[:qr_a_init][3], d->d[:a_init_true][3]),
    # Epsilon
    ("a_{\\varepsilon,1}", d->d[:ml_a_eps1], d->d[:qr_a_eps1], d->d[:a_eps1_true]),
    ("a_{\\varepsilon,3}", d->d[:ml_a_eps3], d->d[:qr_a_eps3], d->d[:a_eps3_true]),
]

function get_stats(data, fn_ml, fn_qr, truth)
    ml_vals = Float64[]; qr_vals = Float64[]
    for d in data
        isnan(d[:ml_nll]) && continue
        ml_v = fn_ml(d); qr_v = fn_qr(d)
        (isnan(ml_v) || isinf(ml_v)) && continue
        push!(ml_vals, ml_v)
        push!(qr_vals, qr_v)
    end
    isempty(ml_vals) && return nothing
    mb = mean(ml_vals) - truth; ms = std(ml_vals); mr = sqrt(mb^2 + ms^2)
    qb = mean(qr_vals) - truth; qs = std(qr_vals); qrr = sqrt(qb^2 + qs^2)
    eff = mr > 0 ? qrr / mr : NaN
    # Coverage: fraction within δ of truth
    cp_ml_05 = mean(abs.(ml_vals .- truth) .< 0.05)
    cp_qr_05 = mean(abs.(qr_vals .- truth) .< 0.05)
    cp_ml_10 = mean(abs.(ml_vals .- truth) .< 0.10)
    cp_qr_10 = mean(abs.(qr_vals .- truth) .< 0.10)
    # Median absolute deviation (robust to outliers)
    mad_ml = median(abs.(ml_vals .- truth))
    mad_qr = median(abs.(qr_vals .- truth))
    (mb=mb, ms=ms, mr=mr, qb=qb, qs=qs, qrr=qrr, eff=eff,
     n_valid=length(ml_vals),
     cp_ml_05=cp_ml_05, cp_qr_05=cp_qr_05,
     cp_ml_10=cp_ml_10, cp_qr_10=cp_qr_10,
     mad_ml=mad_ml, mad_qr=mad_qr,
     ml_vals=ml_vals, qr_vals=qr_vals)
end

# ================================================================
#  LATEX TABLES — all parameters, one table per (rho, T)
# ================================================================

println("%"^80)
println("% FULL LATEX TABLES — all 14 parameters")
println("%"^80)

rho_vals = sort(unique(k[1] for k in keys(results)))
T_vals = sort(unique(k[3] for k in keys(results)))
N_vals = sort(unique(k[2] for k in keys(results)))

for ρ_val in rho_vals, T_obs in T_vals
    configs = [(ρ_val, N, T_obs) for N in N_vals if haskey(results, (ρ_val, N, T_obs))]
    isempty(configs) && continue

    println()
    @printf("\\begin{table}[H]\n\\centering\n\\small\n")
    @printf("\\caption{Profiled MLE vs.\\ QR, \$\\rho = %.2f\$, \$T = %d\$, 1000 replications.}\n", ρ_val, T_obs)
    @printf("\\label{tab:full_rho%.0f_T%d}\n", 100*ρ_val, T_obs)
    println("\\begin{tabular}{ll|r|ccc|ccc|c}")
    println("\\toprule")
    println(" & & & \\multicolumn{3}{c|}{MLE} & \\multicolumn{3}{c|}{QR} & \\\\")
    println("\$N\$ & Parameter & True & bias & std & RMSE & bias & std & RMSE & QR/ML \\\\")
    println("\\midrule")

    for (idx, key) in enumerate(configs)
        _, N, _ = key
        data = results[key]

        for (pidx, (name, fn_ml, fn_qr, fn_true)) in enumerate(params)
            truth = fn_true(data[1])
            st = get_stats(data, fn_ml, fn_qr, truth)
            st === nothing && continue

            n_label = pidx == 1 ? @sprintf("\\multirow{14}{*}{%d}", N) : ""

            # Truncate huge std values for display
            ml_std_str = st.ms > 10 ? @sprintf("%.1f", st.ms) : @sprintf("%.3f", st.ms)
            ml_rmse_str = st.mr > 10 ? @sprintf("%.1f", st.mr) : @sprintf("%.3f", st.mr)
            qr_std_str = st.qs > 10 ? @sprintf("%.1f", st.qs) : @sprintf("%.3f", st.qs)
            qr_rmse_str = st.qrr > 10 ? @sprintf("%.1f", st.qrr) : @sprintf("%.3f", st.qrr)

            eff_str = st.eff > 1.05 ? @sprintf("\\textbf{%.2f}", st.eff) :
                      st.eff < 0.01 ? "0.00" : @sprintf("%.2f", st.eff)

            sign_ml = st.mb >= 0 ? "+" : "-"
            sign_qr = st.qb >= 0 ? "+" : "-"
            ml_bias_str = abs(st.mb) > 10 ? @sprintf("%.1f", abs(st.mb)) : @sprintf("%.3f", abs(st.mb))
            qr_bias_str = abs(st.qb) > 10 ? @sprintf("%.1f", abs(st.qb)) : @sprintf("%.3f", abs(st.qb))

            truth_str = @sprintf("%.3f", truth)

            @printf("%s & \$%s\$ & %s & \$%s\$%s & %s & %s & \$%s\$%s & %s & %s & %s \\\\\n",
                    n_label, name, truth_str,
                    sign_ml, ml_bias_str, ml_std_str, ml_rmse_str,
                    sign_qr, qr_bias_str, qr_std_str, qr_rmse_str, eff_str)
        end
        idx < length(configs) && println("\\midrule")
    end

    println("\\bottomrule")
    println("\\end{tabular}")
    println("\\end{table}")
    println()
end

# ================================================================
#  COVERAGE PROBABILITY TABLE
# ================================================================

println()
println("%"^80)
println("% COVERAGE PROBABILITY: P(|θ̂ - θ₀| < δ)")
println("%"^80)

# Select key parameters for coverage table
key_params = [1, 2, 3, 10, 11, 12, 13, 14]  # q50_0, rho, q50_2, init1-3, ae1, ae3
key_names = ["q_{50,0}", "\\rho", "q_{50,2}",
             "a_{init,1}", "a_{init,2}", "a_{init,3}",
             "a_{\\varepsilon,1}", "a_{\\varepsilon,3}"]

for ρ_val in rho_vals, T_obs in T_vals
    configs = [(ρ_val, N, T_obs) for N in N_vals if haskey(results, (ρ_val, N, T_obs))]
    isempty(configs) && continue

    println()
    @printf("\\begin{table}[H]\n\\centering\n\\small\n")
    @printf("\\caption{Coverage probability \$P(|\\hat\\theta - \\theta_0| < \\delta)\$, \$\\rho = %.2f\$, \$T = %d\$.}\n", ρ_val, T_obs)
    @printf("\\label{tab:coverage_rho%.0f_T%d}\n", 100*ρ_val, T_obs)
    println("\\begin{tabular}{ll|cc|cc|cc}")
    println("\\toprule")
    println(" & & \\multicolumn{2}{c|}{\$\\delta = 0.05\$} & \\multicolumn{2}{c|}{\$\\delta = 0.10\$} & \\multicolumn{2}{c}{MAD} \\\\")
    println("\$N\$ & Parameter & MLE & QR & MLE & QR & MLE & QR \\\\")
    println("\\midrule")

    for (idx, key) in enumerate(configs)
        _, N, _ = key
        data = results[key]

        for (kidx, pidx) in enumerate(key_params)
            name, fn_ml, fn_qr, fn_true = params[pidx]
            truth = fn_true(data[1])
            st = get_stats(data, fn_ml, fn_qr, truth)
            st === nothing && continue

            n_label = kidx == 1 ? @sprintf("\\multirow{%d}{*}{%d}", length(key_params), N) : ""

            # Bold the better coverage
            ml05 = st.cp_ml_05; qr05 = st.cp_qr_05
            ml10 = st.cp_ml_10; qr10 = st.cp_qr_10
            ml05_str = ml05 >= qr05 ? @sprintf("\\textbf{%.3f}", ml05) : @sprintf("%.3f", ml05)
            qr05_str = qr05 > ml05 ? @sprintf("\\textbf{%.3f}", qr05) : @sprintf("%.3f", qr05)
            ml10_str = ml10 >= qr10 ? @sprintf("\\textbf{%.3f}", ml10) : @sprintf("%.3f", ml10)
            qr10_str = qr10 > ml10 ? @sprintf("\\textbf{%.3f}", qr10) : @sprintf("%.3f", qr10)

            # MAD
            mad_ml_str = st.mad_ml > 1 ? @sprintf("%.2f", st.mad_ml) : @sprintf("%.4f", st.mad_ml)
            mad_qr_str = st.mad_qr > 1 ? @sprintf("%.2f", st.mad_qr) : @sprintf("%.4f", st.mad_qr)
            if st.mad_ml <= st.mad_qr
                mad_ml_str = "\\textbf{" * mad_ml_str * "}"
            else
                mad_qr_str = "\\textbf{" * mad_qr_str * "}"
            end

            @printf("%s & \$%s\$ & %s & %s & %s & %s & %s & %s \\\\\n",
                    n_label, name, ml05_str, qr05_str, ml10_str, qr10_str, mad_ml_str, mad_qr_str)
        end
        idx < length(configs) && println("\\midrule")
    end

    println("\\bottomrule")
    println("\\end{tabular}")
    println("\\end{table}")
end

# ================================================================
#  COVERAGE PROBABILITY DATA FOR PLOTTING
#  Output: tab-separated file for each (rho, T)
# ================================================================

println()
println("%"^80)
println("% COVERAGE DATA FOR PLOTTING")
println("% Format: N param delta CP_ML CP_QR")
println("%"^80)

deltas = collect(0.01:0.01:0.50)

for ρ_val in rho_vals, T_obs in T_vals
    configs = [(ρ_val, N, T_obs) for N in N_vals if haskey(results, (ρ_val, N, T_obs))]
    isempty(configs) && continue

    @printf("\n%% Coverage curve data: rho=%.2f T=%d\n", ρ_val, T_obs)
    @printf("%% N\tparam\tdelta\tCP_ML\tCP_QR\n")

    for key in configs
        _, N, _ = key
        data = results[key]

        for (pidx, (name, fn_ml, fn_qr, fn_true)) in enumerate(params)
            truth = fn_true(data[1])
            st = get_stats(data, fn_ml, fn_qr, truth)
            st === nothing && continue

            for δ in deltas
                cp_ml = mean(abs.(st.ml_vals .- truth) .< δ)
                cp_qr = mean(abs.(st.qr_vals .- truth) .< δ)
                @printf("%% %d\t%s\t%.2f\t%.4f\t%.4f\n", N, name, δ, cp_ml, cp_qr)
            end
        end
    end
end

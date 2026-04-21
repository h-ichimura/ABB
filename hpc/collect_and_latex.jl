#=
collect_and_latex.jl — Collect profiled MLE vs QR results and output LaTeX tables.

Usage: julia collect_and_latex.jl [results_dir]

Outputs LaTeX tables for paper_ABB_v3.tex:
  - One table per (rho, T) with rows for each N
  - Columns: bias, std, RMSE for ML and QR, efficiency ratio
  - Coverage probabilities
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
println()

# Parameter extraction
function get_stats(data, param_fn_ml, param_fn_qr, truth)
    ml_vals = Float64[]; qr_vals = Float64[]
    for d in data
        isnan(d[:ml_nll]) && continue
        push!(ml_vals, param_fn_ml(d))
        push!(qr_vals, param_fn_qr(d))
    end
    isempty(ml_vals) && return nothing
    mb = mean(ml_vals) - truth; ms = std(ml_vals); mr = sqrt(mb^2 + ms^2)
    qb = mean(qr_vals) - truth; qs = std(qr_vals); qrr = sqrt(qb^2 + qs^2)
    eff = mr > 0 ? qrr / mr : NaN
    n_nan = length(data) - length(ml_vals)
    # Coverage
    cp_ml_05 = mean(abs.(ml_vals .- truth) .< 0.05)
    cp_qr_05 = mean(abs.(qr_vals .- truth) .< 0.05)
    cp_ml_10 = mean(abs.(ml_vals .- truth) .< 0.10)
    cp_qr_10 = mean(abs.(qr_vals .- truth) .< 0.10)
    (mb=mb, ms=ms, mr=mr, qb=qb, qs=qs, qrr=qrr, eff=eff,
     n_valid=length(ml_vals), n_nan=n_nan,
     cp_ml_05=cp_ml_05, cp_qr_05=cp_qr_05,
     cp_ml_10=cp_ml_10, cp_qr_10=cp_qr_10)
end

params = [
    ("ρ",              d -> d[:ml_a_Q][2,2],   d -> d[:qr_a_Q][2,2],   d -> d[:a_Q_true][2,2]),
    ("a_{ε,3}",        d -> d[:ml_a_eps3],     d -> d[:qr_a_eps3],     d -> d[:a_eps3_true]),
    ("a_{ε,1}",        d -> d[:ml_a_eps1],     d -> d[:qr_a_eps1],     d -> d[:a_eps1_true]),
    ("a_{init}[1]",    d -> d[:ml_a_init][1],  d -> d[:qr_a_init][1],  d -> d[:a_init_true][1]),
    ("a_{init}[2]",    d -> d[:ml_a_init][2],  d -> d[:qr_a_init][2],  d -> d[:a_init_true][2]),
    ("a_{init}[3]",    d -> d[:ml_a_init][3],  d -> d[:qr_a_init][3],  d -> d[:a_init_true][3]),
]

# ================================================================
#  TEXT SUMMARY
# ================================================================

println("="^90)
println("  PROFILED MLE vs QR — FULL RESULTS")
println("="^90)

for key in sort(collect(keys(results)))
    ρ_val, N, T_obs = key
    data = results[key]
    S = length(data)
    @printf("\nρ=%.2f, N=%d, T=%d, S=%d\n", ρ_val, N, T_obs, S)
    @printf("%-14s %8s  %8s %8s %8s  %8s %8s %8s  %5s  %6s %6s %6s %6s\n",
            "Param", "True", "ML_bias", "ML_std", "ML_RMSE",
            "QR_bias", "QR_std", "QR_RMSE", "QR/ML",
            "CP05m", "CP05q", "CP10m", "CP10q")
    println("-"^120)

    for (name, fn_ml, fn_qr, fn_true) in params
        truth = fn_true(data[1])
        st = get_stats(data, fn_ml, fn_qr, truth)
        st === nothing && continue
        eff_str = st.eff > 1.0 ? @sprintf("\\bf%.2f", st.eff) : @sprintf("%.2f", st.eff)
        @printf("%-14s %8.4f  %+8.4f %8.4f %8.4f  %+8.4f %8.4f %8.4f  %5.2f  %6.3f %6.3f %6.3f %6.3f\n",
                name, truth, st.mb, st.ms, st.mr, st.qb, st.qs, st.qrr, st.eff,
                st.cp_ml_05, st.cp_qr_05, st.cp_ml_10, st.cp_qr_10)
    end

    # Timing
    ml_times = [d[:ml_time] for d in data if !isnan(d[:ml_nll])]
    qr_times = [d[:qr_time] for d in data]
    @printf("  Timing: MLE=%.0fs  QR=%.0fs  NaN=%d\n",
            mean(ml_times), mean(qr_times),
            count(d -> isnan(d[:ml_nll]), data))
end

# ================================================================
#  LATEX TABLES — one per (rho, T), rows for different N
# ================================================================

println("\n\n")
println("%"^80)
println("% LATEX TABLES — paste into paper_ABB_v3.tex")
println("%"^80)

rho_vals = sort(unique(k[1] for k in keys(results)))
T_vals = sort(unique(k[3] for k in keys(results)))
N_vals = sort(unique(k[2] for k in keys(results)))

for ρ_val in rho_vals, T_obs in T_vals
    configs = [(ρ_val, N, T_obs) for N in N_vals if haskey(results, (ρ_val, N, T_obs))]
    isempty(configs) && continue

    println()
    @printf("\\begin{table}[H]\n\\centering\n")
    @printf("\\caption{Profiled MLE vs.\\ QR, \$\\rho = %.2f\$, \$T = %d\$, 1000 replications.}\n", ρ_val, T_obs)
    @printf("\\label{tab:profiled_rho%.0f_T%d}\n", 100*ρ_val, T_obs)
    println("\\begin{tabular}{ll|ccc|ccc|c}")
    println("\\toprule")
    println(" & & \\multicolumn{3}{c|}{MLE} & \\multicolumn{3}{c|}{QR} & \\\\")
    println("N & Parameter & bias & std & RMSE & bias & std & RMSE & QR/ML \\\\")
    println("\\midrule")

    for (idx, key) in enumerate(configs)
        _, N, _ = key
        data = results[key]
        S = length(data)

        for (pidx, (name, fn_ml, fn_qr, fn_true)) in enumerate(params)
            truth = fn_true(data[1])
            st = get_stats(data, fn_ml, fn_qr, truth)
            st === nothing && continue

            n_label = pidx == 1 ? @sprintf("\\multirow{%d}{*}{%d}", length(params), N) : ""
            eff_str = st.eff > 1.05 ? @sprintf("\\textbf{%.2f}", st.eff) : @sprintf("%.2f", st.eff)

            sign_ml = st.mb >= 0 ? "+" : "-"
            sign_qr = st.qb >= 0 ? "+" : "-"

            @printf("%s & \$%s\$ & \$%s\$%.3f & %.3f & %.3f & \$%s\$%.3f & %.3f & %.3f & %s \\\\\n",
                    n_label, name,
                    sign_ml, abs(st.mb), st.ms, st.mr,
                    sign_qr, abs(st.qb), st.qs, st.qrr, eff_str)
        end
        idx < length(configs) && println("\\midrule")
    end

    println("\\bottomrule")
    println("\\end{tabular}")
    println("\\end{table}")
    println()
end

# ================================================================
#  SUMMARY TABLE: QR/ML RMSE ratios across all configs
# ================================================================

println()
println("\\begin{table}[H]")
println("\\centering")
println("\\caption{QR/ML RMSE ratios. Values \$> 1\$ (bold) indicate MLE is more efficient.}")
println("\\label{tab:efficiency_all}")
println("\\begin{tabular}{ll" * repeat("c", length(N_vals)) * "}")
println("\\toprule")
@printf(" & & \\multicolumn{%d}{c}{\$N\$} \\\\\n", length(N_vals))
print("\\(\\rho, T\\) & Parameter")
for N in N_vals; @printf(" & %d", N); end
println(" \\\\")
println("\\midrule")

for ρ_val in rho_vals, T_obs in T_vals
    for (pidx, (name, fn_ml, fn_qr, fn_true)) in enumerate(params)
        (name == "a_{ε,1}") && continue  # skip ae1 for brevity
        if pidx == 1
            @printf("\\multirow{%d}{*}{(%.2f, %d)}", length(params)-1, ρ_val, T_obs)
        end
        @printf(" & \$%s\$", name)
        for N in N_vals
            key = (ρ_val, N, T_obs)
            if haskey(results, key)
                data = results[key]
                truth = fn_true(data[1])
                st = get_stats(data, fn_ml, fn_qr, truth)
                if st !== nothing
                    if st.eff > 1.05
                        @printf(" & \\textbf{%.2f}", st.eff)
                    else
                        @printf(" & %.2f", st.eff)
                    end
                else
                    print(" & ---")
                end
            else
                print(" & ---")
            end
        end
        println(" \\\\")
    end
    println("\\midrule")
end

println("\\bottomrule")
println("\\end{tabular}")
println("\\end{table}")

# ================================================================
#  LOUIS MISSING INFORMATION (if available)
# ================================================================

println()
println("% Louis missing information (seed=1 results)")
for key in sort(collect(keys(results)))
    ρ_val, N, T_obs = key
    data = results[key]
    louis_data = filter(d -> d[:louis_Var_S] !== nothing, data)
    isempty(louis_data) && continue
    @printf("\n%% Louis: ρ=%.2f N=%d T=%d\n", ρ_val, N, T_obs)
    Var_S = louis_data[1][:louis_Var_S]
    if isa(Var_S, Matrix)
        for j in 1:size(Var_S,1)
            @printf("%%   param %d: Var=%.4f\n", j, Var_S[j,j])
        end
    else
        @printf("%%   Louis result type: %s\n", typeof(Var_S))
    end
end

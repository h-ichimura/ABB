#=
collect_abb_pw.jl — Collect MLE / QR results from
                    run_abb_pw_comparison.jl and produce summary tables.

Analog of collect_full.jl, but for the 2-method ABB piecewise-uniform
comparison (Section 3 of paper_ABB_v3.tex). SML was dropped — see
the header of run_abb_pw_comparison.jl for the rationale.

Usage: julia collect_abb_pw.jl [results_dir] [out_tex_file]
       (default: results_dir = "results", out = "tables_abb_pw.tex")
=#

using Serialization, Printf, Statistics

results_dir = length(ARGS) >= 1 ? ARGS[1] : "results"
out_tex     = length(ARGS) >= 2 ? ARGS[2] : "tables_abb_pw.tex"

files = filter(f -> startswith(f, "abbpw_") && endswith(f, ".jls"), readdir(results_dir))
isempty(files) && error("No abbpw_*.jls results found in $results_dir")

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

# ABB-PW parameters organised for reporting
params = [
    # Transition median row
    ("q_{50,0}",   d->d[:ml_a_Q][1,2], d->d[:qr_a_Q][1,2], d->d[:a_Q_true][1,2]),
    ("q_{50,1}(\\rho)", d->d[:ml_a_Q][2,2], d->d[:qr_a_Q][2,2], d->d[:a_Q_true][2,2]),
    ("q_{50,2}",   d->d[:ml_a_Q][3,2], d->d[:qr_a_Q][3,2], d->d[:a_Q_true][3,2]),
    # Transition 25th
    ("q_{25,0}",   d->d[:ml_a_Q][1,1], d->d[:qr_a_Q][1,1], d->d[:a_Q_true][1,1]),
    ("q_{25,1}",   d->d[:ml_a_Q][2,1], d->d[:qr_a_Q][2,1], d->d[:a_Q_true][2,1]),
    ("q_{25,2}",   d->d[:ml_a_Q][3,1], d->d[:qr_a_Q][3,1], d->d[:a_Q_true][3,1]),
    # Transition 75th
    ("q_{75,0}",   d->d[:ml_a_Q][1,3], d->d[:qr_a_Q][1,3], d->d[:a_Q_true][1,3]),
    ("q_{75,1}",   d->d[:ml_a_Q][2,3], d->d[:qr_a_Q][2,3], d->d[:a_Q_true][2,3]),
    ("q_{75,2}",   d->d[:ml_a_Q][3,3], d->d[:qr_a_Q][3,3], d->d[:a_Q_true][3,3]),
    # Transition tail rates (MLE only — QR uses IQR-based tails)
    ("b_{1,Q}",    d->d[:ml_b1_Q],     d->NaN, d->d[:b1_Q_true]),
    ("b_{L,Q}",    d->d[:ml_bL_Q],     d->NaN, d->d[:bL_Q_true]),
    # Initial distribution knots
    ("a_{init,1}", d->d[:ml_a_init][1], d->d[:qr_a_init][1], d->d[:a_init_true][1]),
    ("a_{init,2}", d->d[:ml_a_init][2], d->d[:qr_a_init][2], d->d[:a_init_true][2]),
    ("a_{init,3}", d->d[:ml_a_init][3], d->d[:qr_a_init][3], d->d[:a_init_true][3]),
    # Initial tail rates
    ("b_{1,init}", d->d[:ml_b1_init],  d->NaN, d->d[:b1_init_true]),
    ("b_{L,init}", d->d[:ml_bL_init],  d->NaN, d->d[:bL_init_true]),
    # Epsilon
    ("a_{\\varepsilon,1}", d->d[:ml_a_eps][1], d->d[:qr_a_eps1], d->d[:a_eps1_true]),
    ("a_{\\varepsilon,3}", d->d[:ml_a_eps][3], d->d[:qr_a_eps3], d->d[:a_eps3_true]),
    # Epsilon tail rates
    ("b_{1,\\varepsilon}", d->d[:ml_b1_eps], d->NaN, d->d[:b1_eps_true]),
    ("b_{L,\\varepsilon}", d->d[:ml_bL_eps], d->NaN, d->d[:bL_eps_true]),
]

function stats(data, fn, truth; drop_outliers=true)
    vals = Float64[]
    for d in data
        v = fn(d)
        (isnan(v) || isinf(v)) && continue
        drop_outliers && abs(v) > 1e3 && continue
        push!(vals, v)
    end
    isempty(vals) && return nothing
    b = mean(vals) - truth
    s = std(vals)
    rmse = sqrt(b^2 + s^2)
    cp05 = mean(abs.(vals .- truth) .< 0.05)
    cp10 = mean(abs.(vals .- truth) .< 0.10)
    mad  = median(abs.(vals .- truth))
    (b=b, s=s, rmse=rmse, cp05=cp05, cp10=cp10, mad=mad, n=length(vals))
end

# ---- Produce LaTeX tables (per (rho, T) configuration) ----
open(out_tex, "w") do io
    for key in sort(collect(keys(results)))
        ρ, N, T = key
        data = results[key]
        println(io, raw"\begin{table}[H]")
        println(io, raw"\centering")
        println(io, raw"\footnotesize")
        @printf(io, "\\caption{ABB piecewise-uniform model: bias / std / RMSE by method and parameter (\$\\rho=%.2f\$, \$N=%d\$, \$T=%d\$, %d replications).}\n",
                ρ, N, T, length(data))
        @printf(io, "\\label{tab:abbpw_rho%d_N%d_T%d}\n", round(Int,100ρ), N, T)
        println(io, raw"\begin{tabular}{l|cc|cc|cc}")
        println(io, raw"\toprule")
        println(io, raw" & \multicolumn{2}{c|}{Bias} & \multicolumn{2}{c|}{Std} & \multicolumn{2}{c}{RMSE} \\")
        println(io, raw"Param & MLE & QR & MLE & QR & MLE & QR \\")
        println(io, raw"\midrule")
        for (name, fn_ml, fn_qr, fn_true) in params
            t = fn_true(data[1])
            sm = stats(data, fn_ml, t)
            sq = stats(data, fn_qr, t)
            row = "\$" * name * "\$"
            for x in (sm, sq)
                row *= x === nothing ? " & --" : @sprintf(" & %+.3f", x.b)
            end
            for x in (sm, sq)
                row *= x === nothing ? " & --" : @sprintf(" & %.3f", x.s)
            end
            for x in (sm, sq)
                row *= x === nothing ? " & --" : @sprintf(" & %.3f", x.rmse)
            end
            row *= " \\\\"
            println(io, row)
        end
        println(io, raw"\bottomrule")
        println(io, raw"\end{tabular}")
        println(io, raw"\end{table}")
        println(io)
    end
end

@printf("\nWrote LaTeX tables to %s\n", out_tex)

# ---- Dump summary stats to stdout ----
for key in sort(collect(keys(results)))
    ρ, N, T = key
    data = results[key]
    @printf("\nρ=%.2f N=%d T=%d (n=%d):\n", ρ, N, T, length(data))
    @printf("  %-22s  %-28s  %-28s\n", "param", "MLE (bias,std,RMSE)", "QR (bias,std,RMSE)")
    for (name, fn_ml, fn_qr, fn_true) in params
        t = fn_true(data[1])
        sm = stats(data, fn_ml, t)
        sq = stats(data, fn_qr, t)
        fmt = x -> x === nothing ? "  --" : @sprintf("(%+.3f, %.3f, %.3f)", x.b, x.s, x.rmse)
        @printf("  %-22s  %-28s  %-28s\n", name, fmt(sm), fmt(sq))
    end
end

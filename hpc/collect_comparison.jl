#=
collect_comparison.jl — Collect and summarize Grid MLE vs SML vs QR results.

Usage: julia collect_comparison.jl [results_dir]
=#

using Serialization, Printf, Statistics

results_dir = length(ARGS) >= 1 ? ARGS[1] : "results"

files = filter(f -> startswith(f, "comparison_") && endswith(f, ".jls"), readdir(results_dir))
isempty(files) && error("No comparison results found in $results_dir")

# Group by N
results = Dict{Int, Vector{Dict}}()
for f in files
    d = deserialize(joinpath(results_dir, f))
    N = d[:N]
    haskey(results, N) || (results[N] = Dict[])
    push!(results[N], d)
end

for N in sort(collect(keys(results)))
    data = results[N]
    S = length(data)
    println("="^80)
    @printf("N=%d, S=%d simulations\n", N, S)
    println("="^80)

    # Extract estimates
    ml_rho = [d[:ml_a_Q][2,2] for d in data]
    sml_rho = [d[:sml_a_Q][2,2] for d in data]
    qr_rho = [d[:qr_a_Q][2,2] for d in data]

    ml_ae3 = [d[:ml_a_eps3] for d in data]
    sml_ae3 = [d[:sml_a_eps3] for d in data]
    qr_ae3 = [d[:qr_a_eps3] for d in data]

    ml_Me = [d[:ml_M_eps] for d in data]
    sml_Me = [d[:sml_M_eps] for d in data]
    qr_Me = [d[:qr_M_eps] for d in data]

    ml_MQ = [d[:ml_M_Q] for d in data]
    sml_MQ = [d[:sml_M_Q] for d in data]
    qr_MQ = [d[:qr_M_Q] for d in data]

    truth = data[1]

    @printf("%-12s %8s  %8s %8s %8s  %8s %8s %8s  %8s %8s %8s\n",
            "Parameter", "True", "ML_bias", "ML_std", "ML_RMSE",
            "SML_bias", "SML_std", "SML_RMSE", "QR_bias", "QR_std", "QR_RMSE")
    println("-"^110)

    for (name, ml, sml, qr, true_val) in [
        ("ρ", ml_rho, sml_rho, qr_rho, truth[:a_Q_true][2,2]),
        ("ae3", ml_ae3, sml_ae3, qr_ae3, truth[:a_eps3_true]),
        ("M_Q", ml_MQ, sml_MQ, qr_MQ, truth[:M_Q_true]),
        ("M_eps", ml_Me, sml_Me, qr_Me, truth[:M_eps_true])]

        for (method, vals) in [("ML", ml), ("SML", sml), ("QR", qr)]
            bias = mean(vals) - true_val
            s = std(vals)
            rmse = sqrt(bias^2 + s^2)
        end
        mb=mean(ml)-true_val; ms=std(ml); mr=sqrt(mb^2+ms^2)
        sb=mean(sml)-true_val; ss=std(sml); sr=sqrt(sb^2+ss^2)
        qb=mean(qr)-true_val; qs=std(qr); qrr=sqrt(qb^2+qs^2)
        @printf("%-12s %8.4f  %+8.4f %8.4f %8.4f  %+8.4f %8.4f %8.4f  %+8.4f %8.4f %8.4f\n",
                name, true_val, mb, ms, mr, sb, ss, sr, qb, qs, qrr)
    end

    @printf("\nTiming: ML=%.0fs  SML=%.0fs  QR=%.0fs (mean)\n",
            mean(d[:ml_time] for d in data),
            mean(d[:sml_time] for d in data),
            mean(d[:qr_time] for d in data))
end

#=
collect_profiled.jl — Collect and analyse profiled MLE vs QR results.

Usage: julia collect_profiled.jl [results_dir]

Reports: bias, std, MSE, coverage probability, QR/ML RMSE ratio,
         NaN counts, Louis missing information.
=#

using Serialization, Printf, Statistics, LinearAlgebra

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

param_names = ["med_q0","med_q1","med_q2","δL1","δL2","δL3","δR1","δR2","δR3",
                "init_m","log_gL","log_gR","l_ae1","l_ae3"]

function extract_profiled_params(d, method::Symbol)
    if method == :ml
        aQ = d[:ml_a_Q]; ai = d[:ml_a_init]
        ae1 = d[:ml_a_eps1]; ae3 = d[:ml_a_eps3]
    else
        aQ = d[:qr_a_Q]; ai = d[:qr_a_init]
        ae1 = d[:qr_a_eps1]; ae3 = d[:qr_a_eps3]
    end
    # Pack into same order as pack_profiled
    K = size(aQ,1)-1; nk = K+1
    v = zeros(14)
    v[1:3] .= aQ[:,2]  # median
    # Gap params — need delta_from_gap
    # For simplicity, just store the key params directly
    [aQ[2,2], ae1, ae3, ai[1], ai[2], ai[3]]  # ρ, ae1, ae3, init1, init2, init3
end

for key in sort(collect(keys(results)))
    ρ_val, N, T_obs = key
    data = results[key]
    S = length(data)

    println("=" ^ 80)
    @printf("ρ=%.2f, N=%d, T=%d, S=%d replications\n", ρ_val, N, T_obs, S)
    println("=" ^ 80)

    # Extract key parameters
    true_params = Dict(
        "ρ" => data[1][:a_Q_true][2,2],
        "ae3" => data[1][:a_eps3_true],
        "ae1" => data[1][:a_eps1_true],
        "init1" => data[1][:a_init_true][1],
        "init2" => data[1][:a_init_true][2],
        "init3" => data[1][:a_init_true][3],
    )

    ml_params = Dict{String, Vector{Float64}}()
    qr_params = Dict{String, Vector{Float64}}()
    for name in keys(true_params)
        ml_params[name] = Float64[]
        qr_params[name] = Float64[]
    end

    n_nan = 0
    for d in data
        # Check for NaN
        if isnan(d[:ml_nll])
            n_nan += 1
            continue
        end
        push!(ml_params["ρ"], d[:ml_a_Q][2,2])
        push!(ml_params["ae3"], d[:ml_a_eps3])
        push!(ml_params["ae1"], d[:ml_a_eps1])
        push!(ml_params["init1"], d[:ml_a_init][1])
        push!(ml_params["init2"], d[:ml_a_init][2])
        push!(ml_params["init3"], d[:ml_a_init][3])

        push!(qr_params["ρ"], d[:qr_a_Q][2,2])
        push!(qr_params["ae3"], d[:qr_a_eps3])
        push!(qr_params["ae1"], d[:qr_a_eps1])
        push!(qr_params["init1"], d[:qr_a_init][1])
        push!(qr_params["init2"], d[:qr_a_init][2])
        push!(qr_params["init3"], d[:qr_a_init][3])
    end

    n_valid = length(ml_params["ρ"])
    @printf("  Valid: %d/%d (NaN: %d)\n\n", n_valid, S, n_nan)

    # Coverage probability: fraction within δ of truth
    deltas = [0.05, 0.10, 0.20]

    @printf("%-8s %8s  %8s %8s %8s  %8s %8s %8s  %5s",
            "Param", "True", "ML_bias", "ML_std", "ML_RMSE", "QR_bias", "QR_std", "QR_RMSE", "QR/ML")
    for δ in deltas; @printf("  CP_ML(%.2f) CP_QR(%.2f)", δ, δ); end
    println()
    println("-" ^ (80 + 20*length(deltas)))

    for name in ["ρ", "ae3", "init1", "init2", "init3"]
        truth = true_params[name]
        ml = ml_params[name]; qr = qr_params[name]
        isempty(ml) && continue

        mb = mean(ml) - truth; ms = std(ml); mr = sqrt(mb^2 + ms^2)
        qb = mean(qr) - truth; qs = std(qr); qrr = sqrt(qb^2 + qs^2)
        eff = mr > 0 ? qrr / mr : 0.0

        @printf("%-8s %8.4f  %+8.4f %8.4f %8.4f  %+8.4f %8.4f %8.4f  %5.2f",
                name, truth, mb, ms, mr, qb, qs, qrr, eff)

        # Coverage probabilities
        for δ in deltas
            cp_ml = mean(abs.(ml .- truth) .< δ)
            cp_qr = mean(abs.(qr .- truth) .< δ)
            @printf("  %8.3f %8.3f", cp_ml, cp_qr)
        end
        println()
    end

    # Louis missing information (from seed=1 if available)
    louis_data = filter(d -> d[:louis_Var_S] !== nothing, data)
    if !isempty(louis_data)
        println("\n  Louis missing information (seed=1):")
        Var_S = louis_data[1][:louis_Var_S]
        for j in 1:min(size(Var_S,1), length(param_names))
            @printf("    %-10s: Var=%.4f\n", param_names[j], Var_S[j,j])
        end
    end

    # Timing
    ml_times = [d[:ml_time] for d in data if !isnan(d[:ml_nll])]
    qr_times = [d[:qr_time] for d in data]
    @printf("\n  Timing: MLE mean=%.1fs  QR mean=%.1fs\n",
            mean(ml_times), mean(qr_times))
    println()
end

#=
collect_cspline.jl — Collect and summarize cubic spline MC results (MLE vs QR).

Usage: julia collect_cspline.jl [results_dir]
=#

using Serialization, Printf, Statistics

results_dir = length(ARGS) >= 1 ? ARGS[1] : "results"

files = filter(f -> startswith(f, "cspline_") && endswith(f, ".jls"),
               readdir(results_dir))

if isempty(files)
    println("No result files found in $results_dir"); exit(1)
end

results_by_N = Dict{Int, Vector{Dict{Symbol,Any}}}()
for f in files
    r = deserialize(joinpath(results_dir, f))
    N = r[:N]
    haskey(results_by_N, N) || (results_by_N[N] = Dict{Symbol,Any}[])
    push!(results_by_N[N], r)
end

K = 2

for N in sort(collect(keys(results_by_N)))
    rs = results_by_N[N]
    S = length(rs)

    # Extract arrays
    ml_aQ = zeros(S,K+1,3); qr_aQ = zeros(S,K+1,3)
    ml_βLQ = zeros(S); ml_βRQ = zeros(S); qr_βLQ = zeros(S); qr_βRQ = zeros(S)
    ml_βLi = zeros(S); ml_βRi = zeros(S); qr_βLi = zeros(S); qr_βRi = zeros(S)
    ml_βLe = zeros(S); ml_βRe = zeros(S); qr_βLe = zeros(S); qr_βRe = zeros(S)
    ml_ainit = zeros(S,3); qr_ainit = zeros(S,3)
    ml_ae1 = zeros(S); ml_ae3 = zeros(S)
    qr_ae1 = zeros(S); qr_ae3 = zeros(S)
    ml_t = zeros(S); qr_t = zeros(S)

    for (i, r) in enumerate(rs)
        ml_aQ[i,:,:] = r[:ml_a_Q]; qr_aQ[i,:,:] = r[:qr_a_Q]
        ml_βLQ[i] = r[:ml_beta_L_Q]; ml_βRQ[i] = r[:ml_beta_R_Q]
        qr_βLQ[i] = r[:qr_beta_L_Q]; qr_βRQ[i] = r[:qr_beta_R_Q]
        ml_βLi[i] = r[:ml_beta_L_init]; ml_βRi[i] = r[:ml_beta_R_init]
        qr_βLi[i] = r[:qr_beta_L_init]; qr_βRi[i] = r[:qr_beta_R_init]
        ml_βLe[i] = r[:ml_beta_L_eps]; ml_βRe[i] = r[:ml_beta_R_eps]
        qr_βLe[i] = r[:qr_beta_L_eps]; qr_βRe[i] = r[:qr_beta_R_eps]
        ml_ainit[i,:] = r[:ml_a_init]; qr_ainit[i,:] = r[:qr_a_init]
        ml_ae1[i] = r[:ml_a_eps1]; ml_ae3[i] = r[:ml_a_eps3]
        qr_ae1[i] = r[:qr_a_eps1]; qr_ae3[i] = r[:qr_a_eps3]
        ml_t[i] = r[:ml_time]; qr_t[i] = r[:qr_time]
    end

    # True values
    tp = rs[1]
    tp_aQ = tp[:a_Q_true]
    tp_βLQ = tp[:beta_L_Q_true]; tp_βRQ = tp[:beta_R_Q_true]
    tp_βLi = tp[:beta_L_init_true]; tp_βRi = tp[:beta_R_init_true]
    tp_βLe = tp[:beta_L_eps_true]; tp_βRe = tp[:beta_R_eps_true]
    tp_ainit = tp[:a_init_true]
    tp_ae1 = tp[:a_eps1_true]; tp_ae3 = tp[:a_eps3_true]

    println("="^85)
    @printf("N = %d  (S = %d replications)\n", N, S)
    println("="^85)
    @printf("%-12s %6s | %8s %8s %8s | %8s %8s %8s\n",
            "Parameter", "True", "ML Bias", "ML Std", "ML RMSE", "QR Bias", "QR Std", "QR RMSE")
    println("-"^85)

    function report(nm, tr, ml_est, qr_est)
        ml_b = mean(ml_est)-tr; ml_s = std(ml_est); ml_r = sqrt(mean((ml_est.-tr).^2))
        qr_b = mean(qr_est)-tr; qr_s = std(qr_est); qr_r = sqrt(mean((qr_est.-tr).^2))
        @printf("%-12s %6.3f | %+8.4f %8.4f %8.4f | %+8.4f %8.4f %8.4f\n",
                nm, tr, ml_b, ml_s, ml_r, qr_b, qr_s, qr_r)
    end

    report("β_L_Q", tp_βLQ, ml_βLQ, qr_βLQ)
    report("β_R_Q", tp_βRQ, ml_βRQ, qr_βRQ)
    report("β_L_init", tp_βLi, ml_βLi, qr_βLi)
    report("β_R_init", tp_βRi, ml_βRi, qr_βRi)
    report("β_L_eps", tp_βLe, ml_βLe, qr_βLe)
    report("β_R_eps", tp_βRe, ml_βRe, qr_βRe)
    for l in 1:3, k in 1:K+1
        report("a_Q[$k,$l]", tp_aQ[k,l], ml_aQ[:,k,l], qr_aQ[:,k,l])
    end
    report("a_init[1]", tp_ainit[1], ml_ainit[:,1], qr_ainit[:,1])
    report("a_init[2]", tp_ainit[2], ml_ainit[:,2], qr_ainit[:,2])
    report("a_init[3]", tp_ainit[3], ml_ainit[:,3], qr_ainit[:,3])
    report("a_eps1", tp_ae1, ml_ae1, qr_ae1)
    report("a_eps3", tp_ae3, ml_ae3, qr_ae3)

    println("-"^85)
    @printf("Avg time:  ML=%.1fs  QR=%.1fs\n", mean(ml_t), mean(qr_t))

    # Paired RMSE ratio for key parameters
    println("\nPaired RMSE ratio (QR/ML):")
    for l in 1:3
        for k in 1:K+1
            ml_r = sqrt(mean((ml_aQ[:,k,l] .- tp_aQ[k,l]).^2))
            qr_r = sqrt(mean((qr_aQ[:,k,l] .- tp_aQ[k,l]).^2))
            ml_r > 1e-10 && @printf("  a_Q[%d,%d]: %.2f\n", k, l, qr_r/ml_r)
        end
    end
    println()
end

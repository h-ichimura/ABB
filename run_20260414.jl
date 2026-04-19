using Serialization
include("ABB_three_period.jl")
for N in [200, 500, 1000]
    println("\n>>> N=$N <<<\n")
    res = run_comparison(N=N, K=2, L=3, maxiter=100, n_draws=200, M=50,
                         nonlinear=false, seed=42, methods=[:qr, :mle])
    serialize("results_20260414_N$(N).jls", res)
    println("  Saved results_20260414_N$(N).jls")
end

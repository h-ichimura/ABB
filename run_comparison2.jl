include("ABB_three_period.jl")

for N in [200, 500, 1000]
    println("\n>>> LINEAR DGP, N=$N <<<\n")
    run_comparison(N=N, K=2, L=3, maxiter=100, n_draws=200, M=50,
                   nonlinear=false, seed=42, methods=[:qr, :ols, :mle])
end

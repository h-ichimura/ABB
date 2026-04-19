include("ABB_three_period.jl")

open("results_3methods.txt", "w") do io
    for N in [200, 500, 1000]
        msg = "\n>>> LINEAR DGP, N=$N <<<\n"
        print(msg); print(io, msg)
        old_stdout = stdout
        rd, wr = redirect_stdout()
        # QR and OLS are fast; MLE is slow due to NM.
        # Run QR and OLS for all N; skip MLE for N=1000.
        meths = N <= 500 ? [:qr, :ols, :mle] : [:qr, :ols]
        run_comparison(N=N, K=2, L=3, maxiter=100, n_draws=200, M=50,
                       nonlinear=false, seed=42, methods=meths)
        redirect_stdout(old_stdout)
        close(wr)
        output = read(rd, String)
        print(output); print(io, output); flush(io)
    end
end
println("\nResults saved to results_3methods.txt")

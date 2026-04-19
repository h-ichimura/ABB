#!/bin/bash
# Run this ONCE on the HPC login node to install Julia packages including CUDA
# Usage: bash setup_julia_gpu.sh

export PATH="$HOME/julia-1.11.5/bin:$PATH"

julia -e '
using Pkg
Pkg.add("Optim")
Pkg.add("LineSearches")
Pkg.add("Distributions")
Pkg.add("CUDA")
Pkg.precompile()
println("Setup complete. Packages installed:")
Pkg.status()
'

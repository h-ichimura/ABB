#!/bin/bash
# Run this ONCE on the HPC login node to install Julia packages
# Usage: bash setup_julia.sh

module load julia  # adjust if needed: module avail julia to see versions

julia -e '
using Pkg
Pkg.add("Optim")
Pkg.add("LineSearches")
Pkg.precompile()
println("Setup complete. Packages installed:")
Pkg.status()
'

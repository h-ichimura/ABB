#!/bin/bash
# Sync code to HPC. Run from ~/Dropbox/hidehiko/Research/ABB/

rsync -avz \
    --include='ABB_three_period.jl' \
    --include='cspline_abb.jl' \
    --include='hpc/' \
    --include='hpc/run_hpc_cspline.jl' \
    --include='hpc/run_hpc_gpu.jl' \
    --include='hpc/submit_cspline.sh' \
    --include='hpc/submit_gpu.sh' \
    --include='hpc/submit_cpu_analytical.sh' \
    --include='hpc/submit_comparison.sh' \
    --include='hpc/run_hpc_comparison.jl' \
    --include='hpc/collect_comparison.jl' \
    --include='hpc/collect_cspline.jl' \
    --include='hpc/setup_julia.sh' \
    --include='hpc/setup_julia_gpu.sh' \
    --exclude='*' \
    ~/Dropbox/hidehiko/Research/ABB/ \
    ichimura@hpc.arizona.edu:~/ABB/

echo "Synced to HPC"

#!/bin/bash
# Download Monte Carlo results from HPC. Run from ~/Dropbox/hidehiko/Research/ABB/

mkdir -p hpc/mc_results

rsync -avz \
    'ichimura@filexfer.hpc.arizona.edu:~/ABB/hpc/summary_*.jls' \
    hpc/mc_results/

echo "Downloaded MC results to hpc/mc_results/"
ls hpc/mc_results/ | wc -l
echo "files"

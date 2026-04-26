#!/bin/bash
# Per-task batch script for the GPU pipeline.
# Submitted by submit_design_gpu.sh as a slurm array job; each task reads
# a single (rho, N, T, seed) line from a manifest file passed as $1.
#
# SLURM directives that are constant across all GPU tasks live here.
# Variable directives (--array, --partition, --account) come from sbatch CLI
# in submit_design_gpu.sh.
#
# Output filename is identical to the CPU pipeline so result files mix
# safely (existence check at the top short-circuits redundant work).

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --gres=gpu:1
#SBATCH --output=logs/abbpw_gpu_%A_%a.out
#SBATCH --error=logs/abbpw_gpu_%A_%a.err

module load cuda

MANIFEST="$1"
cd ~/ABB/hpc
LINE=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$MANIFEST")
read RHO N T SEED <<< "$LINE"
[ -z "$SEED" ] && { echo "empty manifest line"; exit 1; }

OUTFILE="results/abbpw_r${RHO}_N${N}_T${T}_seed${SEED}.jls"
[ -f "$OUTFILE" ] && { echo "already done: $OUTFILE"; exit 0; }

julia run_abb_pw_gpu.jl "$N" "$T" "$SEED" "$RHO"

SRC="abbpw_r${RHO}_N${N}_T${T}_seed${SEED}.jls"
[ -f "$SRC" ] && mv "$SRC" "$OUTFILE"

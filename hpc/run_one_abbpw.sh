#!/bin/bash
#SBATCH --account=ichimura
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=5G
#SBATCH --array=1-1000

export PATH="$HOME/julia-1.11.5/bin:$PATH"

SEED=$SLURM_ARRAY_TASK_ID
N=$1
T=$2
RHO=$3

cd "$HOME/ABB/hpc"
julia run_abb_pw_comparison.jl $N $T $SEED $RHO
mv abbpw_r${RHO}_N${N}_T${T}_seed${SEED}.jls results/ 2>/dev/null

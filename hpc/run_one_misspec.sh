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
RHO1=$3
RHO2=$4
SIGMA_V=$5

cd "$HOME/ABB/hpc"
julia run_misspec_comparison.jl $N $T $SEED $RHO1 $RHO2 $SIGMA_V
mv misspec_r1${RHO1}_r2${RHO2}_N${N}_T${T}_seed${SEED}.jls results/ 2>/dev/null

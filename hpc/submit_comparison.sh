#!/bin/bash
# Submit Grid MLE vs SML vs QR comparison jobs on UA HPC Puma.
# Data generated with rejection sampling (exact model draws).

SAMPLE_SIZES="500 1000"
G_VALUES="201 501 1001"
SEEDS=$(seq 1 200)
R_SML=500

WORKDIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && cd .. && pwd )"

mkdir -p logs results

COUNT=0
for G in $G_VALUES; do
    for N in $SAMPLE_SIZES; do
        for SEED in $SEEDS; do
            JOBNAME="cmp_G${G}_N${N}_s${SEED}"
            sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=$JOBNAME
#SBATCH --account=ichimura
#SBATCH --partition=gpu_standard
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --output=logs/${JOBNAME}_%j.out
#SBATCH --error=logs/${JOBNAME}_%j.err

module load cuda
export PATH="\$HOME/julia-1.11.5/bin:\$PATH"

cd ${WORKDIR}/hpc
julia run_hpc_comparison.jl $N $SEED $G $R_SML

mv comparison_N${N}_seed${SEED}.jls results/comparison_G${G}_N${N}_seed${SEED}.jls 2>/dev/null
EOF
            COUNT=$((COUNT + 1))
        done
    done
done

echo "Submitted $COUNT comparison jobs (G=$G_VALUES × N=$SAMPLE_SIZES × 200 seeds)"

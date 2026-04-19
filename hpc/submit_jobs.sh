#!/bin/bash
# Submit Monte Carlo jobs to SLURM on UA HPC Puma.
# 200 independent datasets per sample size N.
# Methods: QR and Exact ML.

SAMPLE_SIZES="200 500 1000 2000"
SEEDS=$(seq 1 200)

WORKDIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && cd .. && pwd )"

mkdir -p logs

COUNT=0
for N in $SAMPLE_SIZES; do
    for SEED in $SEEDS; do
        JOBNAME="abb_N${N}_s${SEED}"
        sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=$JOBNAME
#SBATCH --account=ichimura
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:30:00
#SBATCH --output=logs/${JOBNAME}_%j.out
#SBATCH --error=logs/${JOBNAME}_%j.err

export PATH="\$HOME/julia-1.11.5/bin:\$PATH"

cd ${WORKDIR}/hpc
julia run_hpc.jl $N $SEED
EOF
        COUNT=$((COUNT + 1))
    done
done

echo "Submitted $COUNT jobs ($((COUNT/200)) N x 200 seeds)"

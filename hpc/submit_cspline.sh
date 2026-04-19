#!/bin/bash
# Submit cubic spline MLE Monte Carlo jobs to SLURM on UA HPC Puma.
# 200 independent datasets per sample size N.

SAMPLE_SIZES="500 1000"
SEEDS=$(seq 1 200)

WORKDIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && cd .. && pwd )"

mkdir -p logs results

COUNT=0
for N in $SAMPLE_SIZES; do
    for SEED in $SEEDS; do
        JOBNAME="cs_N${N}_s${SEED}"
        sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=$JOBNAME
#SBATCH --account=ichimura
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --output=logs/${JOBNAME}_%j.out
#SBATCH --error=logs/${JOBNAME}_%j.err

export PATH="\$HOME/julia-1.11.5/bin:\$PATH"

cd ${WORKDIR}/hpc
julia run_hpc_cspline.jl $N $SEED

# Move result to results dir
mv cspline_N${N}_seed${SEED}.jls results/ 2>/dev/null
EOF
        COUNT=$((COUNT + 1))
    done
done

echo "Submitted $COUNT jobs ($((COUNT/200)) sample sizes x 200 seeds)"

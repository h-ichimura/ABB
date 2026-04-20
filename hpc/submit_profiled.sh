#!/bin/bash
# Submit profiled MLE vs QR comparison jobs on UA HPC Puma.
# 6000 jobs: N={300,1200,4800} × T={3,6} × seeds 1-1000

SAMPLE_SIZES="300 1200 4800"
TIME_PERIODS="3 6"
RHOS="0.5 0.8 0.95"
SEEDS=$(seq 1 1000)

WORKDIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && cd .. && pwd )"

mkdir -p logs results

COUNT=0
for RHO in $RHOS; do
    for N in $SAMPLE_SIZES; do
        for T in $TIME_PERIODS; do
            for SEED in $SEEDS; do
                JOBNAME="pf_r${RHO}_N${N}_T${T}_s${SEED}"
                sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=$JOBNAME
#SBATCH --account=ichimura
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --output=logs/${JOBNAME}_%j.out
#SBATCH --error=logs/${JOBNAME}_%j.err

export PATH="\$HOME/julia-1.11.5/bin:\$PATH"

cd ${WORKDIR}/hpc
julia run_profiled_comparison.jl $N $T $SEED $RHO

mv profiled_r${RHO}_N${N}_T${T}_seed${SEED}.jls results/ 2>/dev/null
EOF
                COUNT=$((COUNT + 1))
            done
        done
    done
done

echo "Submitted $COUNT jobs (rho={$RHOS} x N={$SAMPLE_SIZES} x T={$TIME_PERIODS} x 1000 seeds)"

#!/bin/bash
# Submit Grid MLE vs SML vs QR comparison jobs on UA HPC Puma.
# Data generated with rejection sampling (exact model draws).

SAMPLE_SIZES="500 1000"
SEEDS=$(seq 1 200)
G=201
R_SML=500

WORKDIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && cd .. && pwd )"

mkdir -p logs results

COUNT=0
for N in $SAMPLE_SIZES; do
    for SEED in $SEEDS; do
        JOBNAME="cmp_N${N}_s${SEED}"
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
julia run_hpc_comparison.jl $N $SEED $G $R_SML

mv comparison_N${N}_seed${SEED}.jls results/ 2>/dev/null
EOF
        COUNT=$((COUNT + 1))
    done
done

echo "Submitted $COUNT comparison jobs"

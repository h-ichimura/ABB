#!/bin/bash
# Submit profiled MLE vs QR comparison using SLURM job arrays.
# Each array job handles one (rho, N, T) configuration with 1000 seeds.
# 18 configurations × 1 array each = 18 submissions (not 18000).

WORKDIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && cd .. && pwd )"
mkdir -p logs results

# 18 configurations: 3 rho × 3 N × 2 T
CONFIGS=(
    "0.5 300 3"
    "0.5 300 6"
    "0.5 1200 3"
    "0.5 1200 6"
    "0.5 4800 3"
    "0.5 4800 6"
    "0.8 300 3"
    "0.8 300 6"
    "0.8 1200 3"
    "0.8 1200 6"
    "0.8 4800 3"
    "0.8 4800 6"
    "0.95 300 3"
    "0.95 300 6"
    "0.95 1200 3"
    "0.95 1200 6"
    "0.95 4800 3"
    "0.95 4800 6"
)

COUNT=0
for CONFIG in "${CONFIGS[@]}"; do
    read RHO N T <<< "$CONFIG"
    JOBNAME="pf_r${RHO}_N${N}_T${T}"

    sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=${JOBNAME}
#SBATCH --account=ichimura
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --array=1-1000
#SBATCH --output=logs/${JOBNAME}_%a_%j.out
#SBATCH --error=logs/${JOBNAME}_%a_%j.err

export PATH="\$HOME/julia-1.11.5/bin:\$PATH"

SEED=\$SLURM_ARRAY_TASK_ID

cd ${WORKDIR}/hpc
julia run_profiled_comparison.jl $N $T \$SEED $RHO

mv profiled_r${RHO}_N${N}_T${T}_seed\${SEED}.jls results/ 2>/dev/null
EOF
    COUNT=$((COUNT + 1))
    echo "Submitted array job $COUNT/18: rho=$RHO N=$N T=$T (1000 tasks)"
done

echo ""
echo "Submitted $COUNT array jobs (18 × 1000 = 18,000 total tasks)"

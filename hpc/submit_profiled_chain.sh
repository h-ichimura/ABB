#!/bin/bash
# Submit profiled MLE vs QR as chained job arrays.
# Each array starts after the previous one finishes.
# This respects the QOS job submission limit.

WORKDIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && cd .. && pwd )"
mkdir -p logs results

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

PREV_JOB=""
COUNT=0

for CONFIG in "${CONFIGS[@]}"; do
    read RHO N T <<< "$CONFIG"
    JOBNAME="pf_r${RHO}_N${N}_T${T}"
    COUNT=$((COUNT + 1))

    if [ -z "$PREV_JOB" ]; then
        # First job: no dependency
        JOBID=$(sbatch --parsable <<EOF
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
        )
    else
        # Subsequent jobs: depend on previous completing
        JOBID=$(sbatch --parsable --dependency=afterany:${PREV_JOB} <<EOF
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
        )
    fi

    PREV_JOB=$JOBID
    echo "[$COUNT/18] Submitted $JOBNAME (job $JOBID, 1000 tasks)"
done

echo ""
echo "All 18 array jobs submitted as dependency chain."
echo "Total: 18,000 tasks. They will run sequentially, 1000 at a time."

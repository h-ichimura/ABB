#!/bin/bash
# Submit profiled MLE vs QR as chained job arrays.
# Uses a script file (not heredoc) to avoid Puma sbatch wrapper issues.

cd "$( dirname "${BASH_SOURCE[0]}" )"
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
        JOBID=$(sbatch --parsable \
            --job-name=${JOBNAME} \
            --output=logs/${JOBNAME}_%a_%j.out \
            --error=logs/${JOBNAME}_%a_%j.err \
            run_one_profiled.sh $N $T $RHO)
    else
        JOBID=$(sbatch --parsable \
            --dependency=afterany:${PREV_JOB} \
            --job-name=${JOBNAME} \
            --output=logs/${JOBNAME}_%a_%j.out \
            --error=logs/${JOBNAME}_%a_%j.err \
            run_one_profiled.sh $N $T $RHO)
    fi

    PREV_JOB=$JOBID
    echo "[$COUNT/18] Submitted $JOBNAME (job $JOBID, 1000 tasks)"
done

echo ""
echo "All 18 array jobs submitted as dependency chain."
echo "Total: 18,000 tasks. They will run sequentially, 1000 at a time."

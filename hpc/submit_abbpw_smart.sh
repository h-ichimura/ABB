#!/bin/bash
# submit_abbpw_smart.sh — Submit ABB piecewise-uniform Monte Carlo configs
# one at a time, waiting for the queue to drain before each submission.
# Matches the working pattern from submit_smart.sh / submit_missing.sh.
#
# 18 configurations × 1000 seeds = 18,000 total tasks.
# QOS limit forces sequential submission (one array runs at a time).

cd ~/ABB/hpc
mkdir -p logs results

submit_after_drain() {
    JOBNAME=$1; shift
    # Wait FIRST (avoid QOSMaxSubmitJobPerUserLimit) then submit.
    while true; do
        NJOBS=$(squeue -u ichimura -h | wc -l)
        [ "$NJOBS" -eq 0 ] && break
        echo "  Queue has $NJOBS jobs, waiting..."
        sleep 60
    done
    echo "Submitting $JOBNAME..."
    sbatch --job-name=${JOBNAME} "$@"
    sleep 10
}

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

for CONFIG in "${CONFIGS[@]}"; do
    read RHO N T <<< "$CONFIG"
    JOBNAME="abbpw_r${RHO}_N${N}_T${T}"

    DONE=$(ls results/abbpw_r${RHO}_N${N}_T${T}_seed*.jls 2>/dev/null | wc -l)
    if [ "$DONE" -ge 1000 ]; then
        echo "Skipping $JOBNAME (already 1000 results)"
        continue
    fi

    submit_after_drain $JOBNAME \
        --array=1-1000 \
        --output=logs/${JOBNAME}_%a_%j.out \
        --error=logs/${JOBNAME}_%a_%j.err \
        run_one_abbpw.sh $N $T $RHO

    # Wait for THIS array to finish before moving on
    while true; do
        NJOBS=$(squeue -u ichimura -h | wc -l)
        [ "$NJOBS" -eq 0 ] && break
        echo "  $JOBNAME running... $NJOBS jobs remaining"
        sleep 60
    done
    echo "  $JOBNAME complete."
done

echo "ALL ABB-PW CONFIGS DONE."

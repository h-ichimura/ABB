#!/bin/bash
# Submit misspecified AR(2) comparison: MLE vs C²-QR vs ABB-QR
# 2 DGP × 3 N × T=6 = 6 arrays, 6000 total jobs

cd "$( dirname "${BASH_SOURCE[0]}" )"
mkdir -p logs results

# DGP configs: "rho1 rho2 sigma_v N T"
CONFIGS=(
    "1.0 -0.3 0.5 300 6"
    "1.0 -0.3 0.5 1200 6"
    "1.0 -0.3 0.5 4800 6"
    "1.3 -0.5 0.5 300 6"
    "1.3 -0.5 0.5 1200 6"
    "1.3 -0.5 0.5 4800 6"
)

COUNT=0
for CONFIG in "${CONFIGS[@]}"; do
    read RHO1 RHO2 SIGMA_V N T <<< "$CONFIG"
    JOBNAME="ms_r1${RHO1}_r2${RHO2}_N${N}"
    COUNT=$((COUNT + 1))

    echo "[$COUNT/6] Submitting $JOBNAME..."
    sbatch --job-name=${JOBNAME} \
        --output=logs/${JOBNAME}_%a_%j.out \
        --error=logs/${JOBNAME}_%a_%j.err \
        run_one_misspec.sh $N $T $RHO1 $RHO2 $SIGMA_V

    # Wait until no jobs are running/pending
    while true; do
        NJOBS=$(squeue -u ichimura -h | wc -l)
        if [ "$NJOBS" -eq 0 ]; then
            break
        fi
        echo "  Waiting... $NJOBS jobs remaining"
        sleep 60
    done
    echo "  Done."
done

echo "All 6 configurations complete."

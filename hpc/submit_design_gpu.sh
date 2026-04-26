#!/bin/bash
# submit_design_gpu.sh — Submit missing N=4800 (rho, T, seed) jobs as a slurm
# array on the gpu_standard or gpu_windfall partition.
#
# Idempotent: rescans results/ each iteration so preempted/failed seeds
# come back automatically.  Mirror of submit_design.sh, but limited to the
# four N=4800 cells which is where GPU acceleration pays.
#
# Usage:
#   bash ~/ABB/hpc/submit_design_gpu.sh [SEED_LO] [SEED_HI] [PARTITION]
#
# Defaults:
#   SEED_LO=1   SEED_HI=1000   PARTITION=gpu_standard
#
# Run inside a tmux session (after sourcing slurm-selector.sh for the
# desired cluster):
#   tmux new -d -s gpu_sub "bash -c '. /usr/local/bin/slurm-selector.sh puma; \
#       bash ~/ABB/hpc/submit_design_gpu.sh 1 1000 gpu_standard \
#       2>&1 | tee -a ~/ABB/hpc/logs/gpu_sub.log'"

SEED_LO=${1:-1}
SEED_HI=${2:-1000}
PARTITION=${3:-gpu_standard}

cd ~/ABB/hpc
mkdir -p logs results manifests

# N=4800 cells only.  GPU win is concentrated here — small N doesn't
# amortise GPU launch overhead.
CONFIGS=(
    "0.8  4800 3"
    "0.95 4800 3"
    "0.8  4800 6"
    "0.95 4800 6"
)
SEEDS=$(seq $SEED_LO $SEED_HI)
BATCH_MAX=999

# --account rule: omit on windfall partitions per UA HPC docs.
SBATCH_BASE=(--parsable
             --partition=$PARTITION
             --gres=gpu:volta:1
             --job-name=abbpw_gpu)
case "$PARTITION" in
    windfall|gpu_windfall) ;;
    *) SBATCH_BASE+=(--account=ichimura) ;;
esac

echo "GPU submitter: seeds $SEED_LO..$SEED_HI partition=$PARTITION (N=4800 only)"

while true; do
    while true; do
        NQ=$(squeue -u $USER -h -p $PARTITION 2>/dev/null | wc -l)
        [ "$NQ" -eq 0 ] && break
        echo "$(date +%H:%M:%S) [$PARTITION] queue=$NQ, waiting..."
        sleep 60
    done

    TS=$(date +%Y%m%d_%H%M%S)
    MANIFEST="manifests/missing_gpu_${PARTITION}_${SEED_LO}_${SEED_HI}_${TS}.txt"
    > "$MANIFEST"
    for CONFIG in "${CONFIGS[@]}"; do
        read RHO N T <<< "$CONFIG"
        for SEED in $SEEDS; do
            F="results/abbpw_r${RHO}_N${N}_T${T}_seed${SEED}.jls"
            [ -f "$F" ] || echo "$RHO $N $T $SEED" >> "$MANIFEST"
        done
    done
    NMISSING=$(wc -l < "$MANIFEST")
    echo "$(date +%H:%M:%S) [$PARTITION] missing=$NMISSING (N=4800)"
    if [ "$NMISSING" -eq 0 ]; then
        echo "GPU range complete."; rm "$MANIFEST"; exit 0
    fi

    head -n $BATCH_MAX "$MANIFEST" > "${MANIFEST}.batch"
    NBATCH=$(wc -l < "${MANIFEST}.batch")

    JOBID=$(sbatch "${SBATCH_BASE[@]}" --array=1-${NBATCH} \
        run_array_task_gpu.sh "$(realpath "${MANIFEST}.batch")")
    echo "  jobid=$JOBID"
    sleep 30
done

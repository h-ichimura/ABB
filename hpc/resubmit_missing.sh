#!/bin/bash
# Resubmit only missing seeds.

WORKDIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && cd .. && pwd )"
cd ${WORKDIR}/hpc

COUNT=0
for N in 500 1000; do
    for SEED in $(seq 1 200); do
        [ -f results/cspline_N${N}_seed${SEED}.jls ] && continue
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

cd ${WORKDIR}/hpc
julia run_hpc_cspline.jl $N $SEED

mv cspline_N${N}_seed${SEED}.jls results/ 2>/dev/null
EOF
        COUNT=$((COUNT + 1))
    done
done

echo "Resubmitted $COUNT missing jobs"

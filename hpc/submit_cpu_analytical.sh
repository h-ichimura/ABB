#!/bin/bash
# Submit CPU jobs with analytical gradient for G=201 and G=501.
# No GPU needed. Compare G resolution effect.

SAMPLE_SIZES="500 1000"
G_VALUES="201 501"
SEEDS=$(seq 1 200)

WORKDIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && cd .. && pwd )"

mkdir -p logs results

COUNT=0
for G in $G_VALUES; do
    for N in $SAMPLE_SIZES; do
        for SEED in $SEEDS; do
            JOBNAME="ana_G${G}_N${N}_s${SEED}"
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
julia run_hpc_gpu.jl $N $SEED $G

mv cspline_gpu_G${G}_N${N}_seed${SEED}.jls results/ 2>/dev/null
EOF
            COUNT=$((COUNT + 1))
        done
    done
done

echo "Submitted $COUNT CPU analytical gradient jobs"

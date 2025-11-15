#!/bin/bash
# submit-sequential.sh - Submit chained FL training jobs that resume from checkpoints
# Usage: ./submit-sequential.sh <num_jobs> [--gpu] [--rounds-per-job N] [--dependency-type TYPE]
# Example: ./submit-sequential.sh 10 --gpu --rounds-per-job 5
# Dependency types: afterok (default), afterany (continue even if job fails), singleton (one at a time)

set -e

[ $# -lt 1 ] && { echo "Usage: $0 <num_jobs> [--gpu] [--rounds-per-job N] [--dependency-type TYPE]"; exit 1; }

NUM_JOBS=$1; shift
ROUNDS_PER_JOB=5
GPU_FLAG=""
DEPENDENCY_TYPE="afterany"  # Changed default to afterany (more forgiving)

while [[ $# -gt 0 ]]; do
  case $1 in
    --gpu) GPU_FLAG="--gpu"; shift ;;
    --rounds-per-job) ROUNDS_PER_JOB="$2"; shift 2 ;;
    --dependency-type) DEPENDENCY_TYPE="$2"; shift 2 ;;
    *) shift ;;
  esac
done

TOTAL_ROUNDS=$((NUM_JOBS * ROUNDS_PER_JOB))
echo "=== Submitting ${NUM_JOBS} chained jobs ==="
echo "Rounds per job: ${ROUNDS_PER_JOB}"
echo "Total rounds: ${TOTAL_ROUNDS}"
echo "Dependency type: ${DEPENDENCY_TYPE}"
echo ""

# Submit first job
FIRST_JOB=$(./submit-job.sh "flwr run . cluster --run-config \"num-server-rounds=${TOTAL_ROUNDS} rounds-this-job=${ROUNDS_PER_JOB}\"" ${GPU_FLAG} --name "fl_round_001" | grep -oP 'job \K[0-9]+')
echo "✓ Job 1/${NUM_JOBS}: ${FIRST_JOB}"

# Chain subsequent jobs with dependencies
PREV_JOB=$FIRST_JOB
for ((i=2; i<=NUM_JOBS; i++)); do
    JOB_SUFFIX=$(printf "fl_round_%03d" $i)
    
    # Submit job that depends on previous job
    # afterany: run regardless of previous job exit status (recommended for checkpoint recovery)
    # afterok: only run if previous job succeeded (stricter)
    # singleton: only one job with this name runs at a time
    NEXT_JOB=$(sbatch --parsable --dependency=${DEPENDENCY_TYPE}:${PREV_JOB} <<EOF | tail -1
#!/bin/bash
#SBATCH --output /home/\${USER}/logs/job%j_${JOB_SUFFIX}.out
#SBATCH --job-name ${JOB_SUFFIX}
#SBATCH --cpus-per-task=$([ -n "$GPU_FLAG" ] && echo 6 || echo 8)
#SBATCH --mem=$([ -n "$GPU_FLAG" ] && echo 120G || echo 46G)
#SBATCH --qos=$([ -n "$GPU_FLAG" ] && echo gpu_qos || echo cpu_qos)
#SBATCH --time=00:20:00
$([ -n "$GPU_FLAG" ] && echo "#SBATCH --gres=gpu:1
#SBATCH --partition=gpu" || echo "#SBATCH --partition=cpu")

set -e
echo "=== Job \${SLURM_JOB_ID} (${i}/${NUM_JOBS}) started at \$(date) ==="

export SCRATCH_DIR=/scratch/\${USER}/\${SLURM_JOB_ID}
mkdir -p "\$SCRATCH_DIR"

cleanup() {
  rm -rf "\$SCRATCH_DIR"
  echo "=== Job \${SLURM_JOB_ID} finished at \$(date) ==="
  exit \$?
}
trap cleanup EXIT

rsync -a "\$HOME/coldstart_justus/" "\$SCRATCH_DIR/repo/" && cd "\$SCRATCH_DIR/repo"
source /home/\$USER/$([ -n "$GPU_FLAG" ] && echo "hackathon-venv" || echo "hackathon-venv-cpu")/bin/activate

export DATASET_DIR="/home/\${USER}/xray-data"
export JOB_SCRATCH="\${SLURM_TMPDIR:-\${TMPDIR:-/tmp}}/job-\${SLURM_JOB_ID}"
export MIOPEN_CUSTOM_CACHE_DIR="\$JOB_SCRATCH/miopen-cache"
export MIOPEN_USER_DB_PATH="\$JOB_SCRATCH/miopen-db"
export XDG_CACHE_HOME="\$JOB_SCRATCH/xdg-cache"

mkdir -p "\$JOB_SCRATCH" "\$MIOPEN_CUSTOM_CACHE_DIR" "\$MIOPEN_USER_DB_PATH" "\$XDG_CACHE_HOME"

# Copy checkpoint directory to scratch for faster access
CHECKPOINT_HOME="\$HOME/coldstart_runs/flower_bigmodel"
CHECKPOINT_SCRATCH="\$SCRATCH_DIR/checkpoints"
if [ -d "\$CHECKPOINT_HOME" ]; then
    echo "=== Copying checkpoints from home to scratch ==="
    rsync -a "\$CHECKPOINT_HOME/" "\$CHECKPOINT_SCRATCH/"
    # Create symlink so code uses scratch checkpoints
    mkdir -p "\$HOME/coldstart_runs"
    ln -sfn "\$CHECKPOINT_SCRATCH" "\$CHECKPOINT_HOME"
fi

echo "=== Running FL training (Job ${i}/${NUM_JOBS}) ==="
flwr run . cluster --run-config "num-server-rounds=${TOTAL_ROUNDS} rounds-this-job=${ROUNDS_PER_JOB}"

# Copy checkpoints back to home
if [ -d "\$CHECKPOINT_SCRATCH" ]; then
    echo "=== Copying checkpoints back to home ==="
    rsync -a "\$CHECKPOINT_SCRATCH/" "\$CHECKPOINT_HOME/"
fi
EOF
)
    
    echo "✓ Job ${i}/${NUM_JOBS}: ${NEXT_JOB} (depends on ${PREV_JOB})"
    PREV_JOB=$NEXT_JOB
done

echo ""
echo "=== All jobs submitted successfully ==="
echo "Monitor with: squeue -u \$USER"
echo "Cancel all with: scancel -u \$USER"
echo "Checkpoints will be saved to: ~/coldstart_runs/flower_bigmodel/"

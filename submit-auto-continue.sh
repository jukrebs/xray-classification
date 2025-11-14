#!/bin/bash
# submit-auto-continue.sh - Submit a self-resubmitting FL job
# The job will automatically continue training until target rounds reached
# Usage: ./submit-auto-continue.sh [--gpu] [--total-rounds N] [--rounds-per-job N]

set -e

TOTAL_ROUNDS=50
ROUNDS_PER_JOB=5
GPU_FLAG=""

while [[ $# -gt 0 ]]; do
  case $1 in
    --gpu) GPU_FLAG="--gpu"; shift ;;
    --total-rounds) TOTAL_ROUNDS="$2"; shift 2 ;;
    --rounds-per-job) ROUNDS_PER_JOB="$2"; shift 2 ;;
    *) shift ;;
  esac
done

echo "=== Submitting auto-continuing FL job ==="
echo "Total rounds: ${TOTAL_ROUNDS}"
echo "Rounds per job: ${ROUNDS_PER_JOB}"

JOB_ID=$(sbatch --parsable <<EOF
#!/bin/bash
#SBATCH --output /home/\${USER}/logs/job%j_fl_auto.out
#SBATCH --job-name fl_auto
#SBATCH --cpus-per-task=$([ -n "$GPU_FLAG" ] && echo 6 || echo 8)
#SBATCH --mem=$([ -n "$GPU_FLAG" ] && echo 120G || echo 46G)
#SBATCH --qos=$([ -n "$GPU_FLAG" ] && echo gpu_qos || echo cpu_qos)
#SBATCH --time=00:20:00
$([ -n "$GPU_FLAG" ] && echo "#SBATCH --gres=gpu:1
#SBATCH --partition=gpu" || echo "#SBATCH --partition=cpu")

set -e
echo "=== Job \${SLURM_JOB_ID} started at \$(date) ==="

export SCRATCH_DIR=/scratch/\${USER}/\${SLURM_JOB_ID}
mkdir -p "\$SCRATCH_DIR"

cleanup() {
  local exit_code=\$?
  rm -rf "\$SCRATCH_DIR"
  echo "=== Job \${SLURM_JOB_ID} finished with exit code \$exit_code ==="
  exit \$exit_code
}
trap cleanup EXIT

rsync -a "\$HOME/coldstart_justus/" "\$SCRATCH_DIR/repo/" && cd "\$SCRATCH_DIR/repo"
source /home/\$USER/$([ -n "$GPU_FLAG" ] && echo "hackathon-venv" || echo "hackathon-venv-cpu")/bin/activate

export DATASET_DIR="/home/\${USER}/xray-data"
export JOB_SCRATCH="\${SLURM_TMPDIR:-\${TMPDIR:-/tmp}}/job-\${SLURM_JOB_ID}"

mkdir -p "\$JOB_SCRATCH"

# Copy checkpoints to/from scratch for performance
CHECKPOINT_HOME="\$HOME/coldstart_runs/flower_bigmodel"
CHECKPOINT_SCRATCH="\$SCRATCH_DIR/checkpoints"

if [ -d "\$CHECKPOINT_HOME" ]; then
    echo "=== Copying existing checkpoints to scratch ==="
    rsync -a "\$CHECKPOINT_HOME/" "\$CHECKPOINT_SCRATCH/"
    # Symlink to use scratch during training
    mkdir -p "\$HOME/coldstart_runs"
    ln -sfn "\$CHECKPOINT_SCRATCH" "\$CHECKPOINT_HOME"
fi

echo "=== Running FL training ==="
flwr run . cluster --run-config "num-server-rounds=${TOTAL_ROUNDS} rounds-this-job=${ROUNDS_PER_JOB}"

# Copy checkpoints back
if [ -d "\$CHECKPOINT_SCRATCH" ]; then
    echo "=== Saving checkpoints to home ==="
    rsync -a "\$CHECKPOINT_SCRATCH/" "\$CHECKPOINT_HOME/"
fi

# Check if we need to continue
if [ -f "\$CHECKPOINT_HOME/latest.pt" ]; then
    # Check current round using Python
    CURRENT_ROUND=\$(python3 -c "
import pickle
try:
    with open('\$CHECKPOINT_HOME/latest.pt', 'rb') as f:
        ckpt = pickle.load(f)
    print(ckpt['round'])
except:
    print(0)
")
    
    echo "Completed round: \${CURRENT_ROUND}/${TOTAL_ROUNDS}"
    
    if [ "\${CURRENT_ROUND}" -lt "${TOTAL_ROUNDS}" ]; then
        echo "=== Submitting continuation job ==="
        sbatch \$0 $([ -n "$GPU_FLAG" ] && echo "--gpu") --total-rounds ${TOTAL_ROUNDS} --rounds-per-job ${ROUNDS_PER_JOB}
    else
        echo "=== Training complete! ==="
    fi
fi
EOF
)

echo "Submitted job: ${JOB_ID}"
echo "Monitor: squeue -j ${JOB_ID}"
echo "Logs: ~/logs/job${JOB_ID}_fl_auto.out"

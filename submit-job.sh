#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="$REPO_ROOT/.env"

if [[ -f "$ENV_FILE" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "$ENV_FILE"
  set +a
fi

usage() {
  cat <<'EOF'
Usage: ./submit-job.sh "COMMAND" [options]

Options:
  --name NAME        Logical job/run name (default: flwr-YYMMDD-HHMMSS)
  --gpu              Request a single GPU (use --gpus for >1)
  --gpus N           Request N GPUs
  --cpus N           Number of CPUs per task (default: 4)
  --mem SIZE         Memory request, e.g. 32G (default: 32G)
  --time HH:MM:SS    Wall-clock limit (default: 00:20:00)
  --partition NAME   SLURM partition/queue (default: value of $SLURM_PARTITION or gpu)
  -h, --help         Show this help

If SLURM/sbatch is unavailable, the command is executed locally.
Environment variables from .env are automatically exported.
EOF
}

if [[ $# -lt 1 ]]; then
  usage
  exit 1
fi

COMMAND="$1"
shift

JOB_NAME="flwr-$(date +%y%m%d-%H%M%S)"
GPUS=0
CPUS=${CPUS_PER_JOB:-4}
MEM=${MEM_PER_JOB:-32G}
TIME=${TIME_PER_JOB:-00:20:00}
PARTITION=${SLURM_PARTITION:-gpu}
LOG_DIR=${LOG_DIR:-$HOME/logs}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --name)
      JOB_NAME="$2"
      shift 2
      ;;
    --gpu)
      GPUS=1
      shift
      ;;
    --gpus)
      GPUS="$2"
      shift 2
      ;;
    --cpus)
      CPUS="$2"
      shift 2
      ;;
    --mem)
      MEM="$2"
      shift 2
      ;;
    --time)
      TIME="$2"
      shift 2
      ;;
    --partition)
      PARTITION="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1"
      usage
      exit 1
      ;;
  esac
done

mkdir -p "$LOG_DIR"

warn_missing_var() {
  local var_name=$1
  local message=$2
  if [[ -z "${!var_name:-}" ]]; then
    echo "[submit-job] Warning: $message"
  fi
}

warn_missing_var "DATASET_DIR" "DATASET_DIR is unset; dataloaders will fail unless you export it."
warn_missing_var "WANDB_API_KEY" "WANDB_API_KEY is unset; runs will skip W&B logging."

if command -v sbatch >/dev/null 2>&1; then
  TMP_SCRIPT="$(mktemp)"
  {
    echo "#!/bin/bash"
    echo "#SBATCH --job-name=${JOB_NAME}"
    echo "#SBATCH --cpus-per-task=${CPUS}"
    echo "#SBATCH --time=${TIME}"
    echo "#SBATCH --mem=${MEM}"
    echo "#SBATCH --partition=${PARTITION}"
    echo "#SBATCH --output=${LOG_DIR}/${JOB_NAME}_%j.out"
    echo "#SBATCH --error=${LOG_DIR}/${JOB_NAME}_%j.err"
    if (( GPUS > 0 )); then
      echo "#SBATCH --gres=gpu:${GPUS}"
    fi
    cat <<'EOF'
set -euo pipefail
EOF
    printf 'cd %q\n' "$REPO_ROOT"
    printf 'export JOB_NAME=%q\n' "$JOB_NAME"
    printf 'COMMAND=%q\n' "$COMMAND"
    cat <<'EOF'
echo "[submit-job] Launching command: ${COMMAND}"
bash -lc "${COMMAND}"
EOF
  } >"$TMP_SCRIPT"

  JOB_ID=$(sbatch "$TMP_SCRIPT")
  rm -f "$TMP_SCRIPT"
  echo "[submit-job] Submitted ${JOB_NAME}: ${JOB_ID}"
  echo "[submit-job] Logs: ${LOG_DIR}/${JOB_NAME}_<jobid>.out"
else
  echo "[submit-job] sbatch not found. Running locally."
  export JOB_NAME
  cd "$REPO_ROOT"
  echo "[submit-job] Launching command: ${COMMAND}"
  bash -lc "${COMMAND}"
fi

# Berlin25 X-Ray – CXformer Baseline

Federated training setup for the NIH Chest X-Ray challenge. This project mirrors the official `coldstart` scaffolding so all infrastructure scripts (including the cluster job launcher) work unmodified, but swaps in a CXformer encoder plus a custom binary head.

## 🩻 Dataset Recap

- **Hospitals:** A (elderly males, AP views), B (younger females, PA views), C (mixed demographics, rare diseases)
- **Task:** Binary classification – “any pathology” vs “no finding”
- **Splits:** Patient-disjoint train/eval/test, preprocessed to 128×128 grayscale tensors per hospital

## 🧠 Model Highlights

- Uses `m42-health/CXformer-base` (Dinov2-derived encoder trained via SSL on chest X-rays)
- Inputs are converted to RGB and passed through the Hugging Face `AutoImageProcessor`, so we can keep the lightweight 128×128 on-disk datasets
- Classification head: LayerNorm → Linear → GELU → Dropout → Linear (single logit), trained with `BCEWithLogitsLoss`
- Encoder is frozen by default (`CXFORMER_FREEZE_ENCODER=1`); unset or set to `0` to fine-tune the full backbone later
- Additional knobs via env vars:
  - `CXFORMER_MODEL_NAME`, `CXFORMER_CLASSIFIER_DIM`, `CXFORMER_CLASSIFIER_DROPOUT`

## ⚙️ Quick Start

### 0. Install Dependencies

```bash
cd /path/to/xray-classification/berlin25_xray
python -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install -e .
```

### 1. Configure Environment

Create `.env` (or export manually) with:

```bash
DATASET_DIR=/shared/hackathon/datasets
WANDB_API_KEY=...
WANDB_ENTITY=...
WANDB_PROJECT=...
```

Source it before launching jobs so both Flower and W&B inherit the variables.

### 2. Local Smoke Test

```bash
DATASET_DIR=/shared/hackathon/datasets \
flwr run berlin25_xray local-simulation --num-supernodes 1 --run-config num-server-rounds=2
```

Set `CXFORMER_FREEZE_ENCODER=0` when you want to train the full encoder.

### 3. Cluster Run

```bash
# CPU run (3 supernodes, matches config defaults)
flwr run berlin25_xray cluster-cpu --stream

# GPU run with a custom job name
JOB_NAME=berlin_cxformer_v1 \
flwr run berlin25_xray cluster-gpu --stream --gpu
```

Both commands read the `[tool.flwr.federations.*]` options in `pyproject.toml` so resource requests match the baseline.

### 4. Evaluation

After training, update `evaluate.py` with your saved checkpoint path and run:

```bash
DATASET_DIR=/shared/hackathon/datasets \
python evaluate.py
```

## 📈 Monitoring & Logging

- Metrics are logged per hospital plus globally; the best round saves a checkpoint under `./models/`
- W&B logging is automatic when `WANDB_API_KEY` and `WANDB_PROJECT` are set
- Use `squeue -u $USER` / `sacct` for SLURM status, or follow your cluster’s monitoring docs

## 🗂️ Repo Layout

```
berlin25_xray/
├── pyproject.toml              # Flower app configuration
├── README.md                   # This file
├── evaluate.py                 # Standalone evaluation script
└── cold_start_hackathon/       # Flower client/server/model code
    ├── client_app.py
    ├── server_app.py
    ├── task.py                 # CXformer model + loaders
    ├── util.py
    └── __init__.py
```

Everything else follows the official Flower baseline so existing tooling (“julian-submit-job.sh”, etc.) can operate without changes.

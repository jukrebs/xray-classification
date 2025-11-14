# Cold Start Hackathon: Federated Learning for X-ray Classification

This challenge builds on the NIH Chest X-Ray dataset, which contains over 112,000 medical images from 30,000 patients. Participants will explore how federated learning can enable robust diagnostic models that generalize across hospitals, without sharing sensitive patient data.

## Background

In real healthcare systems, hospitals differ in their imaging devices, patient populations, and clinical practices. A model trained in one hospital often struggles in another, but because the data distributions differ.

Your task is to design a model that performs reliably across diverse hospital environments. By simulating a federated setup, where each hospital trains on local data and only model updates are shared, you’ll investigate how distributed AI can improve performance and robustness under privacy constraints.

## 🏥 Hospital Data Distribution

Chest X-rays are among the most common and cost-effective imaging exams, yet diagnosing them remains challenging.
For this challenge, the dataset has been artificially partitioned into hospital silos to simulate a federated learning scenario with **strong non-IID characteristics**. Each patient appears in only one silo. However, age, sex, view position, and pathology distributions vary across silos.

Each patient appears in only one hospital. All splits (train/eval/test) are patient-disjoint to prevent data leakage.

### Hospital A: Portable Inpatient (42,093 test, 5,490 eval)
- **Demographics**: Elderly males (age 60+)
- **Equipment**: AP (anterior-posterior) view dominant
- **Common findings**: Fluid-related conditions (Effusion, Edema, Atelectasis)

### Hospital B: Outpatient Clinic (21,753 train, 2,860 eval)
- **Demographics**: Younger females (age 20-65)
- **Equipment**: PA (posterior-anterior) view dominant
- **Common findings**: Nodules, masses, pneumothorax

### Hospital C: Mixed with Rare Conditions (20,594 train, 2,730 eval)
- **Demographics**: Mixed age and gender
- **Equipment**: PA view preferred
- **Common findings**: Rare conditions (Hernia, Fibrosis, Emphysema)


## 📊 Task Details

**Binary classification**: Detect presence of any pathological finding
- **Class 0**: No Finding
- **Class 1**: Any Finding present

**Pathologies (15 types)**: Atelectasis, Cardiomegaly, Effusion, Infiltration, Mass, Nodule, Pneumonia, Pneumothorax, Consolidation, Edema, Emphysema, Fibrosis, Pleural_Thickening, Hernia

**Evaluation Metric**: [AUROC](https://developers.google.com/machine-learning/crash-course/classification/roc-and-auc)


## 🧠 CXformer Baseline

We now fine-tune the [m42-health/CXformer-base](https://huggingface.co/m42-health/CXformer-base) checkpoint instead of the smaller ResNet baseline. CXformer reuses Dinov2 weights that were further self-supervised on large collections of chest X-rays, which gives the federation a much stronger prior and speeds up convergence on each hospital silo.

- The encoder comes from Hugging Face (`AutoModel` + `AutoImageProcessor`) and is frozen by default so that only a lightweight classification head is trained during the first rounds.
- The processor automatically resizes inputs to 518×518 and applies the preprocessing CXformer expects. We load the 224×224 preprocessed hospital datasets and upsample them on the fly so no additional preprocessing step is required.
- Dependencies for `transformers`, `accelerate`, and `safetensors` are included in `pyproject.toml`. Run `pip install -e .` again after pulling these changes.
- You can override a few knobs via environment variables:
  - `CXFORMER_MODEL_NAME`: swap in another checkpoint if needed.
  - `CXFORMER_FREEZE_ENCODER`: set to `0` when you are ready to unfreeze and fully fine-tune CXformer.
  - `CXFORMER_CLASSIFIER_DIM` / `CXFORMER_CLASSIFIER_DROPOUT`: adapt the head capacity for experimentation.

### Start With Hospital A Only

Local simulations now launch all three hospitals by default, mirroring the cluster deployments. If you want to iterate quickly on Hospital A only, supply `--num-supernodes 1` (or edit the config temporarily) when running Flower locally.

Example commands:

```bash
# Install the updated dependencies
pip install -e .

# Launch a single-hospital Flower round locally (Hospital A)
DATASET_DIR=/shared/hackathon/datasets \
flwr run coldstart_real local-simulation --num-supernodes 1

# Unfreeze CXformer and try a longer run once things look good
CXFORMER_FREEZE_ENCODER=0 \
DATASET_DIR=/shared/hackathon/datasets \
flwr run coldstart_real local-simulation --num-supernodes 1
```

## 🚀 Quick Start

### 1. Setup Environment

```bash
# Clone your team's repository
git clone https://github.com/YOUR_ORG/hackathon-2025-team-YOUR_TEAM.git
cd hackathon-2025-team-YOUR_TEAM

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install --upgrade pip
pip install -e .
```

### 2. Configure Dataset + W&B Credentials

Copy the provided template, fill in your dataset path (usually `/shared/hackathon/datasets`) and W&B service-account credentials, then source it before running jobs:

```bash
cp .env.example .env
# edit .env with DATASET_DIR, WANDB_API_KEY, WANDB_ENTITY, WANDB_PROJECT, etc.
source .env
```

The `submit-job.sh` helper (described below) auto-loads `.env` so jobs launched via SLURM inherit these variables without extra typing.

### 3. Test Locally (Optional)

```bash
python local_train.py --hospital A
```

Note: Full datasets are only available on the cluster.

### 4. Submit Jobs to Cluster

```bash
# From the repository root

# Submit training job (defaults: 4 CPUs, 32GB RAM, 20min wall time, no GPU)
./submit-job.sh "flwr run coldstart_real cluster-gpu --stream"

# Request a GPU and custom run name (shows up as JOB_NAME inside Flower/W&B)
./submit-job.sh "flwr run coldstart_real cluster-gpu --stream" --gpu --name exp_lr001

# Launch the evaluation script with the same infrastructure
./submit-job.sh "python coldstart_real/evaluate.py" --gpu --name eval_v5
```

If `sbatch` is available, the helper submits to SLURM and streams logs to `~/logs/<job>_<jobid>.out`. Without SLURM (e.g., local dev), it simply runs the command inline.

### 5. Monitor Results

```bash
# Check job status
squeue -u $USER

# View logs
tail -f ~/logs/exp_lr001_*.out

# View W&B dashboard (project set via WANDB_PROJECT)
# https://wandb.ai/YOUR_ENTITY/YOUR_PROJECT
```

#### W&B Tracking Tips

- Ensure `.env` exports `WANDB_API_KEY`, `WANDB_PROJECT`, and optionally `WANDB_ENTITY`. The server app auto-logins with those values and tags runs using the `JOB_NAME` supplied to `submit-job.sh`.
- For dry runs without logging, unset `WANDB_API_KEY` (or set `WANDB_MODE=offline`) before launching a job.
- Use `wandb agent` or sweep configs by passing additional `WANDB_*` variables into `.env` or the shell before calling the submission script.


## 📚 Dataset Details

Datasets on cluster:
- **Raw**: `/shared/hackathon/datasets/xray_fl_datasets/`
- **Preprocessed (128x128)**: `/shared/hackathon/datasets/xray_fl_datasets_preprocessed_128/`

These are automatically linked in your job workspace.

## ⚙️ Resource Limits

Per job:
- **1 GPU**
- **32GB RAM**
- **20 minutes** runtime
- **Max 4 concurrent jobs** per team

## 📊 Weights & Biases

All metrics automatically logged to W&B: `https://wandb.ai/coldstart2025-teamXX/coldstart2025`

Login with your team's service account credentials (provided by organizers).


## 📝 Dataset Reference

```
@article{wang2017chestxray,
  title={ChestX-ray8: Hospital-scale Chest X-ray Database and Benchmarks},
  author={Wang, Xiaosong and Peng, Yifan and Lu, Le and Lu, Zhiyong and
          Bagheri, Mohammadhadi and Summers, Ronald M},
  journal={CVPR},
  year={2017}
}
```

---

**Good luck, and happy hacking!** 🚀

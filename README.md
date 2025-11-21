## Berlin X-Ray Federated Classification (Hackathon Submission)

This repository contains our hackathon solution for federated chest X‑ray classification built with **Flower** and **PyTorch**.  
The goal is to train a federated model across multiple hospitals.


This repo is based on the boilerplate for the 
[[Cold Start:] Distributed AI Hack Berlin](https://github.com/exalsius/hackathon-coldstart2025/tree/main)
Hackathon.
---

## Prerequisites
You need the following:
- Around 160 GB of space for the dataset
- A GPU to run the model training (adjust batch size based on GPU memory)

## Installation
First, clone this repository:
```bash
git clone https://github.com/jukrebs/xray-classification.git
cd xray-classification
```

We use [uv](https://docs.astral.sh/uv/) for managing our project and virtual env. Make sure you have uv installed.
Create a virtual environment:
```bash
uv venv
```

## Preparation and preprocessing
Run the `prepare_datasets.py` script to prepare the datasets. This will download the **NIH Chest X-ray-14 dataset** and create non-IID (non-independent and identically distributed) federated learning datasets by probabilistically assigning patients to four hospital silos (A, B, C, D). Hospitals A, B, and C are used in training, while Hospital D is used for evaluation.
```bash
uv run scripts/prepare_datasets.py
```

Afterwards run the preprocessing of the datasets for 128 image size.
```bash
uv run scripts/preprocess_datasets.py
```

To create the dataset with 224 image size, use:
```bash
uv run scripts/preprocess_datasets.py --image-size 224
```

## Training a model
To train a model with **Flower** use:
```bash
uv run flwr run . cluster
```

This will spawn three clients, one for each hospital. Each client has equal GPU share. This can be configured in the `pyproject.toml`. Additionally, you can configure epochs per round, total rounds, batch size, image size, and learning rate in this file.

If you want to spawn a quick run, use:
```bash
uv run scripts/local_train.py
```
This will train a model on only one hospital without the use of the **Flower** framework.

## Evaluate the model
All trained models will be saved to `models`. To evaluate your model, configure your model path in `evaluate.py` and run the script:
```bash
uv run scripts/evaluate.py
```

This script will produce a summary of your model's performance like so:

```text
MODEL EVALUATION
Loading model from models/hospital_A_size224_model.pt...
Model loaded on cuda.

Evaluating...
Loaded xray-classification/xray/preprocessed_128/HospitalA/eval
  Hospital A      AUROC: 0.7340 (n=5490)
Loaded xray-classification/xray/preprocessed_128/HospitalB/eval
  Hospital B      AUROC: 0.7288 (n=2860)
Loaded xray-classification/xray/preprocessed_128/HospitalC/eval
  Hospital C      AUROC: 0.7125 (n=2730)
Loaded xray-classification/xray/preprocessed_128/Test/test_A
  Test A          AUROC: 0.7227 (n=5671)
Loaded xray-classification/xray/preprocessed_128/Test/test_B
  Test B          AUROC: 0.7177 (n=2757)
Loaded xray-classification/xray/preprocessed_128/Test/test_C
  Test C          AUROC: 0.7220 (n=2617)
Loaded xray-classification/xray/preprocessed_128/Test/test_D
  Test D (OOD)    AUROC: 0.7205 (n=5539)

  Eval Avg        AUROC: 0.7306
  Test Avg        AUROC: 0.7240
```

## Improving the model
Please feel free to improve the current model or try a new approach. Open a PR, explain what you have done, and add your model performance.


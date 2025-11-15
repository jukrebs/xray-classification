"""Evaluation script for teams and organizers.

Usage: python evaluate.py
"""

import os

import torch
import wandb
from sklearn.metrics import roc_auc_score

from berlin25_xray.task import Net, load_data, test

# Suppress W&B directory warning
os.environ["WANDB_DIR"] = os.path.expanduser("~/.cache/wandb")

# W&B model path: update with your best model
# Format: "your-wandb-username/your-project-name/model-artifact-name:version"
WANDB_MODEL_PATH = "justus-krebs-technische-universit-t-berlin/hackathon/job1266_dense_feder_uf128_round2_auroc7675:v0"
DATASET_DIR = os.environ["DATASET_DIR"]


def evaluate_split(model, dataset_name, split_name, device):
    """Evaluate on any dataset split."""
    loader = load_data(dataset_name, split_name, batch_size=32)
    _, _, _, _, _, probs, labels = test(model, loader, device)
    return roc_auc_score(labels, probs), len(loader.dataset)


def main():
    print("=" * 80)
    print("MODEL EVALUATION")
    print("=" * 80)

    print(f"\nLoading {WANDB_MODEL_PATH}...")
    run = wandb.init()
    artifact = run.use_artifact(WANDB_MODEL_PATH, type="model")
    artifact_dir = artifact.download()
    model_path = next(p for p in __import__("pathlib").Path(artifact_dir).glob("*.pt"))

    # Load model
    model = Net()
    model.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=True))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"Model loaded on {device}.")

    # Save model to disk for further use
    saved_model_path = os.path.expanduser("~/models/uf128_round2.pt")
    torch.save(model.state_dict(), saved_model_path)
    print(f"Model saved to {saved_model_path}")

    # Evaluate
    print("\nEvaluating...")
    datasets_to_test = [
        ("Hospital A", "HospitalA", "eval"),
        ("Hospital B", "HospitalB", "eval"),
        ("Hospital C", "HospitalC", "eval"),
        ("Test A", "Test", "test_A"),
        ("Test B", "Test", "test_B"),
        ("Test C", "Test", "test_C"),
        ("Test D (OOD)", "Test", "test_D"),
    ]

    results = {}
    for display_name, dataset_name, split_name in datasets_to_test:
        try:
            auroc, n = evaluate_split(model, dataset_name, split_name, device)
            results[display_name] = (auroc, n)
            print(f"  {display_name:<15} AUROC: {auroc:.4f} (n={n})")
        except FileNotFoundError:
            # Test dataset doesn't exist for participants - skip silently
            pass

    # Weighted average for Hospital eval splits
    hospital_results = [
        (a, n) for name, (a, n) in results.items() if name.startswith("Hospital")
    ]
    if hospital_results:
        weighted_auroc = sum(a * n for a, n in hospital_results) / sum(
            n for _, n in hospital_results
        )
        print(f"  {'Weighted Avg':<15} AUROC: {weighted_auroc:.4f}")

    print("\n" + "=" * 80)
    return 0


if __name__ == "__main__":
    exit(main())

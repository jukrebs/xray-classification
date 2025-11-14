"""Local training script to sanity-check the CXformer baseline."""

import torch
from sklearn.metrics import roc_auc_score

from berlin25_xray.cold_start_hackathon.task import (
    Net,
    compute_metrics_from_confusion_matrix,
    load_data,
)
from berlin25_xray.cold_start_hackathon.task import test as test_fn
from berlin25_xray.cold_start_hackathon.task import train as train_fn

HOSPITAL = "A"  # A, B, or C
EPOCHS = 3
LEARNING_RATE = 1e-4
BATCH_SIZE = 16
IMAGE_SIZE = 128


def main():
    """Train X-ray classifier locally on a specific hospital's data."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset_name = f"Hospital{HOSPITAL}"
    print(f"\nLoading {dataset_name} data at {IMAGE_SIZE}x{IMAGE_SIZE} resolution...")
    trainloader = load_data(dataset_name, "train", image_size=IMAGE_SIZE, batch_size=BATCH_SIZE)
    evalloader = load_data(dataset_name, "eval", image_size=IMAGE_SIZE, batch_size=BATCH_SIZE)
    print(f"Train batches: {len(trainloader)}, Eval batches: {len(evalloader)}")

    model = Net().to(device)
    print(f"\nStarting training for {EPOCHS} epochs on {device}...")
    for epoch in range(1, EPOCHS + 1):
        print(f"\n[Epoch {epoch}/{EPOCHS}] Training...")
        train_loss = train_fn(
            model, trainloader, epochs=1, lr=LEARNING_RATE, device=device
        )
        print(f"  Train loss: {train_loss:.4f}")

        print(f"[Epoch {epoch}/{EPOCHS}] Evaluating...")
        eval_loss, tp, tn, fp, fn, probs, labels = test_fn(model, evalloader, device)
        auroc = roc_auc_score(labels, probs)
        cm_metrics = compute_metrics_from_confusion_matrix(tp, tn, fp, fn)
        print(f"  Eval loss: {eval_loss:.4f}")
        print(f"  AUROC: {auroc:.4f}")
        print(
            f"  Sensitivity: {cm_metrics['sensitivity']:.4f} | "
            f"Specificity: {cm_metrics['specificity']:.4f} | "
            f"Precision: {cm_metrics['precision']:.4f} | "
            f"F1: {cm_metrics['f1']:.4f}"
        )

    model_path = f"hospital_{HOSPITAL}_model.pt"
    torch.save(model.state_dict(), model_path)
    print(f"\nTraining complete! Model saved to {model_path}")


if __name__ == "__main__":
    main()

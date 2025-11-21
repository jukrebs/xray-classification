"""Local training script for single-hospital X-Ray classification."""

import argparse
import os
from typing import Optional

import torch
from datasets import load_from_disk
from sklearn.metrics import roc_auc_score

from berlin25_xray.task import (
    Net,
    compute_metrics_from_confusion_matrix,
)
from berlin25_xray.task import test as test_fn

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

DEFAULT_EPOCHS = 5
DEFAULT_LR = 1e-3
DEFAULT_IMAGE_SIZE = 224
DEFAULT_BATCH_SIZE = 16

VALID_IMAGE_SIZES = {128, 224}
VALID_HOSPITALS = {"A", "B", "C"}


def load_preprocessed_dataloader(
    hospital: str,
    split: str,
    image_size: int,
    batch_size: int,
) -> torch.utils.data.DataLoader:
    """Load a single split for one hospital from the preprocessed folders."""
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "xray"))
    dataset_dir = os.path.join(base_dir, f"preprocessed_{image_size}")
    dataset_path = os.path.join(dataset_dir, f"Hospital{hospital}")

    if not os.path.exists(dataset_path):
        raise FileNotFoundError(
            f"Could not locate preprocessed dataset for image_size={image_size}, hospital={hospital}."
        )

    full_dataset = load_from_disk(dataset_path)
    if split not in full_dataset:
        raise KeyError(
            f"Split '{split}' not found at {dataset_path}. Available: {list(full_dataset.keys())}"
        )
    # Ensure correct torch formatting
    for s in full_dataset.keys():
        full_dataset[s].set_format(type="torch", columns=["x", "y"])
    data = full_dataset[split]
    shuffle = split == "train"
    num_workers = min(8, os.cpu_count() or 1)
    dl_kwargs = {
        "dataset": data,
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
    }
    if num_workers > 0:
        dl_kwargs["prefetch_factor"] = 2
        dl_kwargs["persistent_workers"] = True
    return torch.utils.data.DataLoader(**dl_kwargs)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Local single-hospital X-ray training")
    parser.add_argument(
        "--hospital",
        type=str,
        default="A",
        choices=sorted(VALID_HOSPITALS),
        help="Hospital letter (A/B/C)",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=DEFAULT_IMAGE_SIZE,
        choices=sorted(VALID_IMAGE_SIZES),
        help="Image size (128 or 224)",
    )
    parser.add_argument(
        "--epochs", type=int, default=DEFAULT_EPOCHS, help="Number of training epochs"
    )
    parser.add_argument(
        "--batch-size", type=int, default=DEFAULT_BATCH_SIZE, help="Batch size"
    )
    parser.add_argument("--lr", type=float, default=DEFAULT_LR, help="Learning rate")
    parser.add_argument(
        "--max-batches",
        type=int,
        default=None,
        help="Limit number of batches per epoch for quick tests",
    )
    return parser.parse_args()


def main():
    """Main training loop for local single-hospital training."""
    args = parse_args()
    hospital = args.hospital
    image_size = args.image_size
    epochs = args.epochs
    batch_size = args.batch_size
    lr = args.lr
    max_batches: Optional[int] = args.max_batches

    if hospital not in VALID_HOSPITALS:
        raise ValueError(f"Hospital must be one of {VALID_HOSPITALS}")
    if image_size not in VALID_IMAGE_SIZES:
        raise ValueError(f"Image size must be one of {VALID_IMAGE_SIZES}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print(f"\nLoading Hospital{hospital} (image_size={image_size}) data...")
    trainloader = load_preprocessed_dataloader(
        hospital, "train", image_size, batch_size
    )
    evalloader = load_preprocessed_dataloader(hospital, "eval", image_size, batch_size)
    print(f"Train batches: {len(trainloader)}, Eval batches: {len(evalloader)}")
    if max_batches:
        print(f"Limiting to {max_batches} batches per epoch")

    model = Net(image_size=image_size).to(device)
    print(f"\nStarting training for {epochs} epochs...")
    for epoch in range(1, epochs + 1):
        print(f"\n[Epoch {epoch}/{epochs}] Training...")
        model.train()
        running_loss = 0.0
        criterion = torch.nn.BCEWithLogitsLoss().to(device)
        optimizer = torch.optim.AdamW(
            (p for p in model.parameters() if p.requires_grad), lr=lr, weight_decay=0.01
        )
        # OneCycle over full epoch (may shorten if max_batches)
        steps_per_epoch = max_batches if max_batches else len(trainloader)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=lr,
            epochs=1,
            steps_per_epoch=steps_per_epoch,
            pct_start=0.1,
            div_factor=10.0,
            final_div_factor=100.0,
        )
        batch_index = 0
        for batch in trainloader:
            batch_index += 1
            x = batch["x"].to(device, non_blocking=True).float()
            y = batch["y"].to(device, non_blocking=True).float()
            optimizer.zero_grad()
            outputs = model(x)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            scheduler.step()
            running_loss += loss.item()
            if max_batches and batch_index >= max_batches:
                break
        avg_train_loss = running_loss / batch_index
        print(f"  Train loss: {avg_train_loss:.4f}")

        print(f"[Epoch {epoch}/{epochs}] Evaluating...")
        eval_loss, tp, tn, fp, fn, all_probs, all_labels = test_fn(
            model, evalloader, device
        )
        metrics = compute_metrics_from_confusion_matrix(tp, tn, fp, fn)
        try:
            auroc = roc_auc_score(all_labels, all_probs)
        except Exception:  # pylint: disable=broad-except
            auroc = float("nan")  # Happens when only one class present
        print(f"  Eval loss: {eval_loss:.4f}")
        print(f"  AUROC: {auroc:.4f}")
        print(f"  F1: {metrics['f1']:.4f}")
        print(f"  Sensitivity (Recall): {metrics['sensitivity']:.4f}")
        print(f"  Specificity: {metrics['specificity']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")

    models_dir = os.path.join(PROJECT_ROOT, "models")
    os.makedirs(models_dir, exist_ok=True)
    model_path = os.path.join(
        models_dir, f"hospital_{hospital}_size{image_size}_model.pt"
    )

    save_dict = {"model_state_dict": model.state_dict(), "auroc": auroc}
    torch.save(save_dict, model_path)
    print(f"\nTraining complete! Model saved to {model_path} with AUROC: {auroc:.4f}")


if __name__ == "__main__":
    main()

"""Improved task.py with feature selection and advanced sampling strategies.

This file demonstrates how to integrate all improvements into your existing codebase.
You can either:
1. Replace your existing task.py with this file, OR
2. Gradually integrate specific improvements from this file into your existing code
"""

import os

import numpy as np
import torch
from datasets import load_from_disk
from torch.utils.data import DataLoader
from tqdm import tqdm

# Import improved components
from berlin25_xray.improved_model import get_model
from berlin25_xray.sampling_strategies import create_sampler
from berlin25_xray.training_utils import (
    GradientClipper,
    calculate_pos_weight,
    get_loss_function,
    setup_optimizer,
)

PARTITION_HOSPITAL_MAP = {
    0: "A",
    1: "B",
    2: "C",
}

hospital_datasets = {}  # Cache loaded hospital datasets


# ============================================================================
# CONFIGURATION: Adjust these to enable/disable improvements
# ============================================================================
CONFIG = {
    # Model architecture: 'resnet18', 'attention_resnet18', 'efficientnet_b0',
    # 'multiscale', 'densenet121'
    "model_architecture": "attention_resnet18",
    # Loss function: 'bce', 'focal', 'weighted_bce', 'label_smoothing'
    "loss_function": "focal",
    "loss_params": {"alpha": 0.25, "gamma": 2.0},  # For focal loss
    # Sampling strategy: None, 'balanced', 'weighted', 'federated'
    "sampling_strategy": "balanced",
    # Optimizer: 'adam', 'adamw', 'sgd', 'rmsprop'
    "optimizer": "adamw",
    # Enable gradient clipping
    "gradient_clipping": True,
    "max_grad_norm": 1.0,
    # Dropout rate (if model supports it)
    "dropout_rate": 0.3,
}


def Net():
    """Create model instance using configuration."""
    return get_model(CONFIG["model_architecture"])


def collate_preprocessed(batch):
    """Collate function for preprocessed data."""
    result = {}
    for key in batch[0].keys():
        if key in ["x", "y"]:
            result[key] = torch.stack([torch.tensor(item[key]) for item in batch])
        else:
            result[key] = [item[key] for item in batch]
    return result


def load_data(
    dataset_name: str,
    split_name: str,
    image_size: int = 128,
    batch_size: int = 16,
):
    """Load hospital X-ray data with optional advanced sampling.

    Args:
        dataset_name: Dataset name ("HospitalA", "HospitalB", "HospitalC")
        split_name: Split name ("train", "eval")
        image_size: Image size (128 or 224)
        batch_size: Number of samples per batch
    """
    dataset_dir = os.environ["DATASET_DIR"]

    # Use preprocessed dataset based on image_size
    cache_key = f"{dataset_name}_{split_name}_{image_size}"
    dataset_path = (
        f"{dataset_dir}/xray_fl_datasets_preprocessed_{image_size}/{dataset_name}"
    )

    # Load and cache dataset
    global hospital_datasets
    if cache_key not in hospital_datasets:
        full_dataset = load_from_disk(dataset_path)
        hospital_datasets[cache_key] = full_dataset[split_name]
        print(f"Loaded {dataset_path}/{split_name}")

    data = hospital_datasets[cache_key]

    # Create sampler based on configuration
    sampler = None
    shuffle = False

    if split_name == "train" and CONFIG["sampling_strategy"]:
        # Use advanced sampling for training
        sampler = create_sampler(
            data,
            strategy=CONFIG["sampling_strategy"],
            batch_size=batch_size
            if CONFIG["sampling_strategy"] == "balanced"
            else None,
        )
        # When using sampler, shuffle must be False
        shuffle = False
    elif split_name == "train":
        # Default shuffling for training without sampler
        shuffle = True

    # Handle batch sampler vs regular sampler
    if sampler is not None and CONFIG["sampling_strategy"] == "balanced":
        # BalancedBatchSampler is a batch sampler
        dataloader = DataLoader(
            data,
            batch_sampler=sampler,
            num_workers=4,
            collate_fn=collate_preprocessed,
        )
    else:
        # Regular sampler or no sampler
        dataloader = DataLoader(
            data,
            batch_size=batch_size,
            shuffle=shuffle,
            sampler=sampler,
            num_workers=4,
            collate_fn=collate_preprocessed,
        )

    return dataloader


def train(net, trainloader, epochs, lr, device):
    """Enhanced training function with advanced techniques.

    Args:
        net: Neural network model
        trainloader: Training data loader
        epochs: Number of training epochs
        lr: Learning rate
        device: Training device (CPU/GPU)

    Returns:
        Average training loss
    """
    net.to(device)

    # Setup loss function based on configuration
    if CONFIG["loss_function"] == "weighted_bce":
        # Calculate pos_weight from data
        pos_weight = calculate_pos_weight(trainloader).to(device)
        criterion = get_loss_function(
            CONFIG["loss_function"], pos_weight=pos_weight
        ).to(device)
    else:
        criterion = get_loss_function(
            CONFIG["loss_function"], **CONFIG.get("loss_params", {})
        ).to(device)

    # Setup optimizer
    optimizer = setup_optimizer(
        net,
        optimizer_type=CONFIG["optimizer"],
        lr=lr,
        weight_decay=1e-4,
    )

    # Setup gradient clipping
    grad_clipper = None
    if CONFIG["gradient_clipping"]:
        grad_clipper = GradientClipper(max_norm=CONFIG["max_grad_norm"])

    net.train()
    running_loss = 0.0

    for epoch in range(epochs):
        epoch_loss = 0.0
        for batch in tqdm(trainloader, desc=f"Epoch {epoch + 1}/{epochs}"):
            x = batch["x"].to(device)
            y = batch["y"].to(device)

            optimizer.zero_grad()
            outputs = net(x)
            loss = criterion(outputs, y)
            loss.backward()

            # Apply gradient clipping if enabled
            if grad_clipper is not None:
                grad_clipper(net)

            optimizer.step()

            epoch_loss += loss.item()

        epoch_loss /= len(trainloader)
        running_loss += epoch_loss
        print(f"  Epoch {epoch + 1}/{epochs} - Loss: {epoch_loss:.4f}")

    avg_loss = running_loss / epochs
    return avg_loss


def test(net, testloader, device):
    """Evaluate the model on the test set (binary classification).

    Returns:
        avg_loss: Average BCE loss
        tp: True Positives
        tn: True Negatives
        fp: False Positives
        fn: False Negatives
        all_probs: Array of prediction probabilities (for ROC-AUC)
        all_labels: Array of true labels (for ROC-AUC)
    """
    net.to(device)
    # Use BCE for evaluation (consistent metric)
    criterion = torch.nn.BCEWithLogitsLoss()
    net.eval()
    total_loss = 0.0

    all_probs = []
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for batch in testloader:
            x = batch["x"].to(device)
            y = batch["y"].to(device)
            outputs = net(x)
            loss = criterion(outputs, y)
            total_loss += loss.item()

            # Get predictions and probabilities
            probs = torch.sigmoid(outputs)
            predictions = (probs > 0.5).float()

            # Store for metric calculation
            all_probs.append(probs.cpu().numpy())
            all_predictions.append(predictions.cpu().numpy())
            all_labels.append(y.cpu().numpy())

    avg_loss = total_loss / len(testloader)

    # Flatten arrays
    all_probs = np.concatenate(all_probs).flatten()
    all_predictions = np.concatenate(all_predictions).flatten()
    all_labels = np.concatenate(all_labels).flatten()

    # Calculate confusion matrix components
    tp = int(np.sum((all_predictions == 1) & (all_labels == 1)))
    tn = int(np.sum((all_predictions == 0) & (all_labels == 0)))
    fp = int(np.sum((all_predictions == 1) & (all_labels == 0)))
    fn = int(np.sum((all_predictions == 0) & (all_labels == 1)))

    return avg_loss, tp, tn, fp, fn, all_probs, all_labels


def compute_metrics_from_confusion_matrix(tp, tn, fp, fn):
    """Compute classification metrics from confusion matrix components."""
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    f1 = (
        2 * (precision * sensitivity) / (precision + sensitivity)
        if (precision + sensitivity) > 0
        else 0.0
    )
    return {
        "sensitivity": sensitivity,
        "specificity": specificity,
        "precision": precision,
        "f1": f1,
    }


# ============================================================================
# USAGE EXAMPLE
# ============================================================================
if __name__ == "__main__":
    """
    Example of how to use the improved training pipeline.

    To customize, modify the CONFIG dictionary at the top of this file.
    """
    print("=" * 80)
    print("IMPROVED FEDERATED LEARNING PIPELINE")
    print("=" * 80)
    print("\nConfiguration:")
    for key, value in CONFIG.items():
        print(f"  {key}: {value}")
    print("=" * 80)

    # Create model
    model = Net()
    print(f"\nModel created: {CONFIG['model_architecture']}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Example: Load data with advanced sampling
    # (Requires DATASET_DIR environment variable)
    # trainloader = load_data("HospitalA", "train", image_size=128, batch_size=16)
    # evalloader = load_data("HospitalA", "eval", image_size=128, batch_size=32)

    print("\n✅ Setup complete! Ready for training.")
    print("\nTo use this in your federated learning setup:")
    print("1. Update client_app.py to import from this file")
    print("2. Adjust CONFIG dictionary for your needs")
    print("3. Run federated training as usual")

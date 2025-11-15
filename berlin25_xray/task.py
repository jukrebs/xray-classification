"""berlin25-xray: A Flower / PyTorch app for federated X-ray classification."""

import os
from typing import Optional
import contextlib

import numpy as np
import torch
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_from_disk
from torch.utils.data import DataLoader
from torchvision import models
from torchvision.transforms import Compose, Grayscale, Normalize, Resize, ToTensor

PARTITION_HOSPITAL_MAP = {
    0: "A",
    1: "B",
    2: "C",
}

hospital_datasets = {}  # Cache loaded hospital datasets


def _get_amp_policy(device: torch.device):
    """Return (use_amp, amp_dtype, use_grad_scaler) for the given device.

    On modern GPUs (including AMD MI300X via ROCm), prefer bfloat16 for
    stability and performance and disable gradient scaling in that case.
    """
    if device.type != "cuda":
        return False, None, False

    use_bf16 = False
    if hasattr(torch.cuda, "is_bf16_supported"):
        try:
            use_bf16 = torch.cuda.is_bf16_supported()
        except Exception:
            use_bf16 = False

    if use_bf16:
        amp_dtype = torch.bfloat16
        use_grad_scaler = False
    else:
        amp_dtype = torch.float16
        use_grad_scaler = True

    return True, amp_dtype, use_grad_scaler


def _autocast(device: torch.device, enabled: bool, dtype):
    """Return an autocast context manager for the given device/dtype."""
    if not enabled:
        return contextlib.nullcontext()

    if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
        return torch.amp.autocast(device_type=device.type, dtype=dtype)

    # Fallback to legacy CUDA AMP API
    return torch.cuda.amp.autocast(dtype=dtype, enabled=enabled)


class DualHeadDenseNetClassifier(nn.Module):
    """Two independent linear heads on top of DenseNet features, averaged at logit level."""

    def __init__(self, in_features: int):
        super().__init__()
        self.head1 = nn.Linear(in_features, 1)
        self.head2 = nn.Linear(in_features, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logit1 = self.head1(x)
        logit2 = self.head2(x)
        return 0.5 * (logit1 + logit2)


class Net(nn.Module):
    """DenseNet-121 baseline with ImageNet initialization and partial fine-tuning."""

    def __init__(self, image_size: int = 224):
        super().__init__()
        self.target_size = image_size
        weights = models.DenseNet121_Weights.IMAGENET1K_V1
        backbone = models.densenet121(weights=weights)

        conv0 = backbone.features.conv0
        new_conv = nn.Conv2d(
            1,
            conv0.out_channels,
            kernel_size=conv0.kernel_size,
            stride=conv0.stride,
            padding=conv0.padding,
            bias=False,
        )
        with torch.no_grad():
            new_conv.weight.copy_(conv0.weight.mean(dim=1, keepdim=True))
        backbone.features.conv0 = new_conv

        num_features = backbone.classifier.in_features
        backbone.classifier = DualHeadDenseNetClassifier(num_features)
        self.model = backbone

        for param in self.model.parameters():
            param.requires_grad = False
        for module in [
            self.model.features.denseblock4,
            self.model.features.norm5,
            self.model.classifier,
        ]:
            for param in module.parameters():
                param.requires_grad = True

    def _prepare_input(self, x: torch.Tensor) -> torch.Tensor:
        x = x.float()
        if x.min() < 0 or x.max() > 1:
            x = x * 0.5 + 0.5
        if tuple(x.shape[-2:]) != (self.target_size, self.target_size):
            x = F.interpolate(
                x,
                size=(self.target_size, self.target_size),
                mode="bilinear",
                align_corners=False,
            )
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._prepare_input(x)
        return self.model(x)


def load_data(
    dataset_name: str,
    split_name: str,
    image_size: int = 224,
    batch_size: int = 32,
):
    """Load hospital X-ray data.

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
        split = full_dataset[split_name]
        # Use efficient tensor formatting for PyTorch DataLoader
        split.set_format(type="torch", columns=["x", "y"])
        hospital_datasets[cache_key] = split
        print(f"Loaded {dataset_path}/{split_name}")

    data = hospital_datasets[cache_key]
    shuffle = split_name == "train"  # shuffle only for training splits
    multiprocessing_ctx = mp.get_context("spawn")
    dataloader = DataLoader(
        data,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=2,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
        multiprocessing_context=multiprocessing_ctx,
    )
    return dataloader


class LabelSmoothingBCELoss(nn.Module):
    """Binary cross-entropy with light label smoothing to improve calibration."""

    def __init__(self, smoothing: float = 0.05):
        super().__init__()
        self.smoothing = float(smoothing)
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        smooth = self.smoothing
        targets_smoothed = targets * (1.0 - smooth) + 0.5 * smooth
        return self.bce(inputs, targets_smoothed)


def train(net, trainloader, epochs, lr, device):
    net.to(device)
    criterion = LabelSmoothingBCELoss(smoothing=0.05).to(device)
    optimizer = torch.optim.AdamW(
        (p for p in net.parameters() if p.requires_grad), lr=lr, weight_decay=0.01
    )
    net.train()
    running_loss = 0.0
    use_amp, amp_dtype, use_grad_scaler = _get_amp_policy(device)
    scaler = torch.cuda.amp.GradScaler(enabled=use_grad_scaler)
    for _ in range(epochs):
        for batch in trainloader:
            x = batch["x"].to(device, non_blocking=True).float()
            y = batch["y"].to(device, non_blocking=True).float()
            optimizer.zero_grad()
            with _autocast(device, enabled=use_amp, dtype=amp_dtype):
                outputs = net(x)
                loss = criterion(outputs, y)
            if use_grad_scaler:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
            running_loss += loss.item()
    avg_loss = running_loss / (len(trainloader) * epochs)
    return avg_loss


def test(net, testloader, device, max_batches: Optional[int] = None):
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
    criterion = torch.nn.BCEWithLogitsLoss()
    net.eval()
    total_loss = 0.0

    all_probs = []
    all_predictions = []
    all_labels = []
    use_amp, amp_dtype, _ = _get_amp_policy(device)
    with torch.no_grad():
        for batch_idx, batch in enumerate(testloader):
            if max_batches is not None and batch_idx >= max_batches:
                break
            x = batch["x"].to(device, non_blocking=True).float()
            y = batch["y"].to(device, non_blocking=True).float()
            with _autocast(device, enabled=use_amp, dtype=amp_dtype):
                outputs = net(x)
                loss = criterion(outputs, y)
            total_loss += loss.item()

            # Get predictions and probabilities
            probs = torch.sigmoid(outputs)
            predictions = (probs > 0.5).float()

            # Store for metric calculation
            all_probs.append(probs.float().cpu().numpy())
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


def apply_transforms(batch, image_size):
    """For reference: This is the apply_transforms we used for image preprocessing."""
    result = {}

    _transform_pipeline = Compose(
        [
            Resize((image_size, image_size)),
            Grayscale(num_output_channels=1),
            ToTensor(),
            Normalize(mean=[0.5], std=[0.5]),  # Normalize for grayscale
        ]
    )

    # Transform images and stack them into a tensor
    transformed_images = [_transform_pipeline(img) for img in batch["image"]]
    result["x"] = torch.stack(transformed_images)

    # Binary classification: 0 for "No Finding", 1 for any finding
    labels = []
    for label_list in batch["label"]:
        # If "No Finding" is the only label, it's 0; otherwise it's 1
        has_finding = not (len(label_list) == 1 and label_list[0] == "No Finding")
        labels.append(torch.tensor([float(has_finding)]))
    result["y"] = torch.stack(labels)

    return result


# For reference: These are all labels in the original dataset.
# In the challenge we only consider a binary classification: (no) finding.
LABELS = [
    "No Finding",
    "Atelectasis",
    "Cardiomegaly",
    "Effusion",
    "Infiltration",
    "Mass",
    "Nodule",
    "Pneumonia",
    "Pneumothorax",
    "Consolidation",
    "Edema",
    "Emphysema",
    "Fibrosis",
    "Pleural_Thickening",
    "Hernia",
]

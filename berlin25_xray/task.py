"""berlin25-xray: A Flower / PyTorch app for federated X-ray classification."""

import os
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_from_disk
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Grayscale, Normalize, Resize, ToTensor
from tqdm import tqdm
from transformers import AutoImageProcessor, AutoModel

PARTITION_HOSPITAL_MAP = {
    0: "A",
    1: "B",
    2: "C",
}

hospital_datasets = {}  # Cache loaded hospital datasets


class Net(nn.Module):
    """CXformer-based encoder with lightweight binary classification head."""

    def __init__(self, model_name: Optional[str] = None, trust_remote_code: Optional[bool] = None):
        super().__init__()
        self.model_name = model_name or os.environ.get(
            "XRAY_BACKBONE", "m42-health/CXformer-base"
        )
        if trust_remote_code is None:
            env_flag = os.environ.get("XRAY_TRUST_REMOTE_CODE", "true").lower()
            trust_remote_code = env_flag in {"1", "true", "yes"}
        self.image_processor = AutoImageProcessor.from_pretrained(
            self.model_name, trust_remote_code=trust_remote_code
        )
        self.encoder = AutoModel.from_pretrained(
            self.model_name, trust_remote_code=trust_remote_code
        )
        for param in self.encoder.parameters():
            param.requires_grad = False

        hidden_size = getattr(self.encoder.config, "hidden_size", None) or getattr(
            self.encoder.config, "embed_dim", None
        )
        if hidden_size is None:
            raise ValueError(
                "Unable to determine hidden size for CXformer encoder configuration."
            )

        head_hidden = max(512, hidden_size // 2)
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Dropout(p=0.1),
            nn.Linear(hidden_size, head_hidden),
            nn.GELU(),
            nn.Dropout(p=0.1),
            nn.Linear(head_hidden, 1),
        )

        size_cfg = self.image_processor.size
        if isinstance(size_cfg, dict):
            height = size_cfg.get("height") or size_cfg.get("shortest_edge") or 518
            width = size_cfg.get("width") or size_cfg.get("shortest_edge") or height
        elif isinstance(size_cfg, (list, tuple)):
            height, width = size_cfg
        else:
            height = width = int(size_cfg)
        self.target_size = (int(height), int(width))

        mean = torch.tensor(
            self.image_processor.image_mean, dtype=torch.float32
        ).view(1, -1, 1, 1)
        std = torch.tensor(
            self.image_processor.image_std, dtype=torch.float32
        ).view(1, -1, 1, 1)
        if mean.shape[1] == 1:
            mean = mean.repeat(1, 3, 1, 1)
            std = std.repeat(1, 3, 1, 1)
        self.register_buffer("processor_mean", mean)
        self.register_buffer("processor_std", std)

    def _prepare_pixel_values(self, x: torch.Tensor) -> torch.Tensor:
        x = x.float()
        # Undo previous grayscale normalization if needed (data stored as [-1, 1]).
        x_detached = x.detach()
        if (x_detached.min() < 0) or (x_detached.max() > 1):
            x = x * 0.5 + 0.5
        x = torch.clamp(x, 0.0, 1.0)
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        if tuple(x.shape[-2:]) != self.target_size:
            x = F.interpolate(
                x,
                size=self.target_size,
                mode="bicubic",
                align_corners=False,
            )
        return (x - self.processor_mean) / self.processor_std

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pixel_values = self._prepare_pixel_values(x)
        outputs = self.encoder(pixel_values=pixel_values)
        if getattr(outputs, "pooler_output", None) is not None:
            features = outputs.pooler_output
        else:
            features = outputs.last_hidden_state[:, 0]
        return self.classifier(features)


def collate_preprocessed(batch):
    """Collate function for preprocessed data: Convert list of dicts to dict of batched tensors."""
    result = {}
    for key in batch[0].keys():
        if key in ["x", "y"]:
            # Convert lists back to tensors and stack
            result[key] = torch.stack(
                [torch.tensor(item[key], dtype=torch.float32) for item in batch]
            )
        else:
            # Keep other fields as lists
            result[key] = [item[key] for item in batch]
    return result


def load_data(
    dataset_name: str,
    split_name: str,
    image_size: int = 224,
    batch_size: int = 1024,
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
        hospital_datasets[cache_key] = full_dataset[split_name]
        print(f"Loaded {dataset_path}/{split_name}")

    data = hospital_datasets[cache_key]
    shuffle = split_name == "train"  # shuffle only for training splits
    dataloader = DataLoader(
        data,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=4,
        collate_fn=collate_preprocessed,
    )
    return dataloader


def train(net, trainloader, epochs, lr, device):
    net.to(device)
    criterion = torch.nn.BCEWithLogitsLoss().to(device)
    optimizer = torch.optim.AdamW(
        (p for p in net.parameters() if p.requires_grad),
        lr=lr,
        weight_decay=0.01,
    )
    net.train()
    running_loss = 0.0
    for _ in range(epochs):
        for batch in tqdm(trainloader):
            x = batch["x"].to(device, non_blocking=True).float()
            y = batch["y"].to(device, non_blocking=True).float()
            optimizer.zero_grad()
            outputs = net(x)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
    avg_loss = running_loss / (len(trainloader) * epochs)
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
    criterion = torch.nn.BCEWithLogitsLoss()
    net.eval()
    total_loss = 0.0

    all_probs = []
    all_predictions = []
    all_labels = []
    with torch.no_grad():
        for batch in testloader:
            x = batch["x"].to(device, non_blocking=True).float()
            y = batch["y"].to(device, non_blocking=True).float()
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

"""berlin25-xray: A Flower / PyTorch app for federated X-ray classification."""

import os
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_from_disk
from huggingface_hub import hf_hub_download
from torch.utils.data import DataLoader
from torchvision import models
from torchvision.transforms import Compose, Grayscale, Normalize, Resize, ToTensor
from tqdm import tqdm

PARTITION_HOSPITAL_MAP = {
    0: "A",
    1: "B",
    2: "C",
}

hospital_datasets = {}  # Cache loaded hospital datasets


class Net(nn.Module):
    """DenseNet-121 initialized from TorchXRayVision weights with binary head."""

    def __init__(self, model_name: Optional[str] = None):
        super().__init__()
        self.repo_id = model_name or os.environ.get(
            "XRAY_DENSENET_MODEL", "torchxrayvision/densenet121-res224-chex"
        )
        self.target_size = int(os.environ.get("XRAY_DENSENET_SIZE", 224))

        self.encoder = models.densenet121(weights=None)
        self.encoder.features.conv0 = nn.Conv2d(
            1, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        num_features = self.encoder.classifier.in_features
        self.encoder.classifier = nn.Linear(num_features, 1)

        self._load_checkpoint()

        self.register_buffer(
            "processor_mean", torch.tensor([0.5], dtype=torch.float32).view(1, 1, 1, 1)
        )
        self.register_buffer(
            "processor_std", torch.tensor([0.5], dtype=torch.float32).view(1, 1, 1, 1)
        )

    def _load_checkpoint(self):
        filename = os.environ.get("XRAY_DENSENET_FILENAME", "pytorch_model.bin")
        path = hf_hub_download(repo_id=self.repo_id, filename=filename)
        state_dict = torch.load(path, map_location="cpu")

        conv_key = "features.conv0.weight"
        if conv_key in state_dict and state_dict[conv_key].shape[1] == 3:
            weight = state_dict[conv_key].mean(dim=1, keepdim=True)
            state_dict[conv_key] = weight

        head_key = "classifier.weight"
        if head_key in state_dict and state_dict[head_key].shape[0] != 1:
            state_dict.pop("classifier.weight")
            state_dict.pop("classifier.bias")

        self.encoder.load_state_dict(state_dict, strict=False)

    def _prepare_pixel_values(self, x: torch.Tensor) -> torch.Tensor:
        x = x.float()
        x_detached = x.detach()
        if (x_detached.min() < 0) or (x_detached.max() > 1):
            x = x * 0.5 + 0.5
        x = torch.clamp(x, 0.0, 1.0)
        if x.shape[1] != 1:
            x = x.mean(dim=1, keepdim=True)
        if tuple(x.shape[-2:]) != (self.target_size, self.target_size):
            x = F.interpolate(
                x,
                size=(self.target_size, self.target_size),
                mode="bicubic",
                align_corners=False,
            )
        return (x - self.processor_mean) / self.processor_std

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pixel_values = self._prepare_pixel_values(x)
        return self.encoder(pixel_values)


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
    batch_size: int = 128,
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
    optimizer = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=0.01)
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

"""berlin25-xray: A Flower / PyTorch app for federated X-ray classification."""

import os

import numpy as np
import torch
import torch.nn as nn
from datasets import load_from_disk
try:
    from torch import amp  # PyTorch >= 2.0 preferred API
except ImportError:  # pragma: no cover - fallback for older runtimes
    from torch.cuda import amp  # type: ignore
from torch.utils.data import DataLoader
from torchvision.models import ViT_B_16_Weights, vit_b_16
from torchvision.transforms import Compose, Grayscale, Normalize, Resize, ToTensor
from tqdm import tqdm

PARTITION_HOSPITAL_MAP = {
    0: "A",
    1: "B",
    2: "C",
}

hospital_datasets = {}  # Cache loaded hospital datasets


class Net(nn.Module):
    """ViT-B/16-based model for binary chest X-ray classification (pretrained)."""

    def __init__(self):
        super(Net, self).__init__()

        weights = ViT_B_16_Weights.IMAGENET1K_V1

        # Load ImageNet-pretrained ViT-B/16 and replace classifier head with single logit
        self.vit = vit_b_16(weights=weights)
        in_features = self.vit.heads.head.in_features
        self.vit.heads.head = nn.Linear(in_features, 1)

        # Freeze all parameters except the new head for a head-only finetune
        for name, param in self.vit.named_parameters():
            param.requires_grad = False
        for param in self.vit.heads.head.parameters():
            param.requires_grad = True

        # Store ImageNet normalization stats so inputs match the pretrained backbone
        # Fallback to standard ImageNet stats if torchvision doesn't expose them via weights.meta
        meta = getattr(weights, "meta", {}) or {}
        mean_vals = meta.get("mean", (0.485, 0.456, 0.406))
        std_vals = meta.get("std", (0.229, 0.224, 0.225))
        mean = torch.tensor(mean_vals).view(1, 3, 1, 1)
        std = torch.tensor(std_vals).view(1, 3, 1, 1)
        self.register_buffer("imagenet_mean", mean, persistent=False)
        self.register_buffer("imagenet_std", std, persistent=False)

    def forward(self, x):
        # ViT expects 3-channel inputs; repeat grayscale channel if needed
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)

        # Undo preprocessing (x was normalized with mean=0.5, std=0.5) and match ViT stats
        x = x * 0.5 + 0.5  # Back to [0, 1]
        x = (x - self.imagenet_mean) / self.imagenet_std

        return self.vit(x)


def collate_preprocessed(batch):
    """Collate function for preprocessed data: Convert list of dicts to dict of batched tensors."""
    result = {}
    tensor_keys = {"x", "y"}
    for key in batch[0].keys():
        if key in tensor_keys:
            first_value = batch[0][key]
            if torch.is_tensor(first_value):
                result[key] = torch.stack([item[key] for item in batch])
            else:
                arr = np.asarray([item[key] for item in batch], dtype=np.float32)
                result[key] = torch.from_numpy(arr)
        else:
            # Keep other fields as lists
            result[key] = [item[key] for item in batch]
    return result


def load_data(
    dataset_name: str,
    split_name: str,
    image_size: int = 128,
    batch_size: int = 16,
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
    loader_kwargs = dict(
        dataset=data,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=4,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_preprocessed,
    )
    if loader_kwargs["num_workers"] > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = 4
    dataloader = DataLoader(**loader_kwargs)
    return dataloader


def train(net, trainloader, epochs, lr, device):
    net.to(device)
    criterion = torch.nn.BCEWithLogitsLoss().to(device)
    params = (p for p in net.parameters() if p.requires_grad)
    optimizer = torch.optim.Adam(params, lr=lr)
    device_type = device.type
    use_amp = device_type == "cuda"
    try:
        scaler = amp.GradScaler(device_type=device_type, enabled=use_amp)
    except TypeError:
        try:
            scaler = amp.GradScaler(device=device_type, enabled=use_amp)
        except TypeError:
            scaler = amp.GradScaler(enabled=use_amp)
    net.train()
    running_loss = 0.0
    total_steps = len(trainloader) * max(epochs, 1)
    for _ in range(epochs):
        for batch in tqdm(trainloader):
            x = batch["x"].to(device)
            y = batch["y"].to(device)
            optimizer.zero_grad(set_to_none=True)
            try:
                autocast_ctx = amp.autocast(device_type=device_type, enabled=use_amp)
            except TypeError:
                try:
                    autocast_ctx = amp.autocast(device=device_type, enabled=use_amp)
                except TypeError:
                    autocast_ctx = amp.autocast(enabled=use_amp)
            with autocast_ctx:
                outputs = net(x)
                loss = criterion(outputs, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running_loss += loss.item()
    avg_loss = running_loss / total_steps if total_steps > 0 else 0.0
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

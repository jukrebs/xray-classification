"""berlin25-xray: A Flower / PyTorch app for federated X-ray classification."""

import logging
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from datasets import Dataset, concatenate_datasets, load_from_disk
from transformers import AutoModel

try:
    from torch import amp  # PyTorch >= 2.0 preferred API
except ImportError:  # pragma: no cover - fallback for older runtimes
    from torch.cuda import amp  # type: ignore
from torch.utils.data import DataLoader
import torchvision.transforms.functional as TF
from torchvision.transforms import (
    Compose,
    Grayscale,
    InterpolationMode,
    Normalize,
    Resize,
    ToTensor,
)
from tqdm import tqdm

from berlin25_xray.logging_utils import (
    configure_logging,
    log_gpu_utilization,
    log_timing,
)

DATASET_ENV_VAR = "DATASET_DIR"
DEFAULT_IMAGE_SIZE = 128
DEFAULT_BATCH_SIZE = 16
DEFAULT_EVAL_BATCH_SIZE = 32

# CXformer tuning hyperparameters
# Unfreeze and finetune the last N transformer blocks.
CXFORMER_TUNE_LAST_N_LAYERS = 8
# Use a moderately higher LR for the (partially) unfrozen backbone.
CXFORMER_BACKBONE_LR_SCALE = 0.3  # backbone LR = head LR * scale

PARTITION_HOSPITAL_MAP = {
    0: "A",
    1: "B",
    2: "C",
}

hospital_datasets = {}  # Cache loaded hospital datasets

configure_logging()
logger = logging.getLogger(__name__)


class Net(nn.Module):
    """CXformer-base backbone (X-ray pretrained) + small head ensemble."""

    def __init__(self, image_size: int = DEFAULT_IMAGE_SIZE, num_heads: int = 8):
        super(Net, self).__init__()

        self.model_name = "m42-health/CXformer-base"
        self.image_size = image_size
        self.num_heads = max(int(num_heads), 1)

        # Load CXformer backbone (DINOv2-with-registers variant), pretrained on chest X-rays.
        self.backbone = AutoModel.from_pretrained(self.model_name)

        # Freeze backbone parameters by default.
        for p in self.backbone.parameters():
            p.requires_grad = False

        # Optionally unfreeze the last few transformer blocks for light finetuning.
        encoder = getattr(self.backbone, "encoder", None)
        layers = getattr(encoder, "layer", None) if encoder is not None else None
        if isinstance(layers, torch.nn.ModuleList) and layers:
            n_layers = len(layers)
            n_tune = min(CXFORMER_TUNE_LAST_N_LAYERS, n_layers)
            for idx in range(n_layers - n_tune, n_layers):
                for p in layers[idx].parameters():
                    p.requires_grad = True

        # Hidden size comes from the config (typically 768).
        hidden_dim = self.backbone.config.hidden_size

        # Small ensemble of binary MLP heads: any finding vs no finding.
        self.heads = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim // 2),
                    nn.GELU(),
                    nn.Dropout(p=0.2),
                    nn.Linear(hidden_dim // 2, 1),
                )
                for _ in range(self.num_heads)
            ]
        )

        # CXformer uses ImageNet-style normalization.
        mean_vals = (0.485, 0.456, 0.406)
        std_vals = (0.229, 0.224, 0.225)
        mean = torch.tensor(mean_vals).view(1, 3, 1, 1)
        std = torch.tensor(std_vals).view(1, 3, 1, 1)
        self.register_buffer("cx_mean", mean, persistent=False)
        self.register_buffer("cx_std", std, persistent=False)

    def forward(self, x):
        # x: [B, 1, H, W], already preprocessed:
        #    - Resize to image-size (128 or 224) in the offline pipeline
        #    - Grayscale
        #    - Normalize(mean=[0.5], std=[0.5]) → ~[-1, 1]

        # Map back to [0, 1].
        x = (x + 1.0) / 2.0  # [-1,1] -> [0,1]

        # CXformer expects 3-channel input; repeat grayscale channel.
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)  # [B, 3, H, W]

        # Match CXformer training normalization.
        x = (x - self.cx_mean) / self.cx_std

        # Forward through backbone (DINOv2-style, uses 'pixel_values' as input key).
        outputs = self.backbone(pixel_values=x)

        # Use the model's pooled representation (handles registers/cls internally).
        pooled = outputs.pooler_output  # [B, hidden_dim]

        # Forward through all heads and average logits: [B, 1, num_heads] -> [B, 1]
        head_logits = [head(pooled) for head in self.heads]  # list of [B, 1]
        logits = torch.stack(head_logits, dim=-1).mean(dim=-1)

        return logits  # BCEWithLogitsLoss expects raw logits


def _build_optimizer(model: nn.Module, lr: float):
    """Create an optimizer with param groups when using CXformer backbone."""

    # Special handling for CXformer-based Net: separate head and backbone params.
    if isinstance(model, Net):
        head_params = list(model.heads.parameters())
        backbone_params = [p for p in model.backbone.parameters() if p.requires_grad]

        param_groups = []
        if head_params:
            param_groups.append({"params": head_params, "lr": lr})
        if backbone_params:
            param_groups.append(
                {
                    "params": backbone_params,
                    "lr": lr * CXFORMER_BACKBONE_LR_SCALE,
                }
            )
        return torch.optim.AdamW(param_groups, weight_decay=0.01)

    # Fallback: single parameter group for any other model.
    params = (p for p in model.parameters() if p.requires_grad)
    return torch.optim.AdamW(params, lr=lr, weight_decay=0.01)


def _compute_pos_weight(trainloader, device):
    """Estimate positive class weight for BCE loss from the training data."""

    try:
        ys = trainloader.dataset["y"]
        labels = torch.as_tensor(ys).view(-1)
        num_pos = (labels >= 0.5).sum().item()
        num_neg = (labels < 0.5).sum().item()
        if num_pos == 0:
            pos_weight_val = 1.0
        else:
            pos_weight_val = num_neg / num_pos
        logger.info(
            "Computed pos_weight for BCE: %.4f (pos=%d, neg=%d)",
            pos_weight_val,
            num_pos,
            num_neg,
        )
        return torch.tensor([pos_weight_val], device=device)
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("Falling back to pos_weight=1.0 due to error: %s", exc)
        return torch.tensor([1.0], device=device)


def _augment_batch(x: torch.Tensor) -> torch.Tensor:
    """On-the-fly GPU-friendly augmentation for chest X-ray tensors in [-1, 1]."""

    # Random horizontal flip
    if torch.rand(1, device=x.device) < 0.5:
        x = torch.flip(x, dims=[3])

    # Small random rotation (±5 degrees)
    if torch.rand(1, device=x.device) < 0.3:
        angle = float(torch.empty(1, device=x.device).uniform_(-5.0, 5.0))
        x = TF.rotate(
            x,
            angle=angle,
            interpolation=InterpolationMode.BILINEAR,
            fill=0.0,
        )

    return x


def maybe_compile_model(model: nn.Module, *, mode: str, enabled: bool = True) -> nn.Module:
    """Best-effort torch.compile wrapper to squeeze more throughput out of ViT."""

    if not enabled:
        return model

    compile_fn = getattr(torch, "compile", None)
    if compile_fn is None:
        logger.info("torch.compile unavailable; running %s model in eager mode", mode)
        return model

    try:
        compiled_model = compile_fn(model)
    except Exception as exc:  # pragma: no cover - backend specific
        logger.warning(
            "torch.compile failed for %s (falling back to eager): %s",
            mode,
            exc,
        )
        return model

    logger.info("Enabled torch.compile for %s", mode)
    return compiled_model


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


def _load_split_from_arrow(dataset_path: str, split_name: str):
    """Load a dataset split directly from Arrow shards, bypassing HF metadata."""
    split_dir = Path(dataset_path) / split_name
    if not split_dir.exists():
        raise FileNotFoundError(
            f"Split directory {split_dir} is missing. Ensure DATASET_DIR is correct."
        )

    arrow_files = sorted(split_dir.glob("*.arrow"))
    if not arrow_files:
        raise FileNotFoundError(
            f"No Arrow files found in {split_dir}. Dataset export could be corrupted."
        )

    datasets = [Dataset.from_file(str(arrow_file)) for arrow_file in arrow_files]
    if len(datasets) == 1:
        return datasets[0]
    return concatenate_datasets(datasets)


def load_data(
    dataset_name: str,
    split_name: str,
    image_size: int = DEFAULT_IMAGE_SIZE,
    batch_size: int = DEFAULT_BATCH_SIZE,
):
    """Load hospital X-ray data.

    Args:
        dataset_name: Dataset name ("HospitalA", "HospitalB", "HospitalC")
        split_name: Split name ("train", "eval")
        image_size: Image size (128 or 224)
        batch_size: Number of samples per batch
    """
    logger.info(
        "Preparing dataloader | dataset=%s | split=%s | image_size=%d | batch_size=%d",
        dataset_name,
        split_name,
        image_size,
        batch_size,
    )
    dataset_dir = os.environ[DATASET_ENV_VAR]

    # Use preprocessed dataset based on image_size
    cache_key = f"{dataset_name}_{split_name}_{image_size}"
    dataset_path = (
        f"{dataset_dir}/xray_fl_datasets_preprocessed_{image_size}/{dataset_name}"
    )

    with log_timing(
        logger, f"Dataloader preparation for {dataset_name}/{split_name}"
    ):
        # Load and cache dataset
        global hospital_datasets
        cache_hit = cache_key in hospital_datasets
        if not cache_hit:
            logger.info(
                "Cache miss for %s/%s. Loading dataset from %s",
                dataset_name,
                split_name,
                dataset_path,
            )
            try:
                full_dataset = load_from_disk(dataset_path)
                split_dataset = full_dataset[split_name]
            except (TypeError, ValueError) as err:
                logger.warning(
                    "load_from_disk failed for %s/%s (%s). Falling back to Arrow loader.",
                    dataset_name,
                    split_name,
                    err,
                )
                split_dataset = _load_split_from_arrow(dataset_path, split_name)
            hospital_datasets[cache_key] = split_dataset
        else:
            logger.info("Cache hit for %s/%s", dataset_name, split_name)

        data = hospital_datasets[cache_key]
        num_examples = len(data)
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
        num_batches = len(dataloader)
        logger.info(
            "Dataloader ready | dataset=%s/%s | samples=%d | batches=%d | workers=%d | shuffle=%s",
            dataset_name,
            split_name,
            num_examples,
            num_batches,
            loader_kwargs["num_workers"],
            shuffle,
        )
    return dataloader


def dataset_name_from_partition(partition_id: int) -> str:
    """Map a Flower partition id to the corresponding hospital dataset name."""

    try:
        hospital = PARTITION_HOSPITAL_MAP[int(partition_id)]
    except (KeyError, TypeError, ValueError) as exc:
        raise KeyError(f"Unknown partition id: {partition_id}") from exc
    return f"Hospital{hospital}"


def train(net, trainloader, epochs, lr, device):
    net.to(device)
    pos_weight = _compute_pos_weight(trainloader, device)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight).to(device)
    optimizer = _build_optimizer(net, lr)
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
    num_batches = len(trainloader)
    num_examples = len(trainloader.dataset)
    total_steps = num_batches * max(epochs, 1)
    logger.info(
        "Starting training | device=%s | epochs=%d | batches=%d | samples=%d | lr=%s | amp=%s",
        device,
        epochs,
        num_batches,
        num_examples,
        lr,
        use_amp,
    )
    log_gpu_utilization(logger, device, prefix="Train/start")
    train_start = time.perf_counter()
    for epoch in range(epochs):
        epoch_loss = 0.0
        epoch_start = time.perf_counter()
        progress = tqdm(trainloader, desc=f"Epoch {epoch + 1}/{epochs}")
        for batch in progress:
            x = batch["x"].to(device)
            y = batch["y"].to(device)
            x = _augment_batch(x)
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
            epoch_loss += loss.item()
        epoch_duration = time.perf_counter() - epoch_start
        avg_epoch_loss = epoch_loss / num_batches if num_batches > 0 else 0.0
        logger.info(
            "Epoch %d/%d finished in %.2fs | avg loss %.4f",
            epoch + 1,
            epochs,
            epoch_duration,
            avg_epoch_loss,
        )
        log_gpu_utilization(logger, device, prefix=f"Train/epoch{epoch + 1}")
    total_duration = time.perf_counter() - train_start
    avg_loss = running_loss / total_steps if total_steps > 0 else 0.0
    logger.info(
        "Training complete in %.2fs | avg loss %.4f",
        total_duration,
        avg_loss,
    )
    log_gpu_utilization(logger, device, prefix="Train/end")
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
    num_batches = len(testloader)
    num_examples = len(testloader.dataset)
    logger.info(
        "Starting evaluation | device=%s | batches=%d | samples=%d",
        device,
        num_batches,
        num_examples,
    )
    log_gpu_utilization(logger, device, prefix="Eval/start")

    all_probs = []
    all_predictions = []
    all_labels = []
    eval_start = time.perf_counter()
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

    elapsed = time.perf_counter() - eval_start
    avg_loss = total_loss / len(testloader)
    logger.info(
        "Evaluation complete in %.2fs | avg loss %.4f",
        elapsed,
        avg_loss,
    )
    log_gpu_utilization(logger, device, prefix="Eval/end")

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

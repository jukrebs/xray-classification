"""berlin25-xray: A Flower / PyTorch app for federated X-ray classification."""

import logging
import multiprocessing as mp
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from datasets import Dataset, concatenate_datasets, load_from_disk
try:
    from torch import amp  # PyTorch >= 2.0 preferred API
except ImportError:  # pragma: no cover - fallback for older runtimes
    from torch.cuda import amp  # type: ignore
from torch.utils.data import DataLoader
from torchvision.models import DenseNet121_Weights, densenet121
from torchvision.transforms import Compose, Grayscale, Normalize, Resize, ToTensor
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

PARTITION_HOSPITAL_MAP = {
    0: "A",
    1: "B",
    2: "C",
}

hospital_datasets = {}  # Cache loaded hospital datasets

configure_logging()
logger = logging.getLogger(__name__)
IS_ROCM = bool(getattr(torch.version, "hip", None))


def _configure_torch_backends():
    """Enable backend knobs that speed up conv-heavy models without hurting accuracy."""

    try:
        torch.set_float32_matmul_precision("medium")
    except Exception:  # pragma: no cover - torch version dependent
        pass
    try:
        torch.backends.cudnn.benchmark = True
    except Exception:  # pragma: no cover - backend might be missing
        pass
    try:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    except Exception:  # pragma: no cover - CPU only or old torch
        pass


_configure_torch_backends()


class Net(nn.Module):
    """DenseNet121 backbone with a lightweight X-ray specific classification head."""

    def __init__(
        self,
        image_size: int = 224,
        head_hidden_dim: int = 256,
        head_dropout: float = 0.2,
    ):
        super().__init__()

        self.image_size = image_size
        weights = DenseNet121_Weights.IMAGENET1K_V1
        backbone = densenet121(weights=weights)

        # Adapt the first convolution to consume single-channel X-ray inputs by
        # averaging the RGB kernels provided by the pretrained checkpoint.
        conv0 = backbone.features.conv0
        backbone.features.conv0 = nn.Conv2d(
            1,
            conv0.out_channels,
            kernel_size=conv0.kernel_size,
            stride=conv0.stride,
            padding=conv0.padding,
            bias=False,
        )
        with torch.no_grad():
            backbone.features.conv0.weight.copy_(conv0.weight.mean(dim=1, keepdim=True))

        feature_dim = backbone.classifier.in_features
        backbone.classifier = nn.Identity()
        self.backbone = backbone
        if head_hidden_dim and head_hidden_dim > 0:
            self.head = nn.Sequential(
                nn.Dropout(head_dropout),
                nn.Linear(feature_dim, head_hidden_dim),
                nn.GELU(),
                nn.Dropout(head_dropout),
                nn.Linear(head_hidden_dim, 1),
            )
        else:
            self.head = nn.Linear(feature_dim, 1)

        for param in self.backbone.parameters():
            param.requires_grad = False
        for param in self.head.parameters():
            param.requires_grad = True

        meta = getattr(weights, "meta", {}) or {}
        mean_vals = meta.get("mean", (0.485, 0.456, 0.406))
        std_vals = meta.get("std", (0.229, 0.224, 0.225))
        grayscale_mean = float(sum(mean_vals) / len(mean_vals))
        grayscale_std = float(sum(std_vals) / len(std_vals))
        mean = torch.tensor(grayscale_mean).view(1, 1, 1, 1)
        std = torch.tensor(grayscale_std).view(1, 1, 1, 1)
        self.register_buffer("imagenet_mean", mean, persistent=False)
        self.register_buffer("imagenet_std", std, persistent=False)

    def forward(self, x):
        if x.shape[1] == 3:
            x = x.mean(dim=1, keepdim=True)

        # Undo preprocessing (x was normalized with mean=0.5, std=0.5) and match ImageNet stats
        x = x * 0.5 + 0.5  # Back to [0, 1]
        x = (x - self.imagenet_mean) / self.imagenet_std

        features = self.backbone(x)
        return self.head(features)


def maybe_compile_model(model: nn.Module, *, mode: str, enabled: bool = True) -> nn.Module:
    """Best-effort torch.compile wrapper to squeeze more throughput out of the model."""

    if not enabled:
        return model
    if IS_ROCM:
        logger.info("torch.compile skipped for %s on ROCm (compile warmup too costly)", mode)
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


def _suggest_num_workers():
    """Pick a DataLoader worker count that scales on both laptops and servers."""

    try:
        affinity = os.sched_getaffinity(0)
        cpu_count = len(affinity)
    except AttributeError:
        cpu_count = os.cpu_count() or 1
    if cpu_count <= 2:
        return 0  # small environments prefer main-thread loading
    return min(8, max(2, cpu_count // 2))


def _supports_channels_last(device: torch.device) -> bool:
    return device.type == "cuda" and not IS_ROCM


def prepare_model_for_device(model: nn.Module, device: torch.device) -> nn.Module:
    """Move a model to the target device using channel-last layout when beneficial."""

    if _supports_channels_last(device):
        return model.to(device=device, memory_format=torch.channels_last)
    return model.to(device=device)


def _move_batch_to_device(batch, device: torch.device):
    """Copy batch tensors to device asynchronously and keep channel-last layout on CUDA."""

    non_blocking = device.type == "cuda"
    x = batch["x"].to(device=device, non_blocking=non_blocking)
    y = batch["y"].to(device=device, non_blocking=non_blocking)
    if _supports_channels_last(device):
        x = x.to(memory_format=torch.channels_last)
    return x, y


def _compute_class_statistics(dataset):
    """Return basic class statistics for BCE loss balancing."""
    try:
        labels_column = dataset["y"]
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("Unable to access labels for class stats computation: %s", exc)
        return None

    if not labels_column:
        return None

    flattened = []
    for label in labels_column:
        if isinstance(label, torch.Tensor):
            flattened.append(float(label.squeeze().item()))
        elif isinstance(label, (list, tuple)):
            if not label:
                continue
            flattened.append(float(label[0]))
        elif isinstance(label, np.ndarray):
            if label.size == 0:
                continue
            flattened.append(float(label.flat[0]))
        else:
            flattened.append(float(label))

    total = len(flattened)
    if total == 0:
        return None

    positives = float(np.sum(flattened))
    negatives = float(total) - positives
    pos_fraction = positives / float(total)
    if positives == 0.0:
        pos_weight = None
    else:
        pos_weight = negatives / positives if negatives > 0 else None
    return {
        "pos_fraction": pos_fraction,
        "pos_weight": pos_weight,
        "num_examples": total,
        "num_positives": positives,
    }


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
        num_workers = _suggest_num_workers()
        pin_memory = torch.cuda.is_available() and not IS_ROCM
        loader_kwargs = dict(
            dataset=data,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            collate_fn=collate_preprocessed,
        )
        if loader_kwargs["num_workers"] > 0:
            loader_kwargs["persistent_workers"] = True
            prefetch_factor = max(2, min(8, max(1, batch_size // 32)))
            loader_kwargs["prefetch_factor"] = prefetch_factor
            try:
                loader_kwargs["multiprocessing_context"] = mp.get_context("spawn")
            except ValueError:
                logger.warning("Unable to set spawn multiprocessing context; using default.")
        dataloader = DataLoader(**loader_kwargs)
        class_stats = _compute_class_statistics(data)
        if class_stats:
            setattr(dataloader, "class_stats", class_stats)
            pos_pct = class_stats["pos_fraction"] * 100.0
            pos_weight = class_stats["pos_weight"]
            logger.info(
                "Class distribution | dataset=%s/%s | positives=%.2f%% | pos_weight=%s",
                dataset_name,
                split_name,
                pos_pct,
                f"{pos_weight:.3f}" if pos_weight is not None else "n/a",
            )
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
    net = prepare_model_for_device(net, device)
    class_stats = getattr(trainloader, "class_stats", None)
    pos_weight_value = None
    if isinstance(class_stats, dict):
        pos_weight_value = class_stats.get("pos_weight")
        if pos_weight_value is not None and pos_weight_value <= 0:
            pos_weight_value = None
    if pos_weight_value is not None:
        pos_weight_tensor = torch.tensor([pos_weight_value], device=device)
        criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor).to(device)
        logger.info(
            "Using class-balanced BCEWithLogitsLoss | pos_weight=%.3f",
            pos_weight_value,
        )
    else:
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
            x, y = _move_batch_to_device(batch, device)
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
    net = prepare_model_for_device(net, device)
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
            x, y = _move_batch_to_device(batch, device)
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

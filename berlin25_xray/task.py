"""berlin25-xray: A Flower / PyTorch app for federated X-ray classification."""

import logging
import math
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import Dataset, concatenate_datasets, load_from_disk
try:
    from torch import amp  # PyTorch >= 2.0 preferred API
except ImportError:  # pragma: no cover - fallback for older runtimes
    from torch.cuda import amp  # type: ignore
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision.models import ViT_B_16_Weights, vit_b_16
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

# ViT tuning hyperparameters
VIT_TUNE_LAST_N_LAYERS = 0
VIT_BACKBONE_LR_SCALE = 0.1  # backbone LR = head LR * scale

PARTITION_HOSPITAL_MAP = {
    0: "A",
    1: "B",
    2: "C",
}

hospital_datasets = {}  # Cache loaded hospital datasets

configure_logging()
logger = logging.getLogger(__name__)

# Cache for ViT weights/state_dicts to avoid repeated, expensive loads
_VIT_WEIGHTS = ViT_B_16_Weights.IMAGENET1K_V1
_VIT_BASE_STATE_DICT = None
_VIT_RESIZED_STATE_DICT_CACHE = {}


class Net(nn.Module):
    """ViT-B/16-based model for binary chest X-ray classification (pretrained)."""

    def __init__(self, image_size: int = 224):
        super(Net, self).__init__()

        self.image_size = image_size

        # Build the ViT backbone at the requested resolution
        self.vit = vit_b_16(weights=None, image_size=image_size)

        # Lazily load and cache the pretrained state_dict, then cache the
        # resized positional embeddings per image_size. This avoids paying
        # the cost of loading/resizing ViT weights on every Net() creation.
        global _VIT_BASE_STATE_DICT, _VIT_RESIZED_STATE_DICT_CACHE
        if _VIT_BASE_STATE_DICT is None:
            _VIT_BASE_STATE_DICT = _VIT_WEIGHTS.get_state_dict(progress=False)

        cache_key = int(image_size)
        cached_state = _VIT_RESIZED_STATE_DICT_CACHE.get(cache_key)
        if cached_state is None:
            state_dict = dict(_VIT_BASE_STATE_DICT)
            state_dict = _resize_vit_positional_embeddings(state_dict, self.vit, image_size)
            state_dict.pop("heads.head.weight", None)
            state_dict.pop("heads.head.bias", None)
            _VIT_RESIZED_STATE_DICT_CACHE[cache_key] = state_dict
            cached_state = state_dict

        missing, unexpected = self.vit.load_state_dict(cached_state, strict=False)
        if unexpected:
            logger.debug("Unexpected ViT weights skipped: %s", unexpected)
        if missing:
            logger.debug("Missing ViT weights (expected due to head swap): %s", missing)

        in_features = self.vit.heads.head.in_features
        self.vit.heads.head = nn.Sequential(
            nn.LayerNorm(in_features),
            nn.Linear(in_features, 512),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(512, 1),
        )

        # Freeze all parameters by default.
        for param in self.vit.parameters():
            param.requires_grad = False

        # Unfreeze the classification head.
        for param in self.vit.heads.head.parameters():
            param.requires_grad = True

        # Optionally unfreeze the last few transformer blocks for light finetuning.
        encoder = getattr(self.vit, "encoder", None)
        layer_container = None
        for attr in ("layers", "layer", "blocks"):
            if encoder is not None and hasattr(encoder, attr):
                candidate = getattr(encoder, attr)
                if isinstance(candidate, torch.nn.ModuleList) and len(candidate) > 0:
                    layer_container = candidate
                    break
        if layer_container is not None:
            n_layers = len(layer_container)
            n_tune = min(VIT_TUNE_LAST_N_LAYERS, n_layers)
            for idx in range(n_layers - n_tune, n_layers):
                for p in layer_container[idx].parameters():
                    p.requires_grad = True

        # Store ImageNet normalization stats so inputs match the pretrained backbone.
        # Fallback to standard ImageNet stats if torchvision doesn't expose them via weights.meta.
        meta = getattr(_VIT_WEIGHTS, "meta", {}) or {}
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


def _resize_vit_positional_embeddings(state_dict, vit_model, target_image_size: int):
    """Resize pretrained ViT positional embeddings when changing image resolution."""

    pos_key = "encoder.pos_embedding"
    if pos_key not in state_dict:
        return state_dict

    pretrained_pos = state_dict[pos_key]
    current_pos = vit_model.encoder.pos_embedding
    if pretrained_pos.shape == current_pos.shape:
        return state_dict

    num_patches = current_pos.shape[1] - 1
    new_size = int(math.sqrt(num_patches))
    cls_token = pretrained_pos[:, :1]
    patch_tokens = pretrained_pos[:, 1:]
    old_num_patches = patch_tokens.shape[1]
    old_size = int(math.sqrt(old_num_patches))

    logger.info(
        "Resizing ViT positional embeddings from %dx%d to %dx%d for image_size=%d",
        old_size,
        old_size,
        new_size,
        new_size,
        target_image_size,
    )

    patch_tokens = patch_tokens.reshape(1, old_size, old_size, -1).permute(0, 3, 1, 2)
    patch_tokens = F.interpolate(
        patch_tokens,
        size=(new_size, new_size),
        mode="bicubic",
        align_corners=False,
    )
    patch_tokens = patch_tokens.permute(0, 2, 3, 1).reshape(1, new_size * new_size, -1)
    state_dict[pos_key] = torch.cat([cls_token, patch_tokens], dim=1)
    return state_dict


def maybe_compile_model(model: nn.Module, *, mode: str, enabled: bool = True) -> nn.Module:
    """Best-effort torch.compile wrapper to squeeze more throughput out of ViT."""

    if not enabled:
        return model

    compile_fn = getattr(torch, "compile", None)
    if compile_fn is None:
        logger.info("torch.compile unavailable; running %s model in eager mode", mode)
        return model

    try:
        try:
            # Prefer reduce-overhead mode for shorter training runs
            compiled_model = compile_fn(model, mode="reduce-overhead")
        except TypeError:
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
    balance: bool = False,
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
        sampler = None

        # Optional balanced sampling for training to mitigate class imbalance.
        if split_name == "train" and balance:
            ys = np.asarray(data["y"], dtype=np.float32).reshape(-1)
            num_pos = float((ys >= 0.5).sum())
            num_neg = float((ys < 0.5).sum())
            if num_pos == 0 or num_neg == 0:
                logger.warning(
                    "Skipping balanced sampler due to degenerate class distribution (pos=%s, neg=%s)",
                    num_pos,
                    num_neg,
                )
            else:
                w_pos = 1.0 / num_pos
                w_neg = 1.0 / num_neg
                weights = np.where(ys >= 0.5, w_pos, w_neg)
                sampler = WeightedRandomSampler(
                    torch.as_tensor(weights, dtype=torch.double),
                    num_samples=len(weights),
                    replacement=True,
                )
                shuffle = False  # sampler and shuffle cannot be used together
        loader_kwargs = dict(
            dataset=data,
            batch_size=batch_size,
            shuffle=shuffle,
            sampler=sampler,
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


def _make_warmup_cosine(optimizer, warmup_steps: int, total_steps: int):
    """Create a lightweight warmup+cosine LR scheduler."""

    def lr_lambda(step: int) -> float:
        if total_steps <= 0:
            return 1.0
        if step < warmup_steps:
            return float(step + 1) / max(1, warmup_steps)
        progress = float(step - warmup_steps) / max(1, total_steps - warmup_steps)
        progress = min(max(progress, 0.0), 1.0)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def dataset_name_from_partition(partition_id: int) -> str:
    """Map a Flower partition id to the corresponding hospital dataset name."""

    try:
        hospital = PARTITION_HOSPITAL_MAP[int(partition_id)]
    except (KeyError, TypeError, ValueError) as exc:
        raise KeyError(f"Unknown partition id: {partition_id}") from exc
    return f"Hospital{hospital}"


def _compute_pos_weight_from_loader(trainloader, device):
    """Compute pos_weight for BCE from the label distribution in the loader dataset."""

    try:
        ys = np.asarray(trainloader.dataset["y"], dtype=np.float32).reshape(-1)
        num_pos = float((ys >= 0.5).sum())
        num_neg = float((ys < 0.5).sum())
    except Exception as exc:  # pragma: no cover - defensive, dataset-specific
        logger.warning(
            "Could not read labels from trainloader.dataset for pos_weight computation: %s",
            exc,
        )
        return None

    if num_pos == 0 or num_neg == 0:
        logger.warning(
            "Skipping pos_weight computation due to degenerate class distribution (pos=%s, neg=%s)",
            num_pos,
            num_neg,
        )
        return None

    pos_weight_value = num_neg / num_pos
    logger.info(
        "Using BCEWithLogitsLoss with pos_weight=%.3f (pos=%d, neg=%d)",
        pos_weight_value,
        int(num_pos),
        int(num_neg),
    )
    return torch.tensor([pos_weight_value], device=device)


def train(net, trainloader, epochs, lr, device):
    net.to(device)
    pos_weight = _compute_pos_weight_from_loader(trainloader, device)
    if pos_weight is not None:
        criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight).to(device)
    else:
        criterion = torch.nn.BCEWithLogitsLoss().to(device)
    if isinstance(net, Net):
        head_params = list(net.vit.heads.head.parameters())
        head_param_ids = {id(p) for p in head_params}
        backbone_params = [
            p for p in net.parameters() if p.requires_grad and id(p) not in head_param_ids
        ]

        def _split_decay(params):
            decay = []
            no_decay = []
            for p in params:
                if not p.requires_grad:
                    continue
                if p.ndim >= 2:
                    decay.append(p)
                else:
                    no_decay.append(p)
            return decay, no_decay

        head_decay, head_no_decay = _split_decay(head_params)
        backbone_decay, backbone_no_decay = _split_decay(backbone_params)
        param_groups = []
        if head_decay:
            param_groups.append(
                {
                    "params": head_decay,
                    "lr": lr,
                    "weight_decay": 0.05,
                }
            )
        if head_no_decay:
            param_groups.append(
                {
                    "params": head_no_decay,
                    "lr": lr,
                    "weight_decay": 0.0,
                }
            )
        if backbone_decay:
            param_groups.append(
                {
                    "params": backbone_decay,
                    "lr": lr * VIT_BACKBONE_LR_SCALE,
                    "weight_decay": 0.05,
                }
            )
        if backbone_no_decay:
            param_groups.append(
                {
                    "params": backbone_no_decay,
                    "lr": lr * VIT_BACKBONE_LR_SCALE,
                    "weight_decay": 0.0,
                }
            )
        optimizer = torch.optim.AdamW(param_groups)

        # Exponential moving average (EMA) of head parameters for more stable evaluation
        ema_decay = 0.99
        ema_head_params = [p.detach().clone() for p in head_params]
    else:
        params = (p for p in net.parameters() if p.requires_grad)
        optimizer = torch.optim.AdamW(
            params,
            lr=lr,
            weight_decay=0.01,
        )
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
    warmup_steps = max(10, int(0.1 * total_steps)) if total_steps > 0 else 0
    scheduler = _make_warmup_cosine(optimizer, warmup_steps, total_steps) if total_steps > 0 else None
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

            # Lightweight train-time augmentation to improve generalization
            if net.training:
                # Random horizontal flip (consistent with evaluation TTA)
                if torch.rand(1).item() < 0.5:
                    x = torch.flip(x, dims=[3])
                # Global brightness/contrast jitter in normalized [-1, 1] space
                bc_prob = 0.8
                if torch.rand(1).item() < bc_prob:
                    # Per-sample scalar brightness/contrast factors
                    brightness_delta = 0.1
                    contrast_delta = 0.1
                    b = (
                        (torch.rand(x.size(0), 1, 1, 1, device=x.device) - 0.5)
                        * 2.0
                        * brightness_delta
                    )
                    c = 1.0 + (
                        (torch.rand(x.size(0), 1, 1, 1, device=x.device) - 0.5)
                        * 2.0
                        * contrast_delta
                    )
                    x = (x * c + b).clamp(-1.0, 1.0)
                # Small Gaussian noise in normalized space
                noise_std = 0.01
                if noise_std > 0.0:
                    x = x + noise_std * torch.randn_like(x)
                    x = x.clamp(-1.0, 1.0)

                # Mixup augmentation (cheap, improves ranking/AUROC)
                mixup_alpha = 0.2
                if mixup_alpha > 0.0 and x.size(0) > 1:
                    lam = float(np.random.beta(mixup_alpha, mixup_alpha))
                    perm = torch.randperm(x.size(0), device=device)
                    x_shuffled = x[perm]
                    y_shuffled = y[perm]
                    x = lam * x + (1.0 - lam) * x_shuffled
                    y_a, y_b, mixup_lam = y, y_shuffled, lam
                else:
                    y_a = y_b = None
                    mixup_lam = None

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
                if mixup_lam is not None and y_a is not None and y_b is not None:
                    loss = mixup_lam * criterion(outputs, y_a) + (1.0 - mixup_lam) * criterion(
                        outputs, y_b
                    )
                else:
                    loss = criterion(outputs, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            if isinstance(net, Net):
                with torch.no_grad():
                    for ema_p, p in zip(ema_head_params, head_params):
                        ema_p.mul_(ema_decay).add_(p, alpha=1.0 - ema_decay)
            if scheduler is not None:
                scheduler.step()
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

    # Swap in EMA head weights for downstream evaluation/aggregation
    if isinstance(net, Net):
        with torch.no_grad():
            for p, ema_p in zip(head_params, ema_head_params):
                p.data.copy_(ema_p.data)

    total_duration = time.perf_counter() - train_start
    avg_loss = running_loss / total_steps if total_steps > 0 else 0.0
    logger.info(
        "Training complete in %.2fs | avg loss %.4f",
        total_duration,
        avg_loss,
    )
    log_gpu_utilization(logger, device, prefix="Train/end")
    return avg_loss


def test(net, testloader, device, use_tta: bool = False):
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
    device_type = device.type
    use_amp = device_type == "cuda"
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
        try:
            autocast_ctx = amp.autocast(device_type=device_type, enabled=use_amp)
        except TypeError:
            try:
                autocast_ctx = amp.autocast(device=device_type, enabled=use_amp)
            except TypeError:
                autocast_ctx = amp.autocast(enabled=use_amp)

        for batch in testloader:
            x = batch["x"].to(device)
            y = batch["y"].to(device)
            with autocast_ctx:
                outputs = net(x)
                loss = criterion(outputs, y)
            total_loss += loss.item()

            # Get predictions and probabilities
            probs = torch.sigmoid(outputs)

            if use_tta:
                # Simple test-time augmentation: horizontal flip (averaged prediction)
                x_flip = torch.flip(x, dims=[3])
                with autocast_ctx:
                    outputs_flip = net(x_flip)
                probs_flip = torch.sigmoid(outputs_flip)
                probs = 0.5 * (probs + probs_flip)

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

import os
import functools

import numpy as np
import torch
import torch.nn as nn
from datasets import load_from_disk
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoImageProcessor, AutoModel

hospital_datasets = {}  # Cache loaded hospital datasets

DEFAULT_MODEL_NAME = os.environ.get("CXFORMER_MODEL_NAME", "m42-health/CXformer-base")
FREEZE_ENCODER = os.environ.get("CXFORMER_FREEZE_ENCODER", "1").lower() not in {"0", "false", "no"}
CLASSIFIER_DIM = int(os.environ.get("CXFORMER_CLASSIFIER_DIM", "512"))
CLASSIFIER_DROPOUT = float(os.environ.get("CXFORMER_CLASSIFIER_DROPOUT", "0.2"))


@functools.lru_cache(maxsize=1)
def get_image_processor():
    """Load and cache the CXformer image processor once per process."""
    return AutoImageProcessor.from_pretrained(DEFAULT_MODEL_NAME, trust_remote_code=True)


def _tensor_to_rgb_np(tensor: torch.Tensor) -> np.ndarray:
    """Convert a tensor in CHW or HW format to float32 RGB numpy array."""
    array = tensor.detach().cpu().numpy()
    if array.ndim == 3:
        array = np.moveaxis(array, 0, -1)
    elif array.ndim == 2:
        array = array[..., None]
    if array.shape[-1] == 1:
        array = np.repeat(array, 3, axis=-1)
    return array.astype(np.float32, copy=False)


def prepare_model_inputs(batch_x: torch.Tensor, device: torch.device) -> torch.Tensor:
    """Convert grayscale tensors to CXformer pixel values."""
    processor = get_image_processor()
    images = [_tensor_to_rgb_np(img) for img in batch_x]
    processed = processor(images=images, return_tensors="pt")
    return processed["pixel_values"].to(device)


class Net(nn.Module):
    """CXformer encoder plus a lightweight binary classification head."""

    def __init__(self):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(DEFAULT_MODEL_NAME, trust_remote_code=True)
        hidden_size = self.encoder.config.hidden_size
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, CLASSIFIER_DIM),
            nn.GELU(),
            nn.Dropout(CLASSIFIER_DROPOUT),
            nn.Linear(CLASSIFIER_DIM, 1),
        )
        if FREEZE_ENCODER:
            for param in self.encoder.parameters():
                param.requires_grad = False

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        outputs = self.encoder(pixel_values=pixel_values)
        cls_token = outputs.last_hidden_state[:, 0]
        return self.classifier(cls_token)


def collate_preprocessed(batch):
    """Collate function for preprocessed data: Convert list of dicts to dict of batched tensors."""
    result = {}
    for key in batch[0].keys():
        if key in ["x", "y"]:
            # Convert lists to tensors and stack
            result[key] = torch.stack([torch.as_tensor(item[key], dtype=torch.float32) for item in batch])
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
    dataset_path = f"{dataset_dir}/xray_fl_datasets_preprocessed_{image_size}/{dataset_name}"

    # Load and cache dataset
    global hospital_datasets
    if cache_key not in hospital_datasets:
        full_dataset = load_from_disk(dataset_path)
        hospital_datasets[cache_key] = full_dataset[split_name]
        print(f"Loaded {dataset_path}/{split_name}")

    data = hospital_datasets[cache_key]
    shuffle = (split_name == "train")  # shuffle only for training splits
    dataloader = DataLoader(data, batch_size=batch_size, shuffle=shuffle, num_workers=4, collate_fn=collate_preprocessed)
    return dataloader


def train(net, trainloader, epochs, lr, device):
    net.to(device)
    criterion = torch.nn.BCEWithLogitsLoss().to(device)
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, weight_decay=1e-2)
    net.train()
    running_loss = 0.0
    for _ in range(epochs):
        for batch in tqdm(trainloader):
            pixel_values = prepare_model_inputs(batch["x"], device)
            y = batch["y"].float().to(device)
            optimizer.zero_grad()
            outputs = net(pixel_values)
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
        all_probs: Array of prediction probabilities (for AUROC)
        all_labels: Array of true labels (for AUROC)
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
            pixel_values = prepare_model_inputs(batch["x"], device)
            y = batch["y"].float().to(device)
            outputs = net(pixel_values)
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

"""Local training script to sanity-check the ViT-B/16 baseline."""

import logging
from dataclasses import dataclass

import torch
from sklearn.metrics import roc_auc_score

from berlin25_xray.logging_utils import configure_logging, log_gpu_utilization
from berlin25_xray.task import (
    Net,
    compute_metrics_from_confusion_matrix,
    load_data,
    maybe_compile_model,
)
from berlin25_xray.task import test as test_fn
from berlin25_xray.task import train as train_fn


@dataclass(frozen=True)
class LocalTrainingConfig:
    """Local fine-tuning defaults used by local_train.py."""

    hospital: str = "A"
    epochs: int = 3
    learning_rate: float = 3e-4
    batch_size: int = 1024
    image_size: int = 128
    compile_model: bool = False

    @property
    def dataset_name(self) -> str:
        hospital_key = self.hospital.strip().upper()
        if len(hospital_key) != 1 or hospital_key < "A" or hospital_key > "Z":
            raise ValueError(f"Invalid hospital identifier: {self.hospital!r}")
        return f"Hospital{hospital_key}"


TRAINING_CONFIG = LocalTrainingConfig()


configure_logging()
logger = logging.getLogger(__name__)


def main():
    """Train X-ray classifier locally on a specific hospital's data."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(
        (
            "Starting local training | hospital=%s | epochs=%d | lr=%.0e | "
            "batch=%d | image=%d | device=%s | compile=%s"
        ),
        TRAINING_CONFIG.hospital,
        TRAINING_CONFIG.epochs,
        TRAINING_CONFIG.learning_rate,
        TRAINING_CONFIG.batch_size,
        TRAINING_CONFIG.image_size,
        device,
        TRAINING_CONFIG.compile_model,
    )
    log_gpu_utilization(logger, device, prefix="Local/device")

    dataset_name = TRAINING_CONFIG.dataset_name
    logger.info(
        "Loading %s data at %dx%d resolution...",
        dataset_name,
        TRAINING_CONFIG.image_size,
        TRAINING_CONFIG.image_size,
    )
    trainloader = load_data(
        dataset_name,
        "train",
        image_size=TRAINING_CONFIG.image_size,
        batch_size=TRAINING_CONFIG.batch_size,
        balance=True,
    )
    evalloader = load_data(
        dataset_name,
        "eval",
        image_size=TRAINING_CONFIG.image_size,
        batch_size=TRAINING_CONFIG.batch_size,
    )
    logger.info(
        "Train batches: %d (%d samples) | Eval batches: %d (%d samples)",
        len(trainloader),
        len(trainloader.dataset),
        len(evalloader),
        len(evalloader.dataset),
    )

    model = Net(image_size=TRAINING_CONFIG.image_size).to(device)
    model = maybe_compile_model(
        model,
        mode="local",
        enabled=TRAINING_CONFIG.compile_model,
    )
    logger.info(
        "Starting model fine-tuning for %d epochs on %s",
        TRAINING_CONFIG.epochs,
        device,
    )
    for epoch in range(1, TRAINING_CONFIG.epochs + 1):
        logger.info("[Epoch %d/%d] Training...", epoch, TRAINING_CONFIG.epochs)
        train_loss = train_fn(
            model,
            trainloader,
            epochs=1,
            lr=TRAINING_CONFIG.learning_rate,
            device=device,
        )
        logger.info(
            "[Epoch %d/%d] Train loss: %.4f",
            epoch,
            TRAINING_CONFIG.epochs,
            train_loss,
        )

        logger.info("[Epoch %d/%d] Evaluating...", epoch, TRAINING_CONFIG.epochs)
        eval_loss, tp, tn, fp, fn, probs, labels = test_fn(
            model,
            evalloader,
            device,
        )
        auroc = roc_auc_score(labels, probs)
        cm_metrics = compute_metrics_from_confusion_matrix(tp, tn, fp, fn)
        logger.info(
            "[Epoch %d/%d] Eval loss: %.4f | AUROC: %.4f",
            epoch,
            TRAINING_CONFIG.epochs,
            eval_loss,
            auroc,
        )
        logger.info(
            "[Epoch %d/%d] Sensitivity: %.4f | Specificity: %.4f | Precision: %.4f | F1: %.4f",
            epoch,
            TRAINING_CONFIG.epochs,
            cm_metrics["sensitivity"],
            cm_metrics["specificity"],
            cm_metrics["precision"],
            cm_metrics["f1"],
        )

    log_gpu_utilization(logger, device, prefix="Local/done")
    model_path = f"hospital_{TRAINING_CONFIG.hospital}_model.pt"
    torch.save(model.state_dict(), model_path)
    logger.info("Training complete! Model saved to %s", model_path)


if __name__ == "__main__":
    main()

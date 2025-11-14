"""Local training script to sanity-check the ViT-B/16 baseline."""

import logging

import torch
from sklearn.metrics import roc_auc_score

from berlin25_xray.logging_utils import configure_logging, log_gpu_utilization, log_timing
from berlin25_xray.task import (
    Net,
    compute_metrics_from_confusion_matrix,
    load_data,
    maybe_compile_model,
)
from berlin25_xray.task import test as test_fn
from berlin25_xray.task import train as train_fn

HOSPITAL = "A"  # A, B, or C
EPOCHS = 3
LEARNING_RATE = 1e-4
BATCH_SIZE = 1024
IMAGE_SIZE = 128
COMPILE_MODEL = True


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
        HOSPITAL,
        EPOCHS,
        LEARNING_RATE,
        BATCH_SIZE,
        IMAGE_SIZE,
        device,
        COMPILE_MODEL,
    )
    log_gpu_utilization(logger, device, prefix="Local/device")

    dataset_name = f"Hospital{HOSPITAL}"
    logger.info(
        "Loading %s data at %dx%d resolution...",
        dataset_name,
        IMAGE_SIZE,
        IMAGE_SIZE,
    )
    with log_timing(logger, f"Train dataloader ({dataset_name})"):
        trainloader = load_data(
            dataset_name,
            "train",
            image_size=IMAGE_SIZE,
            batch_size=BATCH_SIZE,
        )
    with log_timing(logger, f"Eval dataloader ({dataset_name})"):
        evalloader = load_data(
            dataset_name,
            "eval",
            image_size=IMAGE_SIZE,
            batch_size=BATCH_SIZE,
        )
    logger.info(
        "Train batches: %d (%d samples) | Eval batches: %d (%d samples)",
        len(trainloader),
        len(trainloader.dataset),
        len(evalloader),
        len(evalloader.dataset),
    )
    log_gpu_utilization(logger, device, prefix="Local/post-dataload")

    model = Net().to(device)
    model = maybe_compile_model(model, mode="local", enabled=COMPILE_MODEL)
    logger.info(
        "Starting model fine-tuning for %d epochs on %s",
        EPOCHS,
        device,
    )
    for epoch in range(1, EPOCHS + 1):
        logger.info("[Epoch %d/%d] Training...", epoch, EPOCHS)
        with log_timing(logger, f"Local epoch {epoch} training"):
            train_loss = train_fn(
                model,
                trainloader,
                epochs=1,
                lr=LEARNING_RATE,
                device=device,
            )
        logger.info("[Epoch %d/%d] Train loss: %.4f", epoch, EPOCHS, train_loss)
        log_gpu_utilization(logger, device, prefix=f"Local/epoch{epoch}-train")

        logger.info("[Epoch %d/%d] Evaluating...", epoch, EPOCHS)
        with log_timing(logger, f"Local epoch {epoch} eval"):
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
            EPOCHS,
            eval_loss,
            auroc,
        )
        logger.info(
            "[Epoch %d/%d] Sensitivity: %.4f | Specificity: %.4f | Precision: %.4f | F1: %.4f",
            epoch,
            EPOCHS,
            cm_metrics["sensitivity"],
            cm_metrics["specificity"],
            cm_metrics["precision"],
            cm_metrics["f1"],
        )
        log_gpu_utilization(logger, device, prefix=f"Local/epoch{epoch}-eval")

    model_path = f"hospital_{HOSPITAL}_model.pt"
    torch.save(model.state_dict(), model_path)
    logger.info("Training complete! Model saved to %s", model_path)


if __name__ == "__main__":
    main()

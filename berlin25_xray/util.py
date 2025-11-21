"""Helper functions for W&B logging and metric computation.

These functions should not be changed by hackathon participants, unless you want
to log additional metrics or save models based on different criteria.
"""

import os
import warnings

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from berlin25_xray.task import (
    PARTITION_HOSPITAL_MAP,
    compute_metrics_from_confusion_matrix,
)

# Suppress protobuf deprecation warnings
warnings.filterwarnings("ignore", category=DeprecationWarning, module="google.protobuf")


def compute_metrics(reply_metrics):
    """Compute ROC-AUC and confusion matrix metrics."""
    probs = np.array(reply_metrics["probs"])
    labels = np.array(reply_metrics["labels"])
    roc_auc = roc_auc_score(labels, probs)
    cm_metrics = compute_metrics_from_confusion_matrix(
        reply_metrics["tp"],
        reply_metrics["tn"],
        reply_metrics["fp"],
        reply_metrics["fn"],
    )
    return {"roc_auc": roc_auc, "eval_loss": reply_metrics["eval_loss"], **cm_metrics}


def compute_aggregated_metrics(replies):
    """Compute aggregated metrics across all hospitals."""
    all_probs = [p for r in replies for p in r.content["metrics"]["probs"]]
    all_labels = [l for r in replies for l in r.content["metrics"]["labels"]]
    roc_auc = roc_auc_score(np.array(all_labels), np.array(all_probs))

    tp = sum(r.content["metrics"]["tp"] for r in replies)
    tn = sum(r.content["metrics"]["tn"] for r in replies)
    fp = sum(r.content["metrics"]["fp"] for r in replies)
    fn = sum(r.content["metrics"]["fn"] for r in replies)
    cm_metrics = compute_metrics_from_confusion_matrix(tp, tn, fp, fn)

    return {"roc_auc": roc_auc, **cm_metrics}


def log_training_metrics(replies):
    """Log training metrics to W&B."""
    log_dict = {}
    for reply in replies:
        hospital = f"Hospital{PARTITION_HOSPITAL_MAP[reply.content['metrics']['partition-id']]}"
        log_dict[f"{hospital}/train_loss"] = reply.content["metrics"]["train_loss"]


def log_eval_metrics(replies, agg_metrics, weighted_by_key, log_fn):
    """Log evaluation metrics to console and W&B."""
    log_fn("EVALUATION METRICS BY HOSPITAL")
    log_dict = {}

    for reply in replies:
        hospital = f"Hospital{PARTITION_HOSPITAL_MAP[reply.content['metrics']['partition-id']]}"
        metrics = compute_metrics(reply.content["metrics"])
        n = reply.content["metrics"].get(weighted_by_key, 0)

        log_fn(f"  {hospital} (n={n}):")
        for k, v in metrics.items():
            log_fn(f"  {k:12s}: {v:.4f}")
            log_dict[f"{hospital}/{k}"] = v

    log_fn("AGGREGATED METRICS:")
    for k, v in agg_metrics.items():
        log_fn(f"  {k:12s}: {v:.4f}")
        log_dict[f"Global/{k}"] = v


def save_best_model(
    arrays, agg_metrics, server_round, best_auroc_tracker, model_dir=None
):
    """Save model artifact if it's the best so far.

    Returns updated best_auroc_tracker dict with 'auroc' key.
    """
    current_auroc = agg_metrics["roc_auc"]

    if (
        best_auroc_tracker.get("auroc") is None
        or current_auroc > best_auroc_tracker["auroc"]
    ):
        best_auroc_tracker["auroc"] = current_auroc

        auroc_str = f"{int(current_auroc * 10000):04d}"
        state_dict_file = os.path.join(model_dir, f"run_{auroc_str}.pt")
        torch.save(arrays.to_torch_state_dict(), state_dict_file)

        return (
            True,
            f"New best model! Round {server_round}, AUROC: {current_auroc:.4f}, saved to {state_dict_file}",
        )
    else:
        return (
            False,
            f"  Model not saved (AUROC {current_auroc:.4f} ≤ best {best_auroc_tracker['auroc']:.4f})",
        )

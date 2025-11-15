"""Helper functions for W&B logging and metric computation.

These functions should not be changed by hackathon participants, unless you want
to log additional metrics or save models based on different criteria.
"""

import logging
import math
import os
import re
import warnings
from collections.abc import Mapping

import numpy as np
import torch
import wandb
from sklearn.metrics import roc_auc_score

from berlin25_xray.task import (
    compute_metrics_from_confusion_matrix,
    dataset_name_from_partition,
)

# Suppress protobuf deprecation warnings
warnings.filterwarnings("ignore", category=DeprecationWarning, module="google.protobuf")

logger = logging.getLogger(__name__)


def _sanitize_artifact_name(name: str) -> str:
    """Return a W&B-safe artifact name (alnum, dash, underscore, dot)."""

    if not isinstance(name, str):
        name = str(name)
    # Replace any forbidden character with a dash
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", name)


def _iter_metric_replies(replies, stage: str):
    """Yield (reply, metrics_dict) pairs for replies that contain metrics."""
    for reply in replies:
        if reply is None:
            logger.warning("Skipping %s reply: entry is None", stage)
            continue
        has_content_fn = getattr(reply, "has_content", None)
        try:
            if callable(has_content_fn) and not has_content_fn():
                logger.warning(
                    "Skipping %s reply: message has no content (node=%s)",
                    stage,
                    getattr(reply, "node_id", "unknown"),
                )
                continue
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning(
                "Skipping %s reply: unable to determine content presence (%s)",
                stage,
                exc,
            )
            continue

        content = getattr(reply, "content", None)
        if not isinstance(content, Mapping):
            logger.warning(
                "Skipping %s reply: unexpected content type %s",
                stage,
                type(content),
            )
            continue

        metrics = content.get("metrics")
        if not isinstance(metrics, Mapping):
            logger.warning("Skipping %s reply: no 'metrics' payload", stage)
            continue

        yield reply, metrics


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
    metrics_payloads = [metrics for _, metrics in _iter_metric_replies(replies, "evaluation")]
    if not metrics_payloads:
        logger.warning("No evaluation replies with metrics; returning NaN aggregates.")
        return {
            "roc_auc": float("nan"),
            "sensitivity": float("nan"),
            "specificity": float("nan"),
            "precision": float("nan"),
            "f1": float("nan"),
        }

    all_probs = np.concatenate(
        [np.asarray(metrics.get("probs", [])) for metrics in metrics_payloads]
    )
    all_labels = np.concatenate(
        [np.asarray(metrics.get("labels", [])) for metrics in metrics_payloads]
    )
    if all_probs.size == 0 or all_labels.size == 0:
        roc_auc = float("nan")
        logger.warning("Evaluation replies lacked probability/label data; ROC-AUC set to NaN.")
    else:
        try:
            roc_auc = roc_auc_score(all_labels, all_probs)
        except ValueError as err:
            logger.warning("Unable to compute ROC-AUC: %s", err)
            roc_auc = float("nan")

    tp = sum(int(metrics.get("tp", 0)) for metrics in metrics_payloads)
    tn = sum(int(metrics.get("tn", 0)) for metrics in metrics_payloads)
    fp = sum(int(metrics.get("fp", 0)) for metrics in metrics_payloads)
    fn = sum(int(metrics.get("fn", 0)) for metrics in metrics_payloads)
    cm_metrics = compute_metrics_from_confusion_matrix(tp, tn, fp, fn)

    return {"roc_auc": roc_auc, **cm_metrics}


def log_training_metrics(replies, server_round):
    """Log training metrics to W&B."""
    log_dict = {}
    for reply, metrics in _iter_metric_replies(replies, "train"):
        partition_id = metrics.get("partition-id")
        train_loss = metrics.get("train_loss")
        if partition_id is None or train_loss is None:
            logger.warning("Skipping train reply missing partition or train_loss metric.")
            continue
        try:
            hospital = dataset_name_from_partition(int(partition_id))
        except (KeyError, TypeError, ValueError):
            logger.warning(
                "Skipping train reply due to unknown partition-id value: %s", partition_id
            )
            continue
        log_dict[f"{hospital}/train_loss"] = train_loss
    if not log_dict:
        logger.warning(
            "No training metrics collected for server round %s; skipping W&B log.",
            server_round,
        )
        return
    wandb.log(log_dict, step=server_round)


def log_eval_metrics(replies, agg_metrics, server_round, weighted_by_key, log_fn):
    """Log evaluation metrics to console and W&B."""
    metric_replies = list(_iter_metric_replies(replies, "evaluation"))
    if not metric_replies:
        log_fn("No evaluation replies with metrics to report.")
        return

    log_fn("EVALUATION METRICS BY HOSPITAL")
    log_dict = {}
    for reply, reply_metrics in metric_replies:
        partition_id = reply_metrics.get("partition-id")
        if partition_id is None:
            logger.warning("Skipping evaluation reply missing partition-id.")
            continue
        try:
            hospital = dataset_name_from_partition(int(partition_id))
        except (KeyError, TypeError, ValueError):
            logger.warning(
                "Skipping evaluation reply due to unknown partition-id value: %s",
                partition_id,
            )
            continue
        metrics = compute_metrics(reply_metrics)
        n = reply_metrics.get(weighted_by_key, 0)

        log_fn(f"  {hospital} (n={n}):")
        for k, v in metrics.items():
            log_fn(f"  {k:12s}: {v:.4f}")
            log_dict[f"{hospital}/{k}"] = v

    log_fn("AGGREGATED METRICS:")
    for k, v in agg_metrics.items():
        log_fn(f"  {k:12s}: {v:.4f}")
        log_dict[f"Global/{k}"] = v

    if not log_dict:
        logger.warning(
            "Evaluation log dictionary is empty for round %s; skipping W&B log.",
            server_round,
        )
        return
    wandb.log(log_dict, step=server_round)


def save_best_model(arrays, agg_metrics, server_round, run_name, best_auroc_tracker):
    """Save model artifact if it's the best so far.

    Returns updated best_auroc_tracker dict with 'auroc' key.
    """
    if arrays is None:
        logger.warning(
            "Skipping model save for round %s because model arrays are unavailable.",
            server_round,
        )
        return False, "  Model not saved (model arrays unavailable)"

    raw_auroc = agg_metrics["roc_auc"]
    try:
        current_auroc = float(raw_auroc)
    except (TypeError, ValueError):
        current_auroc = float("nan")

    if math.isnan(current_auroc):
        logger.warning(
            "Skipping model save for round %s because aggregated AUROC is NaN.",
            server_round,
        )
        return False, "  Model not saved (AUROC unavailable)"

    if (
        best_auroc_tracker.get("auroc") is None
        or current_auroc > best_auroc_tracker["auroc"]
    ):
        best_auroc_tracker["auroc"] = current_auroc

        auroc_str = f"{int(current_auroc * 10000):04d}"
        state_dict_file = os.path.expanduser(
            f"~/model_round{server_round}_auroc{auroc_str}.pt"
        )
        torch.save(arrays.to_torch_state_dict(), state_dict_file)

        metadata = {**agg_metrics, "round": server_round}
        safe_run_name = _sanitize_artifact_name(run_name or "run")
        artifact_name = f"{safe_run_name}_round{server_round}_auroc{auroc_str}"
        artifact = wandb.Artifact(artifact_name, type="model", metadata=metadata)
        artifact.add_file(state_dict_file)
        wandb.log_artifact(artifact)

        # Clean up temp file after upload
        os.remove(state_dict_file)

        return (
            True,
            f"✓ New best model! Round {server_round}, AUROC: {current_auroc:.4f}",
        )
    else:
        return (
            False,
            f"  Model not saved (AUROC {current_auroc:.4f} ≤ best {best_auroc_tracker['auroc']:.4f})",
        )

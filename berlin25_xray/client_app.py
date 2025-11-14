import logging
from dataclasses import dataclass
from typing import Any, Mapping

import torch
from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp

from berlin25_xray.logging_utils import configure_logging, log_gpu_utilization
from berlin25_xray.task import (
    DEFAULT_BATCH_SIZE,
    Net,
    dataset_name_from_partition,
    load_data,
    maybe_compile_model,
    prepare_model_for_device,
)
from berlin25_xray.task import test as test_fn
from berlin25_xray.task import train as train_fn

app = ClientApp()
configure_logging()
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ClientSettings:
    """Subset of run_config values that every client needs."""

    image_size: int
    batch_size: int
    compile_model: bool
    local_epochs: int


def resolve_client_settings(run_config: Mapping[str, Any]) -> ClientSettings:
    """Convert Flower run_config into a typed settings container."""

    return ClientSettings(
        image_size=int(run_config["image-size"]),
        batch_size=int(run_config.get("batch-size", DEFAULT_BATCH_SIZE)),
        compile_model=bool(run_config.get("compile-model", True)),
        local_epochs=int(run_config["local-epochs"]),
    )


def _unwrap_compiled_model(model):
    """Return the original nn.Module even if torch.compile wrapped it."""

    return getattr(model, "_orig_mod", model)


def _load_partition_split(partition_id: int, split: str, image_size: int, batch_size: int):
    """Reusable helper to load a train/eval split for the given partition."""

    dataset_name = dataset_name_from_partition(partition_id)
    logger.info(
        "Loading %s data | dataset=%s | image_size=%d | batch_size=%d | partition=%s",
        split,
        dataset_name,
        image_size,
        batch_size,
        partition_id,
    )
    loader = load_data(
        dataset_name,
        split,
        image_size=image_size,
        batch_size=batch_size,
    )
    logger.info(
        "%s dataloader ready | dataset=%s | samples=%d | batches=%d",
        split.capitalize(),
        dataset_name,
        len(loader.dataset),
        len(loader),
    )
    return dataset_name, loader


@app.train()
def train(msg: Message, context: Context):
    """Train the model on local data."""

    # Load the model and initialize it with the received weights
    settings = resolve_client_settings(context.run_config)
    model = Net(image_size=settings.image_size)
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger.info("Starting client training on %s", device)
    model = prepare_model_for_device(model, device)
    model = maybe_compile_model(
        model,
        mode="client-train",
        enabled=settings.compile_model,
    )
    log_gpu_utilization(logger, device, prefix="Client/train/device")

    # Load the data
    partition_id = context.node_config["partition-id"]
    dataset_name, trainloader = _load_partition_split(
        partition_id,
        "train",
        settings.image_size,
        settings.batch_size,
    )

    # Call the training function
    lr = msg.content["config"]["lr"]
    train_loss = train_fn(
        model,
        trainloader,
        settings.local_epochs,
        lr,
        device,
    )
    logger.info(
        "Training complete | dataset=%s | train_loss=%.6f",
        dataset_name,
        train_loss,
    )

    # Construct and return reply Message
    state_src = _unwrap_compiled_model(model)
    model_record = ArrayRecord(state_src.state_dict())
    metrics = {
        "partition-id": context.node_config["partition-id"],
        "train_loss": train_loss,
        "num-examples": len(trainloader.dataset),
    }
    metric_record = MetricRecord(metrics)
    content = RecordDict({"arrays": model_record, "metrics": metric_record})
    return Message(content=content, reply_to=msg)


@app.evaluate()
def evaluate(msg: Message, context: Context):
    """Evaluate the model on local data."""
    settings = resolve_client_settings(context.run_config)
    model = Net(image_size=settings.image_size)
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = prepare_model_for_device(model, device)
    model = maybe_compile_model(
        model,
        mode="client-eval",
        enabled=settings.compile_model,
    )
    logger.info("Starting client evaluation on %s", device)
    log_gpu_utilization(logger, device, prefix="Client/eval/device")

    partition_id = context.node_config["partition-id"]
    dataset_name, valloader = _load_partition_split(
        partition_id,
        "eval",
        settings.image_size,
        settings.batch_size,
    )

    eval_loss, tp, tn, fp, fn, probs, labels = test_fn(model, valloader, device)
    logger.info(
        "Evaluation complete | dataset=%s | eval_loss=%.6f",
        dataset_name,
        eval_loss,
    )

    metric_record = MetricRecord(
        {
            "partition-id": context.node_config["partition-id"],
            "eval_loss": eval_loss,
            "tp": tp,
            "tn": tn,
            "fp": fp,
            "fn": fn,
            "num-examples": len(valloader.dataset),
            "probs": probs.tolist(),  # Convert numpy array to list for MetricRecord
            "labels": labels.tolist(),  # Convert numpy array to list for MetricRecord
        }
    )
    content = RecordDict({"metrics": metric_record})
    return Message(content=content, reply_to=msg)

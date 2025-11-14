import logging

import torch
from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp

from berlin25_xray.logging_utils import configure_logging, log_gpu_utilization, log_timing
from berlin25_xray.task import PARTITION_HOSPITAL_MAP, Net, load_data
from berlin25_xray.task import test as test_fn
from berlin25_xray.task import train as train_fn

app = ClientApp()
configure_logging()
logger = logging.getLogger(__name__)


@app.train()
def train(msg: Message, context: Context):
    """Train the model on local data."""

    # Load the model and initialize it with the received weights
    model = Net()
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger.info("Starting client training on %s", device)
    model.to(device)
    log_gpu_utilization(logger, device, prefix="Client/train/device")

    # Load the data
    partition_id = context.node_config["partition-id"]
    dataset_name = f"Hospital{PARTITION_HOSPITAL_MAP[partition_id]}"
    image_size = context.run_config["image-size"]
    batch_size = context.run_config.get("batch-size", 16)
    logger.info(
        "Loading train data | dataset=%s | image_size=%d | batch_size=%d | partition=%s",
        dataset_name,
        image_size,
        batch_size,
        partition_id,
    )
    with log_timing(logger, f"Client train dataloader ({dataset_name})"):
        trainloader = load_data(
            dataset_name,
            "train",
            image_size=image_size,
            batch_size=batch_size,
        )
    logger.info(
        "Train dataloader ready | dataset=%s | samples=%d | batches=%d",
        dataset_name,
        len(trainloader.dataset),
        len(trainloader),
    )
    log_gpu_utilization(logger, device, prefix="Client/train/post-dataloader")

    # Call the training function
    local_epochs = context.run_config["local-epochs"]
    lr = msg.content["config"]["lr"]
    with log_timing(
        logger,
        (
            f"Client train loop dataset={dataset_name} "
            f"epochs={local_epochs} lr={lr}"
        ),
    ):
        train_loss = train_fn(
            model,
            trainloader,
            local_epochs,
            lr,
            device,
        )
    logger.info(
        "Training complete | dataset=%s | train_loss=%.6f",
        dataset_name,
        train_loss,
    )
    log_gpu_utilization(logger, device, prefix="Client/train/done")

    # Construct and return reply Message
    model_record = ArrayRecord(model.state_dict())
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
    model = Net()
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    logger.info("Starting client evaluation on %s", device)
    log_gpu_utilization(logger, device, prefix="Client/eval/device")

    partition_id = context.node_config["partition-id"]
    dataset_name = f"Hospital{PARTITION_HOSPITAL_MAP[partition_id]}"
    image_size = context.run_config["image-size"]
    batch_size = context.run_config.get("batch-size", 16)
    logger.info(
        "Loading eval data | dataset=%s | image_size=%d | batch_size=%d",
        dataset_name,
        image_size,
        batch_size,
    )
    with log_timing(logger, f"Client eval dataloader ({dataset_name})"):
        valloader = load_data(
            dataset_name,
            "eval",
            image_size=image_size,
            batch_size=batch_size,
        )
    logger.info(
        "Eval dataloader ready | dataset=%s | samples=%d | batches=%d",
        dataset_name,
        len(valloader.dataset),
        len(valloader),
    )
    log_gpu_utilization(logger, device, prefix="Client/eval/post-dataloader")

    with log_timing(logger, f"Client eval loop dataset={dataset_name}"):
        eval_loss, tp, tn, fp, fn, probs, labels = test_fn(model, valloader, device)
    logger.info(
        "Evaluation complete | dataset=%s | eval_loss=%.6f",
        dataset_name,
        eval_loss,
    )
    log_gpu_utilization(logger, device, prefix="Client/eval/done")

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

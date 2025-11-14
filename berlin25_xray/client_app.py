import os

import torch
from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp

from berlin25_xray.task import PARTITION_HOSPITAL_MAP, Net, load_data
from berlin25_xray.task import test as test_fn
from berlin25_xray.task import train as train_fn

from ..fl_checkpoints import save_client_ckpt

app = ClientApp()


@app.train()
def train(msg: Message, context: Context):
    """Train the model on local data."""

    model = Net()
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")
    model.to(device)

    partition_id = context.node_config["partition-id"]
    dataset_name = f"Hospital{PARTITION_HOSPITAL_MAP[partition_id]}"
    image_size = context.run_config["image-size"]
    batch_size = context.run_config.get("batch-size", 16)
    trainloader = load_data(
        dataset_name, "train", image_size=image_size, batch_size=batch_size
    )

    server_round = msg.metadata.get("server_round", 0)

    train_loss = train_fn(
        model,
        trainloader,
        context.run_config["local-epochs"],
        msg.content["config"]["lr"],
        device,
    )

    run_dir = os.path.expanduser("~/coldstart_runs/flower_bigmodel")
    cid = str(partition_id)

    optimizer = torch.optim.Adam(model.parameters(), lr=msg.content["config"]["lr"])
    meta = {
        "partition": partition_id,
        "dataset": dataset_name,
        "train_loss": train_loss,
    }

    ckpt_path = save_client_ckpt(
        run_dir, cid, server_round, model.state_dict(), optimizer.state_dict(), meta
    )
    print(f"cid={cid}, round={server_round:04d}, saved {os.path.basename(ckpt_path)}")

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

    partition_id = context.node_config["partition-id"]
    dataset_name = f"Hospital{PARTITION_HOSPITAL_MAP[partition_id]}"
    image_size = context.run_config["image-size"]
    batch_size = context.run_config.get("batch-size", 16)
    valloader = load_data(
        dataset_name, "eval", image_size=image_size, batch_size=batch_size
    )

    eval_loss, tp, tn, fp, fn, probs, labels = test_fn(model, valloader, device)

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

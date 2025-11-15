from collections import OrderedDict

import torch
from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp

from berlin25_xray.task import (
    PARTITION_HOSPITAL_MAP,
    Net,
    get_pos_weight,
    load_data,
)
from berlin25_xray.task import test as test_fn
from berlin25_xray.task import train as train_fn
from berlin25_xray.fedbn import get_batchnorm_keys, split_state_dict_by_bn

app = ClientApp()
LOCAL_BN_STATE: dict[int, OrderedDict[str, torch.Tensor]] = {}


def _restore_local_bn_state(model: Net, partition_id: int) -> set[str]:
    """Load cached BN statistics into the model if we have them."""
    bn_keys = get_batchnorm_keys(model)
    bn_state = LOCAL_BN_STATE.get(partition_id)
    if bn_state:
        model.load_state_dict(bn_state, strict=False)
    return bn_keys


def _cache_local_bn_state(model: Net, partition_id: int, bn_keys: set[str]) -> None:
    """Persist BN parameters/buffers per-partition so they stay local across rounds."""
    _, bn_state = split_state_dict_by_bn(model.state_dict(), bn_keys)
    LOCAL_BN_STATE[partition_id] = OrderedDict(
        (k, v.detach().cpu().clone()) for k, v in bn_state.items()
    )


@app.train()
def train(msg: Message, context: Context):
    """Train the model on local data."""

    # Load the model and initialize it with the received weights
    model = Net()
    partition_id = context.node_config["partition-id"]
    bn_keys = _restore_local_bn_state(model, partition_id)
    state_dict = msg.content["arrays"].to_torch_state_dict()
    state_dict_wo_bn, _ = split_state_dict_by_bn(state_dict, bn_keys)
    model.load_state_dict(state_dict_wo_bn, strict=False)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")
    model.to(device)

    # Load the data
    dataset_name = f"Hospital{PARTITION_HOSPITAL_MAP[partition_id]}"
    image_size = context.run_config["image-size"]
    trainloader = load_data(dataset_name, "train", image_size=image_size)
    pos_weight = get_pos_weight(dataset_name, "train", image_size=image_size)

    # Call the training function
    train_loss = train_fn(
        model,
        trainloader,
        context.run_config["local-epochs"],
        msg.content["config"]["lr"],
        device,
        pos_weight=pos_weight,
    )

    # Construct and return reply Message
    state_dict_wo_bn, _ = split_state_dict_by_bn(model.state_dict(), bn_keys)
    _cache_local_bn_state(model, partition_id, bn_keys)
    model_record = ArrayRecord(state_dict_wo_bn)
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
    partition_id = context.node_config["partition-id"]
    bn_keys = _restore_local_bn_state(model, partition_id)
    state_dict = msg.content["arrays"].to_torch_state_dict()
    state_dict_wo_bn, _ = split_state_dict_by_bn(state_dict, bn_keys)
    model.load_state_dict(state_dict_wo_bn, strict=False)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    dataset_name = f"Hospital{PARTITION_HOSPITAL_MAP[partition_id]}"
    image_size = context.run_config["image-size"]
    valloader = load_data(dataset_name, "eval", image_size=image_size)

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

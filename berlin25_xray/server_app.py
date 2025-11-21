"""Flower server for federated training of X-ray classification model."""

import os
import subprocess
from datetime import datetime
from logging import INFO

from flwr.app import ArrayRecord, ConfigRecord, Context
from flwr.common import log
from flwr.serverapp import Grid, ServerApp
from flwr.serverapp.strategy import FedAvg

from berlin25_xray.task import Net
from berlin25_xray.util import (
    compute_aggregated_metrics,
    log_eval_metrics,
    log_training_metrics,
    save_best_model,
)

app = ServerApp()


@app.main()
def main(grid: Grid, context: Context) -> None:
    """Main server application entry point."""
    num_rounds: int = context.run_config["num-server-rounds"]
    lr: float = context.run_config["lr"]

    def git_cmd(args):
        return subprocess.check_output(["git"] + args).decode("utf-8").strip()

    try:
        branch = git_cmd(["rev-parse", "--abbrev-ref", "HEAD"])
        commit = git_cmd(["rev-parse", "--short", "HEAD"])
    except Exception:  # pylint: disable=broad-except
        branch, commit = "unknown", "unknown"

    run_name = f"{branch}-{commit}"

    # Create model directory: models/run_<commit>_<time>
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_dir_name = f"run_{commit}_{timestamp}"
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    model_dir = os.path.join(project_root, "models", model_dir_name)
    os.makedirs(model_dir, exist_ok=True)
    print(f"Model checkpoints will be saved to: {model_dir}")

    global_model = Net()
    arrays = ArrayRecord(global_model.state_dict())
    strategy = HackathonFedAvg(run_name=run_name, model_dir=model_dir)

    _ = strategy.start(
        grid=grid,
        initial_arrays=arrays,
        train_config=ConfigRecord({"lr": lr}),
        num_rounds=num_rounds,
    )

    log(INFO, "Training complete")


class HackathonFedAvg(FedAvg):
    """FedAvg strategy that logs metrics."""

    def __init__(self, *args, run_name=None, model_dir=None, **kwargs):
        super().__init__(*args, **kwargs)
        self._best_auroc = {}
        self._run_name = run_name
        self._model_dir = model_dir
        self._arrays = None

    def aggregate_train(self, server_round, replies):
        arrays, metrics = super().aggregate_train(server_round, replies)
        self._arrays = arrays
        log_training_metrics(replies)
        return arrays, metrics

    def aggregate_evaluate(self, server_round, replies):
        agg_metrics = compute_aggregated_metrics(replies)
        log_eval_metrics(
            replies,
            agg_metrics,
            self.weighted_by_key,
            lambda msg: log(INFO, msg),
        )
        print(f"DEBUG:{hasattr(self, '_arrays')}")

        if hasattr(self, "_arrays"):
            _, msg = save_best_model(
                arrays=self._arrays,
                agg_metrics=agg_metrics,
                server_round=server_round,
                best_auroc_tracker=self._best_auroc,
                model_dir=self._model_dir,
            )
            log(INFO, msg)

        return agg_metrics

import os
from logging import INFO

import wandb
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
    num_rounds: int = context.run_config["num-server-rounds"]
    lr: float = context.run_config["lr"]
    local_epochs: int = context.run_config["local-epochs"]

    # Get run name from environment variable (set by submit_job.sh). Feel free to change this.
    run_name = os.environ.get("JOB_NAME", "your_custom_run_name")

    # W&B auth and project/entity are configured via environment variables
    wandb.login()
    log(INFO, "Wandb login successful")
    wandb.init(
        project="hackathon",
        entity="justus-krebs-technische-universit-t-berlin",
        config={
            "num_rounds": num_rounds,
            "learning_rate": lr,
            "local_epochs": local_epochs,
        },
    )
    log(INFO, "Wandb initialized with run_id: %s", wandb.run.id)

    global_model = Net()
    arrays = ArrayRecord(global_model.state_dict())

    strategy = HackathonFedAvg(
        fraction_train=1,
        run_name=run_name,
        min_fit_clients=1,
        min_available_clients=1,
        min_evaluate_clients=1,
    )
    result = strategy.start(
        grid=grid,
        initial_arrays=arrays,
        train_config=ConfigRecord({"lr": lr}),
        num_rounds=num_rounds,
    )

    log(INFO, "Training complete")
    wandb.finish()
    log(INFO, "Wandb run finished")


class HackathonFedAvg(FedAvg):
    """FedAvg strategy that logs metrics and saves best model to W&B."""

    def __init__(self, *args, run_name=None, **kwargs):
        super().__init__(*args, **kwargs)
        self._best_auroc = {}
        self._run_name = run_name or "your_run"

    def aggregate_train(self, server_round, replies):
        arrays, metrics = super().aggregate_train(server_round, replies)
        self._arrays = arrays
        log_training_metrics(replies, server_round)
        return arrays, metrics

    def aggregate_evaluate(self, server_round, replies):
        agg_metrics = compute_aggregated_metrics(replies)
        log_eval_metrics(
            replies,
            agg_metrics,
            server_round,
            self.weighted_by_key,
            lambda msg: log(INFO, msg),
        )

        if hasattr(self, "_arrays"):
            saved, msg = save_best_model(
                self._arrays,
                agg_metrics,
                server_round,
                self._run_name,
                self._best_auroc,
            )
            log(INFO, msg)

        return agg_metrics

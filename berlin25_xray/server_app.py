import os
import subprocess
import sys
from logging import INFO

import wandb
from flwr.app import ArrayRecord, ConfigRecord, Context
from flwr.common import log, ndarrays_to_parameters, parameters_to_ndarrays
from flwr.serverapp import Grid, ServerApp
from flwr.serverapp.strategy import FedAvg

from berlin25_xray.task import Net
from berlin25_xray.util import (
    compute_aggregated_metrics,
    log_eval_metrics,
    log_training_metrics,
    save_best_model,
)

from ..fl_checkpoints import load_latest_server_ckpt, save_server_ckpt

app = ServerApp()


@app.main()
def main(grid: Grid, context: Context) -> None:
    num_rounds: int = context.run_config["num-server-rounds"]
    lr: float = context.run_config["lr"]
    local_epochs: int = context.run_config["local-epochs"]
    rounds_this_job: int = context.run_config.get("rounds-this-job", num_rounds)

    run_dir = os.path.expanduser("~/coldstart_runs/flower_bigmodel")
    os.makedirs(run_dir, exist_ok=True)

    ckpt = load_latest_server_ckpt(run_dir)
    if ckpt is not None:
        start_round = ckpt["round"] + 1
        initial_parameters = ndarrays_to_parameters(ckpt["parameters"])
        log(
            INFO,
            f"Resuming from round {ckpt['round']}, starting at round {start_round}",
        )
    else:
        start_round = 1
        initial_parameters = None
        log(INFO, "Starting training from scratch")

    def git_cmd(args):
        return subprocess.check_output(["git"] + args).decode("utf-8").strip()

    try:
        branch = git_cmd(["rev-parse", "--abbrev-ref", "HEAD"])
        commit = git_cmd(["rev-parse", "--short", "HEAD"])
    except Exception:
        branch, commit = "unknown", "unknown"

    run_name = f"{branch}-{commit}"

    wandb.login()
    log(INFO, "Wandb login successful")
    wandb.init(
        project="hackathon",
        entity="justus-krebs-technische-universit-t-berlin",
        name=run_name,
        config={
            "num_rounds": num_rounds,
            "learning_rate": lr,
            "local_epochs": local_epochs,
            "git_branch": branch,
            "git_commit": commit,
            "start_round": start_round,
            "rounds_this_job": rounds_this_job,
        },
        tags=[branch, commit],
    )
    log(INFO, "Wandb initialized with run_id: %s", wandb.run.id)

    if initial_parameters is None:
        global_model = Net()
        arrays = ArrayRecord(global_model.state_dict())
    else:
        arrays = None

    target_round = start_round + rounds_this_job - 1
    strategy = HackathonFedAvg(
        fraction_train=1,
        run_name=run_name,
        run_dir=run_dir,
        start_round=start_round,
        target_round=target_round,
    )
    strategy.start(
        grid=grid,
        initial_arrays=arrays if arrays else None,
        train_config=ConfigRecord({"lr": lr}),
        num_rounds=num_rounds,
    )

    log(INFO, "Training complete")
    wandb.finish()
    log(INFO, "Wandb run finished")


class HackathonFedAvg(FedAvg):
    """FedAvg strategy that logs metrics and saves best model to W&B."""

    def __init__(
        self,
        *args,
        run_name=None,
        run_dir=None,
        start_round=1,
        target_round=None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._best_auroc = {}
        self._run_name = run_name or "your_run"
        self._run_dir = run_dir
        self._start_round = start_round
        self._target_round = target_round

    def aggregate_train(self, server_round, replies):
        arrays, metrics = super().aggregate_train(server_round, replies)
        self._arrays = arrays
        log_training_metrics(replies, server_round)

        if self._run_dir and arrays:
            param_arrays = parameters_to_ndarrays(arrays)
            param_obj = ndarrays_to_parameters(param_arrays)
            save_server_ckpt(
                self._run_dir,
                server_round,
                param_obj,
                strategy_state={"metrics": metrics},
            )
            log(INFO, f"Saved server checkpoint for round {server_round}")

        if self._target_round and server_round >= self._target_round:
            log(INFO, f"Reached target round {self._target_round}, exiting")
            sys.exit(0)

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

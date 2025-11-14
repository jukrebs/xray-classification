import os
import pickle
from pathlib import Path
from typing import Any, Optional

import flwr as fl
import torch
from flwr.common import parameters_to_ndarrays


def atomic_save(obj: Any, path: str) -> None:
    tmp_path = f"{path}.tmp"
    with open(tmp_path, "wb") as f:
        pickle.dump(obj, f)
    os.replace(tmp_path, path)


def symlink_latest(latest_path: str, tgt_filename: str) -> None:
    latest_link = Path(latest_path)
    tgt = Path(tgt_filename)
    
    if latest_link.exists() or latest_link.is_symlink():
        latest_link.unlink()
    
    try:
        latest_link.symlink_to(tgt.name)
    except OSError:
        with open(latest_link, "w") as f:
            f.write(tgt.name)


def save_server_ckpt(
    run_dir: str,
    rnd: int,
    parameters: fl.common.Parameters,
    strategy_state: dict = {}
) -> str:
    run_path = Path(run_dir)
    run_path.mkdir(parents=True, exist_ok=True)
    
    ckpt_filename = run_path / f"server_round_{rnd:04d}.pt"
    latest_filename = run_path / "latest.pt"
    
    param_ndarrays = parameters_to_ndarrays(parameters)
    
    ckpt_data = {
        "round": rnd,
        "parameters": param_ndarrays,
        "strategy_state": strategy_state,
    }
    
    atomic_save(ckpt_data, str(ckpt_filename))
    symlink_latest(str(latest_filename), str(ckpt_filename))
    
    return str(ckpt_filename)


def load_latest_server_ckpt(run_dir: str) -> Optional[dict]:
    run_path = Path(run_dir)
    latest_file = run_path / "latest.pt"
    
    if not latest_file.exists():
        return None
    
    if latest_file.is_symlink():
        target = latest_file.resolve()
    else:
        with open(latest_file, "r") as f:
            target_name = f.read().strip()
        target = run_path / target_name
    
    if not target.exists():
        return None
    
    with open(target, "rb") as f:
        ckpt_data = pickle.load(f)
    
    return ckpt_data


def save_client_ckpt(
    run_dir: str,
    cid: str,
    server_round: int,
    model_sd: dict,
    opt_sd: dict = None,
    meta: dict = {}
) -> str:
    run_path = Path(run_dir) / f"client_{cid}"
    run_path.mkdir(parents=True, exist_ok=True)
    
    ckpt_filename = run_path / f"local_round_{server_round:04d}.pt"
    latest_filename = run_path / "latest.pt"
    
    ckpt_data = {
        "server_round": server_round,
        "model_state_dict": model_sd,
        "optimizer_state_dict": opt_sd,
        "meta": meta,
    }
    
    torch.save(ckpt_data, str(ckpt_filename))
    
    tmp_latest = f"{latest_filename}.tmp"
    torch.save(ckpt_data, tmp_latest)
    os.replace(tmp_latest, str(latest_filename))
    
    return str(ckpt_filename)


def load_client_latest(run_dir: str, cid: str) -> Optional[dict]:
    run_path = Path(run_dir) / f"client_{cid}"
    latest_file = run_path / "latest.pt"
    
    if not latest_file.exists():
        return None
    
    ckpt_data = torch.load(latest_file)
    return ckpt_data

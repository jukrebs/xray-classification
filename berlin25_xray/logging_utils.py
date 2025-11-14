"""Logging helpers for monitoring GPU and runtime performance."""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import time
from contextlib import contextmanager
from typing import Dict, Optional

import torch


def configure_logging(default_level: str = "INFO") -> None:
    """Configure a sane default logging setup (idempotent)."""

    level_name = os.environ.get("XRAY_LOG_LEVEL", default_level).upper()
    level = getattr(logging, level_name, logging.INFO)
    root_logger = logging.getLogger()
    if not root_logger.handlers:
        logging.basicConfig(
            level=level,
            format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        )
    else:
        root_logger.setLevel(level)


def _detect_device_index(device: torch.device) -> int:
    index = device.index
    if index is not None:
        return index
    try:
        return torch.cuda.current_device()
    except Exception:  # pragma: no cover - runtime specific
        return 0


def _query_nvidia_smi(index: int) -> Optional[Dict[str, float]]:
    """Best-effort GPU stats via ``nvidia-smi`` if available."""

    smi_path = shutil.which("nvidia-smi")
    if not smi_path:
        return None

    cmd = [
        smi_path,
        "--query-gpu=utilization.gpu,memory.used,memory.total",
        "--format=csv,noheader,nounits",
    ]
    try:
        proc = subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):  # pragma: no cover
        return None

    lines = [ln.strip() for ln in proc.stdout.strip().splitlines() if ln.strip()]
    if index >= len(lines):
        return None

    try:
        util_str, used_str, total_str = [val.strip() for val in lines[index].split(",")]
        return {
            "utilization": float(util_str),
            "mem_used_mb": float(used_str),
            "mem_total_mb": float(total_str),
        }
    except ValueError:  # pragma: no cover - unexpected output
        return None


def get_gpu_stats(device: torch.device) -> Optional[Dict[str, float]]:
    """Collect GPU utilization/memory stats when running on CUDA."""

    if device.type != "cuda" or not torch.cuda.is_available():
        return None

    index = _detect_device_index(device)
    try:
        props = torch.cuda.get_device_properties(index)
        mem_free, mem_total = torch.cuda.mem_get_info(index)
    except RuntimeError:  # pragma: no cover - CUDA driver hiccups
        return None
    mem_used = mem_total - mem_free

    stats: Dict[str, float] = {
        "index": index,
        "name": props.name,
        "mem_used_mb": round(mem_used / (1024**2), 2),
        "mem_total_mb": round(mem_total / (1024**2), 2),
        "mem_allocated_mb": round(torch.cuda.memory_allocated(index) / (1024**2), 2),
        "mem_reserved_mb": round(torch.cuda.memory_reserved(index) / (1024**2), 2),
    }

    util_fn = getattr(torch.cuda, "utilization", None)
    utilization: Optional[float] = None
    if callable(util_fn):
        try:
            utilization = float(util_fn(index))
        except Exception:  # pragma: no cover - backend dependent
            utilization = None

    if utilization is None:
        smi_stats = _query_nvidia_smi(index)
        if smi_stats:
            utilization = smi_stats.get("utilization")
            stats["mem_used_mb"] = smi_stats.get("mem_used_mb", stats["mem_used_mb"])
            stats["mem_total_mb"] = smi_stats.get("mem_total_mb", stats["mem_total_mb"])

    if utilization is not None:
        stats["utilization"] = round(utilization, 2)

    return stats


def log_gpu_utilization(logger: logging.Logger, device: torch.device, *, prefix: str) -> None:
    """Log GPU memory/utilization with helpful metadata."""

    stats = get_gpu_stats(device)
    if not stats:
        logger.info("%s | running on %s (no CUDA stats available)", prefix, device)
        return

    util_str = f"{stats.get('utilization', 'n/a')}%"
    logger.info(
        "%s | GPU %d (%s): util=%s, used %.1f/%.1f MB (alloc %.1f MB, reserved %.1f MB)",
        prefix,
        stats["index"],
        stats["name"],
        util_str,
        stats["mem_used_mb"],
        stats["mem_total_mb"],
        stats["mem_allocated_mb"],
        stats["mem_reserved_mb"],
    )


@contextmanager
def log_timing(logger: logging.Logger, description: str):
    """Context manager that logs how long a code block took."""

    start = time.perf_counter()
    try:
        yield
    finally:
        duration = time.perf_counter() - start
        logger.info("%s took %.2fs", description, duration)

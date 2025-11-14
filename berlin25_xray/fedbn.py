from __future__ import annotations

from collections import OrderedDict
from typing import Mapping

import torch
from torch import nn
from torch.nn.modules.batchnorm import _BatchNorm

# Type alias for readability
BatchNormState = OrderedDict[str, torch.Tensor]


def _get_bn_keys(model: nn.Module) -> frozenset[str]:
    """Return the set of BatchNorm-related parameter/buffer keys for ``model``."""

    cache_name = "_fedbn_bn_keys"
    cached = getattr(model, cache_name, None)
    if cached is not None:
        return cached

    keys: set[str] = set()
    for module_name, module in model.named_modules():
        if isinstance(module, _BatchNorm):
            prefix = module_name
            for param_name, _ in module.named_parameters(recurse=False):
                key = f"{prefix}.{param_name}" if prefix else param_name
                keys.add(key)
            for buffer_name, _ in module.named_buffers(recurse=False):
                key = f"{prefix}.{buffer_name}" if prefix else buffer_name
                keys.add(key)

    frozen = frozenset(keys)
    setattr(model, cache_name, frozen)
    return frozen


def load_non_bn_state_dict(model: nn.Module, state_dict: Mapping[str, torch.Tensor]) -> None:
    """Load ``state_dict`` while ignoring BatchNorm parameters/buffers."""

    bn_keys = _get_bn_keys(model)
    non_bn_state = OrderedDict((k, v) for k, v in state_dict.items() if k not in bn_keys)
    if not non_bn_state:
        return
    model.load_state_dict(non_bn_state, strict=False)


def get_non_bn_state_dict(model: nn.Module) -> OrderedDict[str, torch.Tensor]:
    """Return a CPU copy of the model state dict without BatchNorm entries."""

    bn_keys = _get_bn_keys(model)
    state = model.state_dict()
    filtered: OrderedDict[str, torch.Tensor] = OrderedDict()
    for key, value in state.items():
        if key in bn_keys:
            continue
        filtered[key] = value.detach().cpu()
    return filtered


def capture_bn_state(model: nn.Module) -> BatchNormState:
    """Capture and return a CPU copy of the local BatchNorm parameters/buffers."""

    bn_keys = _get_bn_keys(model)
    state = model.state_dict()
    bn_state: BatchNormState = OrderedDict()
    for key in bn_keys:
        tensor = state.get(key)
        if tensor is None:
            continue
        bn_state[key] = tensor.detach().cpu().clone()
    return bn_state


def restore_bn_state(model: nn.Module, bn_state: Mapping[str, torch.Tensor] | None) -> None:
    """Load ``bn_state`` back into ``model`` without touching other parameters."""

    if not bn_state:
        return
    state = model.state_dict()
    for key, tensor in bn_state.items():
        target = state.get(key)
        if target is None:
            continue
        target.copy_(tensor.to(target.device))

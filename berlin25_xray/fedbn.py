"""Utilities for implementing FedBN with Flower."""

from __future__ import annotations

from collections import OrderedDict
from typing import Iterable, MutableMapping, Tuple

import torch
import torch.nn as nn


def get_batchnorm_keys(model: nn.Module) -> set[str]:
    """Return the state_dict keys that belong to BatchNorm layers."""
    keys: set[str] = set()
    for module_name, module in model.named_modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            prefix = f"{module_name}." if module_name else ""
            for param_name, _ in module.named_parameters(recurse=False):
                keys.add(prefix + param_name)
            for buffer_name, _ in module.named_buffers(recurse=False):
                keys.add(prefix + buffer_name)
    return keys


def split_state_dict_by_bn(
    state_dict: MutableMapping[str, torch.Tensor],
    bn_keys: Iterable[str],
) -> Tuple[OrderedDict[str, torch.Tensor], OrderedDict[str, torch.Tensor]]:
    """Split a state dict into (non-BN, BN) OrderedDicts."""
    bn_key_set = set(bn_keys)
    non_bn_state: "OrderedDict[str, torch.Tensor]" = OrderedDict()
    bn_state: "OrderedDict[str, torch.Tensor]" = OrderedDict()
    for key, value in state_dict.items():
        target = bn_state if key in bn_key_set else non_bn_state
        target[key] = value
    return non_bn_state, bn_state

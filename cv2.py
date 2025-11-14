"""Minimal stub implementation of required OpenCV APIs using PIL/numpy.

This satisfies the CXformer preprocessing dependencies without installing the
full opencv-python package, which is not available in the environment.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
from PIL import Image

__all__ = [
    "IMREAD_COLOR",
    "IMREAD_GRAYSCALE",
    "COLOR_BGR2RGB",
    "COLOR_RGB2BGR",
    "COLOR_BGR2GRAY",
    "COLOR_RGB2GRAY",
    "COLOR_GRAY2BGR",
    "COLOR_GRAY2RGB",
    "INTER_LINEAR",
    "INTER_AREA",
    "INTER_CUBIC",
    "INTER_NEAREST",
    "cvtColor",
    "imread",
    "resize",
]


IMREAD_COLOR = 1
IMREAD_GRAYSCALE = 0

COLOR_BGR2RGB = 0
COLOR_RGB2BGR = 1
COLOR_BGR2GRAY = 2
COLOR_RGB2GRAY = 3
COLOR_GRAY2BGR = 4
COLOR_GRAY2RGB = 5

INTER_NEAREST = Image.NEAREST
INTER_LINEAR = Image.BILINEAR
INTER_AREA = Image.BOX
INTER_CUBIC = Image.BICUBIC

__version__ = "0.0.0-stub"


def _to_image_array(arr: np.ndarray) -> np.ndarray:
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return arr


def _as_pil(arr: np.ndarray) -> Image.Image:
    arr = _to_image_array(arr)
    if arr.ndim == 2:
        return Image.fromarray(arr, mode="L")
    if arr.ndim == 3:
        if arr.shape[2] == 1:
            return Image.fromarray(arr.squeeze(-1), mode="L")
        return Image.fromarray(arr, mode="RGB")
    raise ValueError(f"Unsupported image array with shape {arr.shape}")


def imread(path: str, flags: int = IMREAD_COLOR) -> np.ndarray:
    img = Image.open(path)
    if flags == IMREAD_GRAYSCALE:
        img = img.convert("L")
        return np.array(img)
    img = img.convert("RGB")
    arr = np.array(img)
    # OpenCV returns BGR ordering by default.
    return arr[..., ::-1]


def resize(
    image: np.ndarray,
    dsize: Tuple[int, int],
    interpolation: int = INTER_LINEAR,
) -> np.ndarray:
    width, height = dsize
    pil_img = _as_pil(image)
    resized = pil_img.resize((width, height), interpolation)
    arr = np.array(resized)
    if image.ndim == 3 and image.shape[2] == 1:
        arr = arr[..., None]
    return arr


def cvtColor(image: np.ndarray, code: int) -> np.ndarray:
    if code in (COLOR_BGR2RGB, COLOR_RGB2BGR):
        return image[..., ::-1]
    if code in (COLOR_BGR2GRAY, COLOR_RGB2GRAY):
        if image.ndim == 2:
            return image
        r, g, b = image[..., 2], image[..., 1], image[..., 0]
        gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
        return gray.astype(image.dtype)
    if code in (COLOR_GRAY2BGR, COLOR_GRAY2RGB):
        if image.ndim == 3 and image.shape[2] == 3:
            return image
        if image.ndim == 2:
            return np.stack([image, image, image], axis=-1)
        if image.shape[2] == 1:
            img = image.squeeze(-1)
            return np.stack([img, img, img], axis=-1)
    raise NotImplementedError(f"cvtColor code {code} not supported in stub")

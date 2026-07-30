"""Audit oracle for the vendor-literal XSlim image preprocessing path."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np


def vendor_literal_rgb_nchw(path: str | Path, size: int = 640) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"cannot decode image: {path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (size, size), interpolation=cv2.INTER_AREA)
    return np.ascontiguousarray(
        image.astype(np.float32).transpose(2, 0, 1) / np.float32(255.0)
    )


def preprocess_impl(
    file_names: Sequence[str], input_info: dict[str, Any]
) -> "torch.Tensor":
    import torch

    input_shape = input_info.get("input_shape") or [1, 3, 640, 640]
    size = int(input_shape[-1])
    tensors = [vendor_literal_rgb_nchw(path, size) for path in file_names]
    return torch.from_numpy(np.stack(tensors, axis=0))


# The measured vendor-literal configurations deliberately omit preprocess_file
# and exercise XSlim's internal implementation. This module is the independent
# audit equivalent used to state that implementation precisely.

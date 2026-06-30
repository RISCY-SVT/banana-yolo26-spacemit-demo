#!/usr/bin/env python3
"""@file xslim_yolo26_letterbox_preprocess.py
@brief XSlim calibration preprocessing for YOLO26 ONNX exports.
@details XSlim calls the function named in `preprocess_file` with a batch of
image paths and one input-parameter dictionary. This helper mirrors the
YOLO26 FP32 oracle preprocessing used in this R&D repo: OpenCV BGR load,
square letterbox with value 114, RGB conversion, float32 normalization to
0..1, and NCHW batching. It intentionally does not use ImageNet mean/std.
"""

from __future__ import annotations

from typing import Sequence

import cv2
import numpy as np
import torch


def _letterbox_rgb_nchw(path: str, input_shape: Sequence[int]) -> np.ndarray:
    """Return one RGB/NCHW/float32 YOLO-style letterboxed tensor.

    Args:
        path: Input image path readable by OpenCV.
        input_shape: Model input shape in NCHW order.

    Returns:
        A `[C,H,W]` float32 array normalized to `0..1`.

    Raises:
        RuntimeError: If the image cannot be read or the shape is unsupported.
    """
    if len(input_shape) != 4:
        raise RuntimeError(f"expected NCHW input_shape, got {input_shape}")
    out_h = int(input_shape[-2])
    out_w = int(input_shape[-1])
    if out_h != out_w:
        raise RuntimeError(f"YOLO26 calibration expects square input, got {out_h}x{out_w}")

    image = cv2.imread(path, cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"failed to read calibration image: {path}")
    src_h, src_w = image.shape[:2]
    ratio = min(out_w / src_w, out_h / src_h)
    new_w = int(round(src_w * ratio))
    new_h = int(round(src_h * ratio))
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    canvas = np.full((out_h, out_w, 3), 114, dtype=np.uint8)
    pad_x = (out_w - new_w) / 2.0
    pad_y = (out_h - new_h) / 2.0
    x0 = int(round(pad_x - 0.1))
    y0 = int(round(pad_y - 0.1))
    canvas[y0 : y0 + new_h, x0 : x0 + new_w] = resized

    rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
    nchw = np.transpose(rgb.astype(np.float32) / 255.0, (2, 0, 1))
    return np.ascontiguousarray(nchw)


def preprocess_impl(path_list: Sequence[str], input_parametr: dict) -> torch.Tensor:
    """Build one XSlim calibration batch for YOLO26.

    Args:
        path_list: Image paths in the current calibration batch.
        input_parametr: XSlim input configuration dictionary.

    Returns:
        A torch tensor with shape `[batch,3,H,W]`.
    """
    input_shape = input_parametr.get("input_shape")
    tensors = [_letterbox_rgb_nchw(path, input_shape) for path in path_list]
    return torch.from_numpy(np.stack(tensors, axis=0)).to(torch.float32)

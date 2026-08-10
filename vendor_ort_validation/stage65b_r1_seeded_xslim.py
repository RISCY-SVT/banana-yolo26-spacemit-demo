#!/usr/bin/env python3
"""Run the released XSlim CLI after explicitly seeding host RNGs."""

from __future__ import annotations

import os
import random
import runpy

import cv2
import numpy as np
import torch


def main() -> None:
    seed_text = os.environ.get("STAGE65B_R1_RANDOM_SEED")
    if seed_text is None:
        raise RuntimeError("STAGE65B_R1_RANDOM_SEED is required")
    seed = int(seed_text)
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    cv2.setRNGSeed(seed % (2**31))
    torch.use_deterministic_algorithms(True)
    runpy.run_module("xslim", run_name="__main__", alter_sys=True)


if __name__ == "__main__":
    main()

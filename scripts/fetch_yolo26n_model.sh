#!/usr/bin/env bash
# @file fetch_yolo26n_model.sh
# @brief Fetch the public Ultralytics YOLO26n checkpoint used for R&D bootstrap.
# @details The checkpoint is stored under `.deps/` and remains untracked. Export
# and decode validation must still prove semantics before any acceleration or
# INT8 work.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST_DIR="${DEST_DIR:-${ROOT_DIR}/.deps/models/yolo26}"
URL="${YOLO26N_URL:-https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26n.pt}"
SHA256_EXPECTED="${YOLO26N_SHA256:-9b09cc8bf347f0fc8a5f7657480587f25db09b34bf33b0652110fb03a8ad4fef}"
MODEL="${DEST_DIR}/yolo26n.pt"

mkdir -p "${DEST_DIR}"
curl -fL "${URL}" -o "${MODEL}"
echo "${SHA256_EXPECTED}  ${MODEL}" | sha256sum -c -

printf 'yolo26n_model=%s\n' "${MODEL}"

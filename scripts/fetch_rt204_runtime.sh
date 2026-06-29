#!/usr/bin/env bash
# @file fetch_rt204_runtime.sh
# @brief Fetch the public SpacemiT ONNX Runtime 2.0.4 RISC-V package for YOLO26 R&D.
# @details The archive is stored under `.deps/` and is intentionally ignored by
# git. This helper is for reproducibility only; it does not change production
# YOLO11 runtime policy.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST_DIR="${DEST_DIR:-${ROOT_DIR}/.deps/runtimes/rt204/downloads}"
UNPACK_DIR="${UNPACK_DIR:-${ROOT_DIR}/.deps/runtimes/rt204/unpacked}"
URL="${RT204_URL:-https://github.com/spacemit-com/onnxruntime/releases/download/2.0.4/spacemit-ort.riscv64.2.0.4.tar.gz}"
SHA256_EXPECTED="${RT204_SHA256:-bcf02bd12b8a1df969d6986658a9270c1121e5d58f5947d91ea5eba1bd6cd435}"
ARCHIVE="${DEST_DIR}/spacemit-ort.riscv64.2.0.4.tar.gz"

mkdir -p "${DEST_DIR}" "${UNPACK_DIR}"
curl -fL "${URL}" -o "${ARCHIVE}"
echo "${SHA256_EXPECTED}  ${ARCHIVE}" | sha256sum -c -
tar -xzf "${ARCHIVE}" -C "${UNPACK_DIR}"

printf 'rt204_archive=%s\n' "${ARCHIVE}"
printf 'rt204_unpack_dir=%s\n' "${UNPACK_DIR}/spacemit-ort.riscv64.2.0.4"

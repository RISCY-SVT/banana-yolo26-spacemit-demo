#!/usr/bin/env bash
## @file fetch_vendor_runtime.sh
## @brief Fetch and unpack the pinned public vendor runtime tarballs.
## @details The helper prefers local cache copies when available, validates
## SHA256, and stages each runtime under `third_party/vendor`.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOCK_FILE="${ROOT_DIR}/third_party_manifest/runtime.lock"
OUT_DIR="${ROOT_DIR}/third_party/vendor"
CACHE_DIR="${ROOT_DIR}/.deps/cache"

mkdir -p "${OUT_DIR}" "${CACHE_DIR}"

## @brief Fetch one locked runtime archive and unpack it into the vendor tree.
## @param name Archive base name and extracted runtime directory name.
## @param url Public archive URL for the runtime tarball.
## @param sha256_expected Expected archive SHA256.
fetch_one_runtime() {
  local name="$1"
  local url="$2"
  local sha256_expected="$3"
  local archive="${CACHE_DIR}/${name}.tar.gz"
  local extract_dir="${OUT_DIR}/${name}"

  if [[ -d "${extract_dir}" && -f "${extract_dir}/include/spacemit_ort_env.h" ]]; then
    echo "Vendor runtime already prepared: ${extract_dir}"
    return 0
  fi

  if [[ ! -f "${archive}" ]]; then
    if [[ -f "/data/SpacemiT/${name}.tar.gz" ]]; then
      cp -f "/data/SpacemiT/${name}.tar.gz" "${archive}"
    elif [[ -f "${ROOT_DIR}/.deps/cache/runtime_matrix/${name}.tar.gz" ]]; then
      cp -f "${ROOT_DIR}/.deps/cache/runtime_matrix/${name}.tar.gz" "${archive}"
    elif [[ -f "${ROOT_DIR}/.deps/runtimes/rt204/downloads/${name}.tar.gz" ]]; then
      cp -f "${ROOT_DIR}/.deps/runtimes/rt204/downloads/${name}.tar.gz" "${archive}"
    else
      curl -L --fail --output "${archive}" "${url}"
    fi
  fi

  echo "${sha256_expected}  ${archive}" | sha256sum -c

  rm -rf "${extract_dir}"
  tar -xf "${archive}" -C "${OUT_DIR}"
  test -f "${extract_dir}/include/spacemit_ort_env.h"
  echo "Prepared vendor runtime at ${extract_dir}"
}

mapfile -t RUNTIME_ENTRIES < <(awk -F= '$1=="runtime"{print $2}' "${LOCK_FILE}")

if [[ "${#RUNTIME_ENTRIES[@]}" -eq 0 ]]; then
  NAME="$(awk -F= '$1=="name"{print $2}' "${LOCK_FILE}")"
  URL="$(awk -F= '$1=="url"{print $2}' "${LOCK_FILE}")"
  SHA256_EXPECTED="$(awk -F= '$1=="sha256"{print $2}' "${LOCK_FILE}")"
  fetch_one_runtime "${NAME}" "${URL}" "${SHA256_EXPECTED}"
  exit 0
fi

for entry in "${RUNTIME_ENTRIES[@]}"; do
  IFS='|' read -r runtime_tag name url sha256_expected <<<"${entry}"
  [[ -n "${runtime_tag}" ]] || {
    echo "Malformed runtime entry in ${LOCK_FILE}: ${entry}" >&2
    exit 2
  }
  fetch_one_runtime "${name}" "${url}" "${sha256_expected}"
done

#!/usr/bin/env bash
set -uo pipefail

: "${BOARD_ROOT:?BOARD_ROOT is required}"
: "${STAGE63_ROOT:?STAGE63_ROOT is required}"

RUNTIME_LIB="${STAGE63_ROOT}/runtimes/rt206/lib"
RUNNER="${STAGE63_ROOT}/runners/runner_rt206"
MODEL_ROOT="${BOARD_ROOT}/models/tiny"
INPUT_ROOT="${BOARD_ROOT}/fixtures/tiny"
OUTPUT_ROOT="${BOARD_ROOT}/tiny-controls"
RAW="${OUTPUT_ROOT}/tiny_vendor_contract_matrix.raw.tsv"

mkdir -p \
  "${OUTPUT_ROOT}/outputs" \
  "${OUTPUT_ROOT}/logs" \
  "${OUTPUT_ROOT}/profiles" \
  "${OUTPUT_ROOT}/tmp" \
  "${OUTPUT_ROOT}/cache"

printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
  test_id provider cpus exit_code signal timed_out output_sha256 oracle_sha256 \
  exact session_created result_marker > "${RAW}"

signal_for_rc() {
  local rc="$1"
  if ((rc >= 128)); then
    printf '%s' "$((rc - 128))"
  else
    printf '0'
  fi
}

run_case() {
  local test_id="$1"
  local provider="$2"
  local cpus="$3"
  local suffix="$4"
  local model="${MODEL_ROOT}/${test_id}.onnx"
  local input="${INPUT_ROOT}/${test_id}.input.bin"
  local oracle="${INPUT_ROOT}/${test_id}.oracle.bin"
  local output="${OUTPUT_ROOT}/outputs/${test_id}__${provider}__${suffix}.bin"
  local log="${OUTPUT_ROOT}/logs/${test_id}__${provider}__${suffix}.log"
  local profile="${OUTPUT_ROOT}/profiles/${test_id}__${provider}__${suffix}"
  local rc timed_out signal output_hash oracle_hash exact session marker

  mkdir -p "${profile}"
  TMPDIR="${OUTPUT_ROOT}/tmp" \
  XDG_CACHE_HOME="${OUTPUT_ROOT}/cache" \
  LD_LIBRARY_PATH="${RUNTIME_LIB}" \
  SPACEMIT_EP_DUMP_SUBGRAPHS=1 \
  SPACEMIT_EP_DUMP_SUBGRAPHS_DIR="${profile}" \
    timeout --signal=TERM --kill-after=5 90 \
    taskset -c "${cpus}" "${RUNNER}" \
      --provider "${provider}" \
      --model "${model}" \
      --input "${input}" \
      --output "${output}" \
      --opt-level disable \
      --execution-mode sequential \
      --intra-threads 1 \
      --inter-threads 1 \
      --thread-spinning 0 \
      --log-severity 1 \
      --log-verbosity 1 \
      --warmup 0 \
      --runs 1 \
      --repeats 1 \
      --profile-prefix "${profile}/ort-profile" \
      >"${log}" 2>&1
  rc=$?

  timed_out=0
  if [[ "${rc}" -eq 124 || "${rc}" -eq 137 ]]; then
    timed_out=1
  fi
  signal="$(signal_for_rc "${rc}")"
  output_hash=""
  if [[ -f "${output}" ]]; then
    output_hash="$(sha256sum "${output}" | awk '{print $1}')"
  fi
  oracle_hash="$(sha256sum "${oracle}" | awk '{print $1}')"
  exact=0
  if [[ -f "${output}" ]] && cmp -s "${output}" "${oracle}"; then
    exact=1
  fi
  session=0
  if rg -q 'stage46_session status=created' "${log}"; then
    session=1
  fi
  marker="$(
    {
      rg -o 'stage46_result status=[^ ]+|stage46_session status=created' "${log}" ||
        true
    } | tail -1 | tr '\t ' '__'
  )"
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "${test_id}" "${provider}" "${cpus}" "${rc}" "${signal}" \
    "${timed_out}" "${output_hash}" "${oracle_hash}" "${exact}" \
    "${session}" "${marker}" >>"${RAW}"
}

supported=(
  c1_s8_conv_pc_explicit
  c2_s8_conv_pc_nonzero_zp
  c3_s8_conv_pc_no_kernel
  c5_s8_conv_pt_explicit
  m1_s8_matmul
  m2_s8_matmul_nonzero_zp
)
negative=(
  c4_u8_conv_pc_explicit
  m3_u8_matmul
)

for test_id in "${supported[@]}" "${negative[@]}"; do
  run_case "${test_id}" cpu 0-3 primary
done

for test_id in "${supported[@]}"; do
  run_case "${test_id}" spacemit 0-3 primary
done

for test_id in "${negative[@]}"; do
  run_case "${test_id}" spacemit 0 negative-once
done

for test_id in c1_s8_conv_pc_explicit m1_s8_matmul; do
  for cpus in 0 0-3 4 4-7 0-7; do
    suffix="affinity-${cpus//-/_}"
    run_case "${test_id}" spacemit "${cpus}" "${suffix}"
    last_rc="$(tail -1 "${RAW}" | cut -f4)"
    if [[ "${last_rc}" -ne 0 ]]; then
      break
    fi
  done
done

cat "${RAW}"

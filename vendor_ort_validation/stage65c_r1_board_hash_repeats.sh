#!/usr/bin/env bash
set -euo pipefail

: "${STAGE65C_R1_BOARD_ROOT:?STAGE65C_R1_BOARD_ROOT is required}"
[[ $# -eq 1 && ($1 == smoke || $1 == repeats) ]] || {
  printf 'usage: %s smoke|repeats\n' "$0" >&2
  exit 2
}

root=$STAGE65C_R1_BOARD_ROOT
mode=$1
selection="$root/manifests/selected_for_boundary_diagnostic.tsv"
frozen_status="$root/determinism/status.tsv"
runner="$root/bin/stage65c_r1_hashing_runner"
[[ -x $runner && -f $selection && -f $frozen_status ]]

loss_id=$(awk -F '\t' 'NR>1 && $1=="large-loss" {print $3; exit}' "$selection")
control_id=$(awk -F '\t' 'NR>1 && $1=="matched-control" {print $3; exit}' "$selection")
[[ -n $loss_id && -n $control_id ]]

inference_for() {
  case "$1" in
    B2) printf '%s' "$root/models/stage65b_r1_b2.inference.onnx" ;;
    A1) printf '%s' "$root/models/stage65b_r1_a1.inference.onnx" ;;
    *) return 2 ;;
  esac
}

manifest_sha() {
  local directory=$1
  for path in "$directory"/boundary-*.bin; do
    printf '%s  %s\n' "$(sha256sum "$path" | awk '{print $1}')" "$(basename "$path")"
  done | LC_ALL=C sort | sha256sum | awk '{print $1}'
}

expected_field() {
  local group=$1 model=$2 provider=$3 field=$4
  awk -F '\t' -v group="$group" -v model="$model" -v provider="$provider" -v field="$field" '
    NR == 1 {
      for (i = 1; i <= NF; ++i) column[$i] = i
      next
    }
    $1 == group && $3 == model && $4 == provider && $5 == "one-session" {
      print $column[field]
      found += 1
    }
    END {if (found != 1) exit 2}
  ' "$frozen_status"
}

run_surface() {
  local group=$1 image_id=$2 model=$3 provider=$4 runs=$5 output_root=$6 status=$7
  local inference output sample_header sample_hash unique_hashes output_sha boundary_sha
  local expected_output expected_boundary
  inference=$(inference_for "$model")
  output="$output_root/${group}-${image_id}/${model}-${provider}"
  [[ ! -e $output ]] || { printf 'refusing existing output: %s\n' "$output" >&2; return 2; }
  timeout --signal=TERM --kill-after=20s 1800s \
    "$root/bin/stage64_run_two_stage_board.sh" \
      --runner "$runner" \
      --runtime-lib "$root/runtime/lib" \
      --inference "$inference" \
      --tail "$root/models/stage65b_r1_a1.postprocess.onnx" \
      --input "$root/diagnostic-inputs/$(printf '%012d' "$image_id").nchw-f32.bin" \
      --output-dir "$output" \
      --provider "$provider" \
      --cpu-list 0-3 \
      --intra-threads 4 \
      --inter-threads 1 \
      --warmup 2 \
      --runs "$runs" \
      --repeats 1

  sample_header=$(head -1 "$output/samples.tsv")
  [[ $sample_header == $'repeat\trun\tinference_us\ttail_us\ttotal_us\toutput_fnv1a64' ]]
  [[ $(($(wc -l <"$output/samples.tsv") - 1)) -eq $runs ]]
  unique_hashes=$(tail -n +2 "$output/samples.tsv" | cut -f6 | LC_ALL=C sort -u | wc -l)
  [[ $unique_hashes -eq 1 ]]
  sample_hash=$(tail -n +2 "$output/samples.tsv" | cut -f6 | head -1)
  output_sha=$(sha256sum "$output/output.bin" | awk '{print $1}')
  boundary_sha=$(manifest_sha "$output/boundaries")
  expected_output=$(expected_field "$group" "$model" "$provider" output_sha256)
  expected_boundary=$(expected_field "$group" "$model" "$provider" boundary_manifest_sha256)
  [[ $output_sha == "$expected_output" ]]
  [[ $boundary_sha == "$expected_boundary" ]]
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\tpass\n' \
    "$group" "$image_id" "$model" "$provider" "$runs" "$unique_hashes" \
    "$sample_hash" "$output_sha" "$boundary_sha" "$(sha256sum "$output/samples.tsv" | awk '{print $1}')" \
    >>"$status"
}

if [[ $mode == smoke ]]; then
  output_root="$root/hashing-smoke"
  status="$output_root/status.tsv"
  [[ ! -e $output_root ]]
  mkdir -p "$output_root"
  printf 'case_group\timage_id\tmodel\tprovider\truns\tunique_sample_hashes\tsample_output_fnv1a64\toutput_sha256\tboundary_manifest_sha256\tsamples_sha256\tstatus\n' >"$status"
  for model in B2 A1; do
    for provider in cpu spacemit; do
      run_surface large-loss "$loss_id" "$model" "$provider" 1 "$output_root" "$status"
    done
  done
else
  output_root="$root/determinism-hash"
  status="$output_root/status.tsv"
  [[ ! -e $output_root ]]
  mkdir -p "$output_root"
  printf 'case_group\timage_id\tmodel\tprovider\truns\tunique_sample_hashes\tsample_output_fnv1a64\toutput_sha256\tboundary_manifest_sha256\tsamples_sha256\tstatus\n' >"$status"
  for case_spec in "large-loss:$loss_id" "matched-control:$control_id"; do
    group=${case_spec%%:*}
    image_id=${case_spec#*:}
    for model in B2 A1; do
      for provider in cpu spacemit; do
        run_surface "$group" "$image_id" "$model" "$provider" 100 "$output_root" "$status"
      done
    done
  done
fi

printf 'stage65c_r1_board_hash_repeats mode=%s status=pass\n' "$mode"

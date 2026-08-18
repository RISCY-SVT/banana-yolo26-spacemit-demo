#!/usr/bin/env bash
set -euo pipefail

: "${STAGE65C_R1_BOARD_ROOT:?STAGE65C_R1_BOARD_ROOT is required}"
root=$STAGE65C_R1_BOARD_ROOT
selection="$root/manifests/selected_for_boundary_diagnostic.tsv"
status="$root/determinism/status.tsv"
printf 'case_group\timage_id\tmodel\tprovider\tmode\trecreation\texit_code\toutput_sha256\tboundary_manifest_sha256\tsamples_sha256\n' >"$status"

loss_id=$(awk -F '\t' 'NR>1 && $1=="large-loss" {print $3; exit}' "$selection")
control_id=$(awk -F '\t' 'NR>1 && $1=="matched-control" {print $3; exit}' "$selection")
[[ -n $loss_id && -n $control_id ]] || { printf 'loss/control selections missing\n' >&2; exit 2; }

inference_for() {
  case "$1" in
    B2) printf '%s' "$root/models/stage65b_r1_b2.inference.onnx" ;;
    A1) printf '%s' "$root/models/stage65b_r1_a1.inference.onnx" ;;
  esac
}

run_one() {
  local case_group=$1 image_id=$2 model=$3 provider=$4 mode=$5 recreation=$6 warmup=$7 runs=$8
  local input="$root/diagnostic-inputs/$(printf '%012d' "$image_id").nchw-f32.bin"
  local output="$root/determinism/${case_group}-${image_id}/${model}-${provider}/${mode}-${recreation}"
  local inference
  inference=$(inference_for "$model")
  [[ ! -e $output ]] || { printf 'refusing existing output: %s\n' "$output" >&2; return 2; }
  timeout --signal=TERM --kill-after=20s 1800s \
    "$root/bin/stage64_run_two_stage_board.sh" \
      --runner "$root/bin/stage64_two_stage_runner" \
      --runtime-lib "$root/runtime/lib" \
      --inference "$inference" \
      --tail "$root/models/stage65b_r1_a1.postprocess.onnx" \
      --input "$input" \
      --output-dir "$output" \
      --provider "$provider" \
      --cpu-list 0-3 \
      --intra-threads 4 \
      --inter-threads 1 \
      --warmup "$warmup" \
      --runs "$runs" \
      --repeats 1
  local output_sha boundary_sha samples_sha
  output_sha=$(sha256sum "$output/output.bin" | awk '{print $1}')
  boundary_sha=$(
    for path in "$output"/boundaries/*.bin; do
      printf '%s  %s\n' "$(sha256sum "$path" | awk '{print $1}')" "$(basename "$path")"
    done | LC_ALL=C sort | sha256sum | awk '{print $1}'
  )
  samples_sha=$(sha256sum "$output/samples.tsv" | awk '{print $1}')
  printf '%s\t%s\t%s\t%s\t%s\t%s\t0\t%s\t%s\t%s\n' \
    "$case_group" "$image_id" "$model" "$provider" "$mode" "$recreation" \
    "$output_sha" "$boundary_sha" "$samples_sha" >>"$status"
}

for case_spec in "large-loss:$loss_id" "matched-control:$control_id"; do
  case_group=${case_spec%%:*}
  image_id=${case_spec#*:}
  for model in B2 A1; do
    for provider in cpu spacemit; do
      run_one "$case_group" "$image_id" "$model" "$provider" one-session 0 2 100
      for recreation in $(seq 1 10); do
        run_one "$case_group" "$image_id" "$model" "$provider" recreate "$recreation" 0 1
      done
    done
  done
done

printf 'stage65c_r1_board_determinism status=pass\n'

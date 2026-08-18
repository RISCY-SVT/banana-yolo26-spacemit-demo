#!/usr/bin/env bash
set -euo pipefail

: "${STAGE65C_R1_BOARD_ROOT:?STAGE65C_R1_BOARD_ROOT is required}"
root=$STAGE65C_R1_BOARD_ROOT
selection="$root/manifests/selected_for_boundary_diagnostic.tsv"
[[ -f $selection ]] || { printf 'missing selection manifest\n' >&2; exit 2; }
status="$root/boundaries/status.tsv"
printf 'image_id\tmodel\tprovider\texit_code\toutput_sha256\tboundary_manifest_sha256\ttail_replay_sha256\n' >"$status"

inference_for() {
  case "$1" in
    B2) printf '%s' "$root/models/stage65b_r1_b2.inference.onnx" ;;
    A1) printf '%s' "$root/models/stage65b_r1_a1.inference.onnx" ;;
    *) return 2 ;;
  esac
}

tail -n +2 "$selection" | while IFS=$'\t' read -r group rank image_id rest; do
  input="$root/diagnostic-inputs/$(printf '%012d' "$image_id").nchw-f32.bin"
  [[ -f $input ]] || { printf 'missing input for %s\n' "$image_id" >&2; exit 2; }
  for model in B2 A1; do
    inference=$(inference_for "$model")
    for provider in cpu spacemit; do
      output="$root/boundaries/$image_id/${model}-${provider}"
      [[ ! -e $output ]] || { printf 'refusing existing output: %s\n' "$output" >&2; exit 2; }
      mkdir -p "$output"
      rc=0
      set +e
      timeout --signal=TERM --kill-after=20s 900s \
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
          --warmup 0 \
          --runs 1 \
          --repeats 1
      rc=$?
      set -e
      [[ $rc -eq 0 ]] || exit "$rc"
      env LD_LIBRARY_PATH="$root/runtime/lib" taskset -c 0-3 \
        "$root/bin/stage65c_r1_tail_replay" \
          --tail-model "$root/models/stage65b_r1_a1.postprocess.onnx" \
          --boundary-dir "$output/boundaries" \
          --output "$output/tail-replay.bin" \
          --threads 4 \
          --runs 2 >"$output/tail-replay.log" 2>&1
      output_sha=$(sha256sum "$output/output.bin" | awk '{print $1}')
      boundary_sha=$(sha256sum "$output"/boundaries/*.bin | LC_ALL=C sort | sha256sum | awk '{print $1}')
      replay_sha=$(sha256sum "$output/tail-replay.bin" | awk '{print $1}')
      printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "$image_id" "$model" "$provider" "$rc" "$output_sha" "$boundary_sha" "$replay_sha" >>"$status"
    done
  done
done

printf 'stage65c_r1_board_boundaries status=pass\n'

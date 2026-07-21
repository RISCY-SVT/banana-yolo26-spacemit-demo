#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "usage: $0 STAGE_ROOT RESOLUTION [RESOLUTION ...]" >&2
  exit 2
fi

stage_root=$1
shift
dump=$stage_root/bin/stage61_boundary_dump
output=$stage_root/exactness/full-fixture-parity
ram=/dev/shm/y26-stage61-exactness-$$
if [[ -n ${Y26_STAGE61_FIXTURES:-} ]]; then
  read -r -a fixtures <<<"$Y26_STAGE61_FIXTURES"
else
  fixtures=(F0 F1 F2 F3 F4 F5 F6 F7 bus canonical zidane)
fi
mkdir -p "$output" "$ram"
trap 'rm -rf "$ram"' EXIT

[[ -x $dump ]] || { echo "missing Stage61 boundary dumper: $dump" >&2; exit 1; }
if [[ ${Y26_STAGE61_APPEND:-0} != 1 || ! -s $output/summary.tsv ]]; then
  printf 'resolution\tfixture\tfiles\thost_vs_board_scalar\thost_vs_board_optimized\tboard_scalar_vs_optimized\tscalar_elapsed_s\toptimized_elapsed_s\tscalar_summary\toptimized_summary\n' \
    >"$output/summary.tsv"
fi

export Y26_STAGE54_E2C3=1
export Y26_STAGE55_E2C4=1
export Y26_STAGE55_DENSE_FAMILY_A=1
export Y26_STAGE54_DIRECT_1X1=1
export Y26_STAGE54_DENSE_PACK_RVV=1
export Y26_STAGE53_FUSED_LUT=1
export Y26_STAGE54_DEPTHWISE_V2=1
export Y26_STAGE54_DEPTHWISE_X2=1
export Y26_STAGE54_DEPTHWISE_BORDER_V2=1
export Y26_STAGE54_INPUT_RVV_V2=1
export Y26_STAGE54_INPUT_COMPACT_C3=1
export Y26_STAGE54_LUT2_RVV=1
export Y26_STAGE54_ATTENTION_V2=1
export Y26_STAGE56_HEAD_PRODUCER_REDUCTION=1
export Y26_STAGE56_ATTENTION_DIRECT_PACK=1
export Y26_STAGE57_E2C5=1
export Y26_STAGE57_ATTENTION_MATMUL_C8=1
export Y26_STAGE57_RGB_COPY_RVV=1
export Y26_STAGE61_ATTENTION_NTAIL=1

hash_tree() {
  local directory=$1 destination=$2
  (
    cd "$directory"
    find . -maxdepth 1 -type f -printf '%P\0' | sort -z | xargs -0 sha256sum
  ) >"$destination"
}

require_hash_match() {
  local expected=$1 actual=$2 label=$3
  if ! cmp -s "$expected" "$actual"; then
    echo "exactness mismatch: $label" >&2
    diff -u "$expected" "$actual" | sed -n '1,80p' >&2 || true
    return 1
  fi
}

for resolution in "$@"; do
  package=$stage_root/packages/r$resolution
  [[ -r $package/asset_hashes.tsv ]] || {
    echo "missing Stage61 package for R$resolution" >&2
    exit 1
  }
  for fixture in "${fixtures[@]}"; do
    if [[ $fixture == bus || $fixture == canonical || $fixture == zidane ]]; then
      input=$stage_root/fixtures/r$resolution/${fixture}_r${resolution}_nchw_f32.bin
    else
      input=$stage_root/fixtures/r$resolution/${fixture}_nchw_f32.bin
    fi
    host_hash=$stage_root/oracles/host-scalar/r$resolution/$fixture.sha256
    [[ -r $input && -r $host_hash ]] || {
      echo "missing Stage61 fixture/oracle: R$resolution $fixture" >&2
      exit 1
    }
    prefix=$ram/r${resolution}_${fixture}
    scalar_dir=$prefix.scalar
    optimized_dir=$prefix.optimized
    mkdir -p "$scalar_dir" "$optimized_dir"

    scalar_begin=$(date +%s%N)
    timeout 300 "$dump" "$package" "$input" "$scalar_dir" --scalar \
      >"$prefix.scalar.stdout" 2>"$prefix.scalar.stderr"
    scalar_end=$(date +%s%N)
    optimized_begin=$(date +%s%N)
    timeout 300 "$dump" "$package" "$input" "$optimized_dir" \
      >"$prefix.optimized.stdout" 2>"$prefix.optimized.stderr"
    optimized_end=$(date +%s%N)

    hash_tree "$scalar_dir" "$prefix.scalar.sha256"
    hash_tree "$optimized_dir" "$prefix.optimized.sha256"
    require_hash_match "$host_hash" "$prefix.scalar.sha256" \
      "R$resolution $fixture host vs board scalar"
    require_hash_match "$host_hash" "$prefix.optimized.sha256" \
      "R$resolution $fixture host vs board optimized"
    require_hash_match "$prefix.scalar.sha256" "$prefix.optimized.sha256" \
      "R$resolution $fixture board scalar vs optimized"
    files=$(wc -l <"$host_hash")
    [[ $files -eq 216 ]] || {
      echo "unexpected oracle file count: R$resolution $fixture has $files" >&2
      exit 1
    }
    scalar_summary=$(tr '\t\n' '  ' <"$prefix.scalar.stdout")
    optimized_summary=$(tr '\t\n' '  ' <"$prefix.optimized.stdout")
    scalar_elapsed=$(awk -v a="$scalar_begin" -v b="$scalar_end" 'BEGIN {printf "%.6f", (b-a)/1e9}')
    optimized_elapsed=$(awk -v a="$optimized_begin" -v b="$optimized_end" 'BEGIN {printf "%.6f", (b-a)/1e9}')
    printf '%s\t%s\t%s\texact\texact\texact\t%s\t%s\t%s\t%s\n' \
      "$resolution" "$fixture" "$files" "$scalar_elapsed" "$optimized_elapsed" \
      "$scalar_summary" "$optimized_summary" >>"$output/summary.tsv"
    cp "$prefix.scalar.sha256" "$output/r${resolution}_${fixture}_scalar.sha256"
    cp "$prefix.optimized.sha256" "$output/r${resolution}_${fixture}_optimized.sha256"
    cp "$prefix.scalar.stderr" "$output/r${resolution}_${fixture}_scalar.stderr"
    cp "$prefix.optimized.stderr" "$output/r${resolution}_${fixture}_optimized.stderr"
    rm -rf "$scalar_dir" "$optimized_dir"
  done
done

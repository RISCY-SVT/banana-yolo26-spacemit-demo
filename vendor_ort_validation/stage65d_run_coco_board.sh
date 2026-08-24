#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  stage65d_run_coco_board.sh \
    --runner FILE --runtime-lib DIR --opencv-lib DIR \
    --inference FILE --tail FILE --images DIR --output-dir DIR \
    --provider cpu|spacemit [--cpu-list 0-3] [--threads 4] \
    [--confidence 0.001] [--limit 0]

The output directory must be a new or empty directory under /data.
EOF
}

runner=
runtime_lib=
opencv_lib=
inference=
tail=
images=
output_dir=
provider=
cpu_list=0-3
threads=4
confidence=0.001
limit=0

while (($#)); do
  case "$1" in
    --runner) runner=${2:?}; shift 2 ;;
    --runtime-lib) runtime_lib=${2:?}; shift 2 ;;
    --opencv-lib) opencv_lib=${2:?}; shift 2 ;;
    --inference) inference=${2:?}; shift 2 ;;
    --tail) tail=${2:?}; shift 2 ;;
    --images) images=${2:?}; shift 2 ;;
    --output-dir) output_dir=${2:?}; shift 2 ;;
    --provider) provider=${2:?}; shift 2 ;;
    --cpu-list) cpu_list=${2:?}; shift 2 ;;
    --threads) threads=${2:?}; shift 2 ;;
    --confidence) confidence=${2:?}; shift 2 ;;
    --limit) limit=${2:?}; shift 2 ;;
    --help|-h) usage; exit 0 ;;
    *) printf 'unknown argument: %s\n' "$1" >&2; usage >&2; exit 2 ;;
  esac
done

for required in runner runtime_lib opencv_lib inference tail images output_dir provider; do
  if [[ -z ${!required} ]]; then
    printf 'missing --%s\n' "${required//_/-}" >&2
    usage >&2
    exit 2
  fi
done
if [[ $provider != cpu && $provider != spacemit ]]; then
  printf 'invalid provider: %s\n' "$provider" >&2
  exit 2
fi
if [[ $output_dir != /data/* ]]; then
  printf 'output directory must be under /data: %s\n' "$output_dir" >&2
  exit 2
fi
for file in "$runner" "$inference" "$tail"; do
  [[ -f $file ]] || { printf 'missing file: %s\n' "$file" >&2; exit 2; }
done
for directory in "$runtime_lib" "$opencv_lib" "$images"; do
  [[ -d $directory ]] || {
    printf 'missing directory: %s\n' "$directory" >&2
    exit 2
  }
done
if [[ -d $output_dir ]] && find "$output_dir" -mindepth 1 -print -quit | grep -q .; then
  printf 'output directory is not empty: %s\n' "$output_dir" >&2
  exit 2
fi

mkdir -p "$output_dir/tmp" "$output_dir/cache"
cat /proc/sys/kernel/random/boot_id >"$output_dir/boot_id.txt"
find /sys/devices/system/cpu/cpufreq -name scaling_cur_freq -print0 \
  | sort -z \
  | xargs -0 -r -n1 sh -c 'printf "%s\t" "$1"; cat "$1"' sh \
  >"$output_dir/frequency_before.tsv"
{
  for temp_path in /sys/class/thermal/thermal_zone*/temp; do
    [[ -r "$temp_path" ]] || continue
    printf '%s\t' "$temp_path"
    cat "$temp_path"
  done
} >"$output_dir/thermal_before.tsv"

command=(
  "$runner"
  --inference-model "$inference"
  --tail-model "$tail"
  --images "$images"
  --output "$output_dir/predictions.json"
  --timing-tsv "$output_dir/timing.tsv"
  --provider "$provider"
  --threads "$threads"
  --conf "$confidence"
  --limit "$limit"
)

{
  printf 'provider=%s\n' "$provider"
  printf 'cpu_list=%s\n' "$cpu_list"
  printf 'threads=%s\n' "$threads"
  printf 'confidence=%s\n' "$confidence"
  printf 'limit=%s\n' "$limit"
  printf 'runtime_lib=%s\n' "$runtime_lib"
  printf 'opencv_lib=%s\n' "$opencv_lib"
  printf 'inference=%s\n' "$inference"
  printf 'tail=%s\n' "$tail"
  printf 'images=%s\n' "$images"
  printf 'command='
  printf '%q ' "${command[@]}"
  printf '\n'
} >"$output_dir/effective-config.txt"

sha256sum "$runner" "$inference" "$tail" >"$output_dir/input-sha256.txt"
env LD_LIBRARY_PATH="$runtime_lib:$opencv_lib" \
  ldd "$runner" >"$output_dir/ldd.txt"

(
  cd "$output_dir"
  env \
    LD_LIBRARY_PATH="$runtime_lib:$opencv_lib" \
    TMPDIR="$output_dir/tmp" \
    XDG_CACHE_HOME="$output_dir/cache" \
    /usr/bin/time -v -o "$output_dir/time-v.txt" \
    taskset -c "$cpu_list" \
    "${command[@]}"
) >"$output_dir/run.log" 2>&1

sha256sum "$output_dir/predictions.json" "$output_dir/timing.tsv" \
  >"$output_dir/output-sha256.txt"
find /sys/devices/system/cpu/cpufreq -name scaling_cur_freq -print0 \
  | sort -z \
  | xargs -0 -r -n1 sh -c 'printf "%s\t" "$1"; cat "$1"' sh \
  >"$output_dir/frequency_after.tsv"
{
  for temp_path in /sys/class/thermal/thermal_zone*/temp; do
    [[ -r "$temp_path" ]] || continue
    printf '%s\t' "$temp_path"
    cat "$temp_path"
  done
} >"$output_dir/thermal_after.tsv"

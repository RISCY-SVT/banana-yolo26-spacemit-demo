#!/usr/bin/env bash
set -euo pipefail

root=${1:-/data/k1x-yolo26-int8-executor}
input=${2:-"$root/fixtures/bus_640_nchw_f32.bin"}
output=${3:-"$root/outputs/benchmark.json"}
profile=${4:-compatibility}
stage56_cgroup=
stage56_child=
stage56_gate=

operator_profile="$root/config/k1x-int8-executor-stage56.env"
if [[ ! -r "$operator_profile" ]]; then
  operator_profile="$root/config/k1x-int8-executor-stage55.env"
fi
if [[ ! -r "$operator_profile" ]]; then
  operator_profile="$root/config/k1x-int8-executor-stage54.env"
fi
if [[ -r "$operator_profile" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "$operator_profile"
  set +a
fi
case "$profile" in
  compatibility)
    unset Y26_STAGE53_SPIN_POOL Y26_STAGE55_FRAME_GATED_SPIN
    ;;
  low-latency)
    export Y26_STAGE53_SPIN_POOL=1
    export Y26_STAGE55_FRAME_GATED_SPIN=1
    ;;
  low-latency-dedicated)
    export Y26_STAGE53_SPIN_POOL=1
    export Y26_STAGE55_FRAME_GATED_SPIN=1
    state_dir="$root/state/stage56-o2"
    cleanup_stage56_profile() {
      if [[ -n $stage56_child ]]; then
        kill "$stage56_child" 2>/dev/null || true
        wait "$stage56_child" 2>/dev/null || true
      fi
      [[ -z $stage56_gate ]] || rm -f "$stage56_gate"
      "$root/scripts/stage56-system-profile.sh" restore "$state_dir"
    }
    trap cleanup_stage56_profile EXIT INT TERM
    "$root/scripts/stage56-system-profile.sh" apply "$state_dir"
    stage56_cgroup=/sys/fs/cgroup/y26-stage56-inference
    ;;
  *)
    echo "unknown profile: $profile (expected compatibility, low-latency, or low-latency-dedicated)" >&2
    exit 2
    ;;
esac

mkdir -p "$(dirname "$output")"
command=("$root/bin/yolo26_k1x_int8" \
  --package "$root/package" --image "$input" --input-mode preprocessed-f32 \
  --output-json "$output" --threads 4 --pin 0-3 --scheduler safe \
  --warmup 10 --runs 100 --repeats 5 --verify --benchmark)

if [[ -z $stage56_cgroup ]]; then
  "${command[@]}"
  exit
fi

# Keep the rollback controller outside the isolated cgroup. The child blocks on
# a FIFO until its PID is moved, then execs the benchmark with no helper left in
# the cgroup when inference completes.
stage56_gate="$state_dir/executor-start.$$"
rm -f "$stage56_gate"
mkfifo "$stage56_gate"
bash -c 'gate=$1; shift; IFS= read -r _ <"$gate"; exec "$@"' \
  bash "$stage56_gate" "${command[@]}" &
stage56_child=$!
printf '%s\n' "$stage56_child" | sudo -n tee "$stage56_cgroup/cgroup.procs" >/dev/null
printf 'run\n' >"$stage56_gate"
set +e
wait "$stage56_child"
status=$?
set -e
stage56_child=
rm -f "$stage56_gate"
stage56_gate=
exit "$status"

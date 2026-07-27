#!/usr/bin/env bash
set -euo pipefail

: "${STAGE63_BOARD_ROOT:?set STAGE63_BOARD_ROOT}"

root=$STAGE63_BOARD_ROOT
out_root="$root/performance/resource-probe"
mkdir -p "$out_root/raw" "$out_root/outputs"
summary="$out_root/runtime_resource_inventory.tsv"
printf 'arm\tprovider\tsurface\tpid_samples\tmax_threads\tmax_fds\tmax_rss_kib\texit_code\toutput_sha256\tstatus\n' >"$summary"

run_probe() {
    local provider=$1
    local surface=$2
    local arm="rt206_${provider}_${surface}_resource_probe"
    local model
    case "$surface" in
        fp32) model="$root/models/yolo26n_640_e2e_fp32.onnx" ;;
        fp16) model="$root/models/yolo26n_640_e2e_native_fp16_body_headfp32_keep_io.onnx" ;;
        int8) model="$root/models/manual_e2e_rep_conv_matmul_qdq.onnx" ;;
        *) printf 'unsupported surface: %s\n' "$surface" >&2; return 2 ;;
    esac

    local runner="$root/runners/runner_rt206"
    local runtime_lib="$root/runtimes/rt206/lib"
    local log="$out_root/raw/${arm}.log"
    local output="$out_root/outputs/${arm}.bin"
    local pid watchdog rc=0 samples=0 max_threads=0 max_fds=0 max_rss=0
    cd "$out_root"
    unset SPACEMIT_EP_DUMP_SUBGRAPHS

    taskset -c 0-3 env LD_LIBRARY_PATH="$runtime_lib" \
      "$runner" --provider "$provider" --model "$model" \
      --input "$root/fixtures/preprocessed/images_F0_f32.bin" \
      --output "$output" --opt-level all --execution-mode sequential \
      --intra-threads 4 --inter-threads 1 --thread-spinning 0 \
      --warmup 2 --runs 1 --repeats 30 >"$log" 2>&1 &
    pid=$!
    (
        sleep 120
        kill -TERM "$pid" 2>/dev/null || true
        sleep 5
        kill -KILL "$pid" 2>/dev/null || true
    ) &
    watchdog=$!

    while kill -0 "$pid" 2>/dev/null; do
        if [[ -r "/proc/$pid/status" ]]; then
            local threads rss fds
            threads=$(awk '/^Threads:/{print $2}' "/proc/$pid/status" 2>/dev/null || true)
            rss=$(awk '/^VmRSS:/{print $2}' "/proc/$pid/status" 2>/dev/null || true)
            fds=$(find "/proc/$pid/fd" -mindepth 1 -maxdepth 1 2>/dev/null | wc -l) ||
                fds=0
            [[ $threads =~ ^[0-9]+$ ]] || threads=0
            [[ $rss =~ ^[0-9]+$ ]] || rss=0
            ((threads > max_threads)) && max_threads=$threads
            ((fds > max_fds)) && max_fds=$fds
            ((rss > max_rss)) && max_rss=$rss
            samples=$((samples + 1))
        fi
        sleep 0.1
    done
    wait "$pid" || rc=$?
    kill "$watchdog" 2>/dev/null || true
    wait "$watchdog" 2>/dev/null || true

    local output_sha=missing status=fail
    if ((rc == 0)) && [[ -f $output ]]; then
        output_sha=$(sha256sum "$output" | awk '{print $1}')
        status=pass
    fi
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
      "$arm" "$provider" "$surface" "$samples" "$max_threads" "$max_fds" \
      "$max_rss" "$rc" "$output_sha" "$status" >>"$summary"
}

run_probe cpu int8
run_probe cpu fp32
run_probe cpu fp16
run_probe spacemit fp32
run_probe spacemit fp16

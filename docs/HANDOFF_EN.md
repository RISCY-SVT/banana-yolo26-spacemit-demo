# YOLO26 K1X INT8 Executor 0.9.3

## What This Is

This release is an exact, standalone C/C++ executor for one frozen model on the
Banana-Pi BPI-F3 (SpacemiT K1X). It runs the complete YOLO26n 640 graph with the
`K1X_INT8_V1` integer contract and returns `1x300x6` detections. The measured
path contains no ONNX Runtime call, Python call, per-inference file I/O, float
Q/DQ materialization, or dynamic graph dispatch.

Handoff label: **optimized-engineering-handoff-ready, not-production-certified**.

## Exact Support Contract

| Item | Required value |
|---|---|
| Board | Banana-Pi BPI-F3 / SpacemiT K1X |
| Model | `manual_e2e_rep_conv_matmul_qdq.onnx` |
| Model SHA-256 | `30a94e4738606673b5e0a73499cbc977167f046f8fa8637d6040ce744f429c0c` |
| Integer contract | `K1X_INT8_V1` |
| Graph profile | `K1X_INT8_V1_YOLO26N_640_FULL_GRAPH_001` |
| Package manifest SHA-256 | `fab4a72cf524ce0a205ceca0384144f2eee7bc79dff3f4db8b7208614e8407be` |
| Input | `1x3x640x640` float32 NCHW in `[0,1]`, or 640x640 RGB8 |
| Output | 300 rows of `[x1,y1,x2,y2,score,class]` float32 |

Other models, resolutions, graph profiles, boards, layouts, and quantization
contracts are rejected or unsupported. Letterbox geometry remains the caller's
responsibility unless the image-mode CLI is used with OpenCV support.

## Dataflow

```text
float32 NCHW or RGB8
        |
        v
exact RNE quantize / compact RGB stem
        |
        v
NCHWc8 arena -> IME dense + RVV E2c5 -> depthwise / LUT / concat
        |                                     |
        +------ model10/model22 attention ----+
        |
        v
producer-direct head reduction -> deterministic top-300 -> 1x300x6

CPU0-3: IME workers       CPU4: controller       CPU5-7: housekeeping only
```

## Ten-Minute Start

The release is installed under NVMe `/data`:

```bash
RELEASE=/data/releases/banana-yolo26-k1x-int8-executor/0.9.3-stage60m-maintenance-runtime
PACKAGE=$RELEASE/package

$RELEASE/bin/y26_k1x_healthcheck \
  "$PACKAGE" \
  fab4a72cf524ce0a205ceca0384144f2eee7bc79dff3f4db8b7208614e8407be \
  "$RELEASE/fixtures/bus_640_nchw_f32.bin" \
  0xd43f5e018b415631

taskset -c 0-4 $RELEASE/bin/yolo26_k1x_int8 \
  --package "$PACKAGE" \
  --image "$RELEASE/fixtures/bus_640_nchw_f32.bin" \
  --input-mode preprocessed-f32 \
  --profile compatibility \
  --expected-manifest-sha256 \
    fab4a72cf524ce0a205ceca0384144f2eee7bc79dff3f4db8b7208614e8407be \
  --verify-known-fixture --expected-output-hash 0xd43f5e018b415631 \
  --verify-determinism --output-json /data/y26-output.json
```

For the dedicated low-latency profile:

```bash
sudo -n true
$RELEASE/scripts/o2-system-profile.sh run -- \
  taskset -c 0-4 $RELEASE/bin/yolo26_k1x_int8 \
  --package "$PACKAGE" --image "$RELEASE/fixtures/bus_640_nchw_f32.bin" \
  --input-mode preprocessed-f32 --profile low-latency-dedicated \
  --expected-manifest-sha256 fab4a72cf524ce0a205ceca0384144f2eee7bc79dff3f4db8b7208614e8407be \
  --output-json /data/y26-output.json
```

The wrapper restores system placement on normal exit, command failure, timeout,
`INT`, `TERM`, and `HUP`. Use `restore-stale` after a wrapper `SIGKILL`.

## Profiles

| Profile | Wake policy | System placement | Use |
|---|---|---|---|
| compatibility | condition variable | original system | integration and shared boards |
| low-latency | frame-gated spin | original system | dedicated process, no OS changes |
| low-latency-dedicated | frame-gated spin | reversible O2 | measured dedicated-board surface |

No stage-numbered environment variable is required. Research builds retain
candidate overrides, but the installed release ignores them for operator choice.

## C Integration

Use one prepared handle per serialized stream. Independent handles are supported;
the same handle rejects concurrent calls with `Y26_STATUS_BUSY`.

```c
y26_executor_options options;
y26_executor_options_init(&options);
options.wake_policy = Y26_WAKE_FRAME_GATED_SPIN;

y26_executor *executor = y26_executor_create();
y26_status status = y26_executor_prepare(
    executor, package_dir,
    "fab4a72cf524ce0a205ceca0384144f2eee7bc79dff3f4db8b7208614e8407be",
    &options);
if (status == Y26_STATUS_OK) {
    status = y26_executor_run_preprocessed(
        executor, input, Y26_K1X_EXECUTOR_INPUT_ELEMENTS,
        output, Y26_K1X_EXECUTOR_OUTPUT_ELEMENTS, NULL);
}
if (status != Y26_STATUS_OK)
    fprintf(stderr, "%s: %s\n", y26_status_string(status),
            y26_executor_last_error(executor));
y26_executor_destroy(executor);
```

See `INTEGRATION_GUIDE.md` and `examples/c_api_consumer.c` for complete builds
using CMake or pkg-config.

## Correctness and Performance

The release is byte-exact at all 215 integer boundaries, F0-F7, bus, and Zidane.
COCO val2017 completed 5000/5000 at mAP50-95 `0.3707408944391919`.
The canonical executor and camera timing rows are in `PERFORMANCE_AND_ACCURACY.md`; fixed
input, real corpus, RGB, serial pipeline, and double-buffer throughput are
separate surfaces and must not be mixed.

## Release Tree

| Path | Contents |
|---|---|
| `bin/` | release CLI and healthcheck |
| `lib/`, `include/` | ABI1 shared/static libraries, CMake/pkg-config metadata, public C header |
| `package/`, `fixtures/` | frozen full-graph package and known exact input |
| `config/`, `scripts/` | safe profile configuration, build/deploy/smoke/O2 helpers |
| `docs/`, `examples/` | handoff guides and compilable C consumer |
| `licenses/`, `sbom/` | notices and machine-readable component inventory |
| `outputs/` | known-fixture, COCO, and separately labeled performance summaries |

`release_manifest.json`, `release_sha256.txt`, and `SHA256SUMS` identify and
verify the complete tree. Research candidates and stage environment files are
not part of this primary handoff root.

## Operational Rules

- Keep binaries, packages, logs, and outputs on NVMe `/data`.
- IME may execute only on CPU0-3. CPU4 is the controller; CPU5-7 are not IME CPUs.
- Keep the original boot profile. O2 makes no persistent boot, sysctl, THP,
  frequency, storage, or real-time scheduling change.
- Validate the package manifest hash explicitly. This prevents accidental model
  or stale-package mixing; it is not a hostile security boundary.
- Do not share one handle concurrently.

## Removal

Stop active processes, restore O2 if necessary, and remove only the versioned
release directory:

```bash
$RELEASE/scripts/o2-system-profile.sh restore-stale
rm -rf -- "$RELEASE"
```

The release does not install into `/usr` and removal does not touch other files.

## Limits

This is not a 20 FPS claim, a camera service, a vendor runtime plugin, or a
production certification. Stage58 is release maintenance on the frozen graph. Q31,
different models/resolutions, training, students, and co-design require a new
branch/project and separate authorization.

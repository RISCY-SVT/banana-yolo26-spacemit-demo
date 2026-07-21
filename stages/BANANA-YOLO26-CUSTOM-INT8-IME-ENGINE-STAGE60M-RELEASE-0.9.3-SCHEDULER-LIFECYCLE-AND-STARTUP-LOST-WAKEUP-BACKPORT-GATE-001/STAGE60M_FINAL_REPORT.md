# Stage60M Final Report

Classification: `stage60m-release-0.9.3-scheduler-maintenance-pass`.

Stage60M is a narrow 0.9.3 maintenance backport from frozen R640 release
baseline `175c1d939cc93fba0e730dba3f1281704e8f25b9`. It imports exactly two
non-arithmetic repairs proven in Stage60 and does not import resolution,
profile, quantization, graph, model, layout, dispatch, camera-policy, or ABI
changes.

## Backport Scope

- `stage49_persistent_slice.cpp` serializes frame-gated lifecycle transitions,
  acknowledges worker park/wake state, rechecks activity under the lifecycle
  mutex, and rejects unchanged generations instead of replaying a stale job.
- `conv_threaded.cpp` publishes worker readiness while holding the condition-
  variable predicate mutex and notifies after that protected transition.
- All 37 other Stage60 implementation/research paths are explicitly rejected
  in `scheduler_backport_rejected_stage60_hunks.tsv`.

Tests, 0.9.3 release identity, RPATH-free packaging, release-root launchers,
documentation, and append-only evidence are the only additional changes. The
runtime binary embeds implementation commit `c0c3f1a`; packaging commit
`2965f59` changes shell launch lookup only.

## Host And Binary Validation

Native, ASan/UBSan, and TSan builds each pass all 53 CTests. TSan reports zero
races. Twenty repeated startup runs cover 2,000 iterations, 8,000 workspace
constructions, and 20,000 readiness transitions. Twenty repeated lifecycle
runs cover 40,000 active-window transitions with watchdog timeouts, worker
counts 1-4, empty-window stale-generation checks, and parked destruction.

The official SpacemiT 1.1.2 build retains
`-march=rv64gcv_zvfh -mabi=lp64d -mtune=spacemit-x60 -funroll-loops`.
ABI1 exports the same 15 C functions under `Y26_K1X_ABI_1`, SONAME remains 1,
all five `DT_NEEDED` entries are unchanged, and official artifacts have no
RPATH, RUNPATH, TEXTREL, or unexpected text relocation. Arithmetic and package
translation units are byte-identical to the matched 0.9.2 rebuild; only the
documented scheduler control flow and immutable version metadata differ.

## Board Correctness And Stability

F0-F7, bus, Zidane, scalar/optimized comparison, all 215 integer boundaries,
FRM 0-4, ambient vcsr 0/2/3/4/7 restoration, affinity, and CPU4-7 IME-zero
checks pass. The fixed output remains `0xd43f5e018b415631`; the package manifest
remains `fab4a72cf524ce0a205ceca0384144f2eee7bc79dff3f4db8b7208614e8407be`.

Board stress passes 10,000 active-window transitions, 1,000 in-process startup
iterations (4,000 workspaces and 10,000 readiness transitions), and 100 clean
process construction/destruction starts without timeout, stale replay, or
wrong completion.

The same-session 1,000-sample-per-arm ABI1 ABBA is a maintenance
non-regression result, not a speed claim. Pure executor mean is 131451.758 us
for 0.9.2 and 131200.166 us for 0.9.3 (`-0.191%`); p95, p99, and p99.9 are all
lower for 0.9.3. The separate 13,500-run 0.9.3 O2 soak records mean
132367.100222 us, p95 132874 us, p99 134150.010 us, p99.9 136077.044 us,
maximum 137243 us, CV 0.317932%, exact output, zero affinity failures, and zero
IME executions on CPU4-7.

## Accuracy

Full COCO val2017 completes 5000/5000 with zero image failures, mAP50-95
`0.3707408944391919`, and byte-identical prediction SHA-256
`cda5c8c7a46d61d9c90f6292001eea190cb8f6617efe647a33dc6134dd57ccda`.
A preliminary research-default invocation was stopped and preserved as an
aborted scout; no conclusion uses it. The selected run explicitly enables the
frozen release route.

## Camera And Rollback

The selected clean-extract R640 camera route completes 30 minutes with 12,265
measured GUI frames. OpenCV returns 14.984741 decoded frames/s; the application
processes and displays 6.813771 frames/s. Latest-slot replacement is 54.528603%
and is not labeled as sensor loss. Executor mean is 132.534921 ms, temperature
remains 47-61 C, and all observed CPU samples remain at 1.6 GHz.

Normal exit, direct-child SIGINT/SIGTERM/SIGHUP, USB capture reset, and an
unwritable recorder target all flush readable metrics/detections, join capture
and writer threads, leave no demo process, and restore the camera profile. The
initial outer-wrapper SIGINT attempt did not reach the demo and is retained as
a rejected harness result; the corrected direct-child test passes.

Final state matches preflight: original boot ID, `performance` governor,
workqueue mask `ff`, absent O2 snapshot/cgroup, inactive camera profile, xHCI
IRQ 89 affinity `0-7`, and no persistent service, sysctl, boot, or storage
change.

## Release 0.9.3

The runtime and internal-R&D tar/zip deliveries are byte-identical across two
final generations and all four clean extracted trees verify `SHA256SUMS`.
Every board extract passes dependency verification, healthcheck, known fixture,
image/video/camera smoke, and CMake shared/static plus pkg-config consumers.
The uninstall-isolation test removes only its target and preserves an adjacent
sentinel. The runtime delivery excludes source ONNX. The internal-R&D delivery
contains the authorized source ONNX at exact SHA-256
`30a94e4738606673b5e0a73499cbc977167f046f8fa8637d6040ce744f429c0c`;
external redistribution remains not cleared.

Final archive SHA-256 values are recorded in
`distribution_archive_hashes.tsv`. Large archives, predictions, media, and raw
logs remain under `/data`; none are committed to Git and no project artifact is
written to eMMC.

## Scope Note

Variable worker-count regression coverage uses the generic
`all_workers_complete` strategy. The frozen release uses its established
four-worker completion policy. An unsupported research-only combination of
`active_workers_complete` with fewer workers than the pool exposed pre-existing
metadata ownership outside the selected release route; Stage60M does not add a
third repair beyond its explicitly authorized two-fix scope.

The frozen source branch remains unchanged. The 0.9.3 maintenance branch is
not merged by this stage, and no Stage61/Stage62 work is opened.

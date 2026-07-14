# Stage 52 final report

classification: stage52-full-executor-release-ready-functional
stage_id: BANANA-YOLO26-CUSTOM-INT8-IME-ENGINE-STAGE52-FULL-YOLO26-K1X-INT8-V1-EXECUTOR-COMPLETION-COCO-HANDOFF-AND-RELEASE-GATE-001
repo: /data/banana-yolo26-spacemit-demo
branch: yolo26-custom-int8-engine
start_head: 6ce755e4f4120c718279a434e96d8b582fce0b6e
end_head: self-identity and remote parity are recorded in the result packet
commit_count: three Stage52 commits after the final tree is sealed
pushed: final fast-forward verification is recorded outside this tracked tree

## Decision

The complete current YOLO26n-640 graph now runs as a standalone static
`K1X_INT8_V1` executor. Correctness, full COCO accuracy, API/CLI, package,
documentation, stable timing, the 10,000-run safe soak, and the release bundle
pass their functional handoff gates. The release is not in the `fast` band:
selected E2c2 SCHED_OTHER averages 504137.644000 us, while the matched B120 ORT
diagnostic averages 460208.112134 us.

The selected strategy is release maintenance plus measured full-executor hotspot
optimization. Model/executor co-design, training, and student selection remain
unauthorized.

## Frozen identity

- Arithmetic contract: `K1X_INT8_V1`.
- Full profile: `K1X_INT8_V1_YOLO26N_640_FULL_GRAPH_001`.
- Model SHA-256: `30a94e4738606673b5e0a73499cbc977167f046f8fa8637d6040ce744f429c0c`.
- Package manifest SHA-256: `d3b4cb794f1373aa712d77bab177a5f7da58530361c9af58c0caf5bbcd6dc75f`.
- Repeated package-tree name/size manifest SHA-256:
  `1c413f8a71988386d493ae4004af0b43470104f2f4dc41ac1882fde5c304a8`.
- Static schedule: 215 operations and 215 package-defined integer boundaries.
- Arena: 8192000 bytes; packed weights: 3430400 bytes.
- Dense E2c compatibility: 2665292800/2665292800 dense MACs, or 97.268007166% of all graph MACs.

## Runtime contract

- Input is frozen preprocessed `1x3x640x640` float32 RGB NCHW `/255`; the RGB API
  provides the documented OpenCV letterbox path.
- Output is deterministic `1x300x6` float32 in
  `x1,y1,x2,y2,confidence,zero_based_class` order.
- The measured graph path has zero ORT calls, Python calls, per-run allocations,
  per-run file I/O, float Q/DQ materializations, and per-operator string dispatch.
- Resident feature tensors use `NCHWc8_SPATIAL_INNER_V1`; declared graph transforms
  are integer schedule operations, not adapter conversions around custom islands.
- CPU0-3 execute IME. CPU4 is the controller. CPU4-7 execute zero IME instructions.
- `SCHED_OTHER` is the handoff default; RR20 is a bounded privileged lab sidecar.

## Selected implementations

- Dense Conv: M12xN16 plus exact tails, P3 direct A delivery, exact E2c2 Q62
  `vsmul.e64` requantization/LUT/store.
- Stem: exact generic C3-through-K8 dense route; a dedicated RGB/RGBX stem remains
  an optimization target.
- Grouped/depthwise Conv: exact four-worker direct scalar integer route.
- Attention: static packed integer MatMul and package-defined fixed-point/LUT Softmax.
- Small-N head: exact masked-N16 IME route for N4/N8/N16.
- Head selection: deterministic fixed TopK/Gather implementation with frozen tie order.

## Correctness

- Independent Python package/operator audit, portable C++ scalar, board scalar, and
  board optimized surfaces agree exactly on F0-F7 at all 215 integer boundaries.
- Bus and Zidane fixtures agree exactly between host/board scalar and optimized paths.
- The final output hash for F0 is `0xd43f5e018b415631`; all repeated hashes are stable.
- FRM RNE/RTZ/RDN/RUP/RMM and ambient `vcsr` states are restored; E2c2 focused
  adversarial and all-package-channel checks pass.
- CPU affinity passes, CPU4-7 IME count is zero, and no SIGILL occurred.

## Accuracy

All 5000 official COCO val2017 images completed with no image failure. The K1X
prediction JSON contains 721755 rows and has SHA-256
`cda5c8c7a46d61d9c90f6292001eea190cb8f6617efe647a33dc6134dd57ccda`.

| Surface | mAP50-95 | mAP50 | AP small | AP medium | AP large |
|---|---:|---:|---:|---:|---:|
| Accepted FP32 | 0.401438855549 | not rerun | not rerun | not rerun | not rerun |
| Legacy semantic INT8 | 0.372453424642 | 0.526269698607 | 0.181116238958 | 0.415342169837 | 0.547344785459 |
| K1X_INT8_V1 | 0.370740894439 | 0.525846530087 | 0.183972946262 | 0.414262735261 | 0.544043381180 |

The K1X delta is -0.001712530203 absolute versus semantic INT8. It passes the
preferred <=0.002 gate. The 2000-resample paired per-image F1-proxy interval is
reported separately and is not mislabeled as a COCO mAP confidence interval.
The COCO run used exact E2c; selected E2c2 is byte-exact under the same arithmetic
contract and changes timing, not this measured accuracy surface.

## Performance

- Selected E2c2 SCHED_OTHER 10/100/5: 504137.644000 us mean,
  502911.500000 us median, 506182.200000 us p95, 527434.100000 us p99.
- Selected E2c2 RR20 sidecar: 503473.764000 us mean, 515912.800000 us p95.
- Exact E2c control: 558339.400000 us mean; E2c2 improves the full-model mean by
  9.707672% and improves p99 by 9.211568%.
- Selected E2c2 SCHED_OTHER 10000-run soak: 503576.174900 us mean,
  502440.000000 us median, 505918.400000 us p95, 527271.240000 us p99,
  546540.109000 us p99.9, and 595021.000000 us maximum. All 10000 runs
  produced `0xd43f5e018b415631`; affinity passed and CPU4-7 executed no IME.
- Matched B120 ORT: 460208.112134 us mean across five 100-inference repeat means.
  The custom/ORT mean delta is +9.545580%, and ORT/custom is 0.912862x.
- Pure-model throughput is 1.983585 inferences/s; this is not a 20 FPS result.
- Dedicated preloaded-image path: 46464.740510 us preprocess, 470259.641654 us
  fixture-specific pure graph, 5.417302 us output decode, and 516859.673560 us total.
  The 5000-image COCO dataset timing includes board-NVMe JPEG reads and is reported
  separately from this preloaded fixture surface.

## Handoff

The public C ABI, C++ API, static/shared libraries, CLI, deterministic package,
build/deploy/smoke/benchmark/uninstall scripts, documentation set, and release
checksum manifest are complete. The bundle is rooted at
`/data/releases/banana-yolo26-k1x-int8-executor` and deploys to board NVMe under
`/data/k1x-yolo26-int8-executor`. It contains 1201 checksummed files totaling
29361374 bytes. The release manifest SHA-256 is
`aeb7712395efee7f0a49ddc280e4e4002122aebaa4ba3e644b2ea6615741ea44`;
the SHA-256 of `release_sha256.txt`, used as the bundle identity in handoff, is
`82b5a4144a67208dbcf7047b1c8d31f4bcf65c8232e8b12da246f3442f1a1959`.
Host and deployed-board checksum verification passed, followed by exact C ABI
and CLI smoke tests. The board eMMC exception count is zero.

## Broken or rejected

- The complete custom executor is slower than matched B120 ORT and is far from
  20 FPS. The release is functional, not fast or production-ready.
- Dedicated stem, true N4/N8 kernels, vector grouped/depthwise Conv, attention,
  input quantization, materialized Concat/Add, and head paths remain measured hotspots.
- Synchronous cluster1 offload was previously rejected and was not reopened.

## Unknown

- Camera-service latency, power, thermal behavior outside the recorded lab surface,
  production reliability, and trained-student accuracy remain unproven.
- No claim is made that the preferred accuracy delta generalizes to a different
  model package, preprocessing contract, scheduler, board image, or arithmetic profile.

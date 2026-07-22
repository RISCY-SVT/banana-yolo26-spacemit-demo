# Stage62 Final Report

## Classification

`stage62-final-main-integration-pass-internal-rd-packages-built-legal-clearance-not-certified`

Stage62 consolidated the immutable 0.9.3 maintenance and Stage61 Q0 research histories without rebase, squash, cherry-pick, or force update. R640 remains the only default. R512, R448, R416, R384, R352, R320, R256, and R768 are explicit experimental Q0 profiles with no automatic fallback or deployment promotion.

## Integration

- Main start: `175c1d939cc93fba0e730dba3f1281704e8f25b9`.
- Stable maintenance: `d0e3611c8d99dfade049bd261cb557509222a456`.
- Stage61 research: `fa668ccaf7938336bd10313455ab81557b33e020`.
- Merge base: `175c1d939cc93fba0e730dba3f1281704e8f25b9`.
- No-ff merge: `7a8138ac2ef78d26ff92f1f8cc40f6d0b3286d93`.
- Integrated release source: `9f88644aef6a9eb304cae3e95b62da6a0aa22cc3`.

The only textual merge conflict was the test target list in `custom_int8_engine/tests/CMakeLists.txt`; both Stage60M scheduler tests and Stage61 profile/N-tail tests were retained. Scheduler implementation files were byte-identical between accepted parents. The final protocol contains one readiness/lifecycle repair and preserves Stage61's fail-closed partial-worker rule.

## Exactness And Accuracy

All 99 profile/fixture combinations passed all 215 integer boundaries across host scalar, board scalar, and optimized execution. FRM/vector-CSR restoration, affinity, final outputs, and CPU4-7 IME count zero passed. R640 remains `0xd43f5e018b415631`.

Full COCO val2017 completed for all nine profiles, 45,000 image evaluations total, with zero image failures and exact Stage61 prediction hashes. R640 remained `0.3707408944391919` mAP50-95 with prediction SHA-256 `cda5c8c7a46d61d9c90f6292001eea190cb8f6617efe647a33dc6134dd57ccda`. R768 measured `0.37354959453260156` with prediction SHA-256 `5ca1639a6b46545f21d298501727f9273bebf8825b7ef8eee2a6fb7a4f73668e`.

The R768 point estimate is +0.281 AP, but the paired per-image IoU-averaged F1 bootstrap delta was -0.003302 with 95% CI [-0.004867, -0.001747]. This is a mixed size/class effect, not grounds for default promotion.

## Performance And Stability

The merged 500-sample preprocessed pure-model means were:

| Profile | Mean | p95 | Pure FPS | Policy |
| --- | ---: | ---: | ---: | --- |
| R640 | 131.751 ms | 132.362 ms | 7.590 | accepted default |
| R384 | 47.581 ms | 47.747 ms | 21.017 | diagnostic; 6.42 AP loss |
| R768 | 197.239 ms | 197.979 ms | 5.070 | quality research; not promoted |

All nine means satisfy the Stage61 non-regression gate. Randomized ABBA found merged R640 0.696% faster than immutable 0.9.3 and merged R768 0.074% faster than the Stage61 control; Stage62 makes no speed claim from these integration-only differences.

The R640, R384, and R768 10,000-run soaks passed with exact outputs and zero CPU4-7 IME use. Their means were 132.017 ms, 47.640 ms, and 197.838 ms. Rare p99/p99.9 tails are reported separately as operating-system/IRQ effects rather than kernel regressions.

## Camera And Cleanup

The 30-minute R640 camera soak processed/displayed 6.623871 FPS from 14.997737 OpenCV-decoded frames/s, with 55.834198% latest-slot replacement. The 180-second merged R768 confirmation processed/displayed 4.403098 FPS from 15.002738 decoded frames/s and is explicitly an upsample from the 640x480 camera source. Neither rate is labeled raw sensor FPS or sensor-to-screen latency.

Normal, INT, TERM, HUP, injected capture-failure, and recorder-failure paths restored the camera/O2 profile. Final clean-extract image, video, and live-camera smokes passed for all four deliveries. No demo process, temporary cgroup, IRQ policy, workqueue policy, or persistent system change remains.

## Deliveries

Stable 0.9.3 delivers an R640 runtime and an R640 internal-R&D source/model bundle. Integrated `0.10.0-internal-rd.1` delivers a nine-profile binary SDK and a complete-source bundle. ABI and SOVERSION remain 1; the public ABI symbol set is unchanged. The integrated build embeds source `9f88644aef6a9eb304cae3e95b62da6a0aa22cc3`.

Every tar.gz and zip was independently generated twice with byte-identical SHA-256. Both formats were clean-extracted, internally verified, and compared file-for-file. CMake shared/static and pkg-config consumers passed from each extracted release with no RPATH/RUNPATH. The complete-source tree independently rebuilt the same integrated shared-library SHA-256.

The authoritative paths and hashes are in `FINAL_RELEASE_MANIFEST.tsv`, `stable_release_archive_hashes.tsv`, and `internal_rd_archive_hashes.tsv`.

## Licensing Status

The selected technical route is `agpl-complete-source-route-selected`. The complete-source delivery includes project and assembly source, build files, generators, tests, source ONNX, all nine static ONNX graphs, prepared packages, profile manifests, notices, SBOMs, and build instructions; it excludes COCO images.

Legal clearance is not certified. No Enterprise agreement was found, project ownership/relicensing authority is unresolved, exact source-model export authority is incomplete, and external model conveyance is not cleared. The technical package follows the requested AGPL route but is not a legal opinion. This distinction follows the published [Ultralytics licensing guidance](https://www.ultralytics.com/license), its [AGPL explanation](https://www.ultralytics.com/legal/agpl-3-0-software-license), and the [GNU GPL FAQ](https://www.gnu.org/licenses/gpl-faq.en.html).

## Freeze

This executor line is frozen for maintenance and evidence. Future PTQ, training, topology changes, model-executor co-design, or new performance research requires a separate branch/project and explicit authorization. No production certification is claimed.

Exact post-publication branch/tag parity and the result-packet tree hash are recorded by the final publication/export step because a Git commit cannot contain its own object ID.

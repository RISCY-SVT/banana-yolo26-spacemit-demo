# Release 0.9.3 Maintenance Notes

Release 0.9.3 is a scheduler-liveness maintenance update to the frozen YOLO26n-640 K1X INT8 executor.

## Fixed

- Frame-gated worker-pool begin/end transitions now serialize under the lifecycle mutex, acknowledge park/wake state, and reject unchanged job generations.
- Threaded-convolution workers publish startup readiness while holding the condition-variable predicate mutex.

## Unchanged

- ABI 1 and shared-library SOVERSION 1.
- `K1X_INT8_V1` arithmetic, Q62/RNE/saturation semantics, and `NCHWc8_SPATIAL_INNER_V1` layout.
- `K1X_INT8_V1_YOLO26N_640_FULL_GRAPH_001` package/profile and all model assets.
- Public C symbols, input/output contracts, selected operator dispatch, camera policy, and supported platform.
- Runtime bundle excludes source ONNX; external ONNX redistribution remains not cleared.

The Stage60 resolution/profile implementation is intentionally absent. This release is a non-regression maintenance backport, not a performance or model release.

## Build identity

The official runtime binaries embed implementation commit
`c0c3f1a13662aec9ba168963c651e164541905ba`. Packaging commit `2965f59`
adds only release-root-relative library lookup to the extracted-tree shell
launchers after RPATH/RUNPATH removal. The final evidence/publication commit is
recorded separately in `commit_inventory.tsv` and `final_remote_parity.tsv`.

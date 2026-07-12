
# Stage49 prompt: Persistent NCHWc8 contiguous-slice and LUT-v2 gate

## Start

- repo: `/data/banana-yolo26-spacemit-demo`
- branch: `yolo26-custom-int8-engine`
- expected HEAD: Stage48 local commit reported in the final response
- authority: `K1X_INT8_V1`; legacy float-QDQ remains diagnostic only

## Mission

Prove one persistent `NCHWc8_SPATIAL_INNER_V1` contiguous model4-to-model6
slice using offline integer packages, one arena, shared immutable packed weights,
persistent CPU0-3 workers, no internal layout conversion, no float Q/DQ, and no
ORT in the measured slice. Measure entry/exit adapters separately.

1. Extend the independent integer exporter/oracle only to the exact model4-6
   operations needed by the slice.
2. Preserve exact Python/C++/board scalar/IME parity at every integer boundary.
3. Reuse Stage48 direct model5 M12/spatial/four-worker route.
4. Add one exact RVV epilogue candidate only if measured slice attribution says
   the scalar epilogue is the remaining bounded bottleneck.
5. Compare the exact custom internal slice and adapter-inclusive slice against a
   resource-matched B120 ORT diagnostic cut.

No RT205 work, student blueprint/training, CPU4-7 IME, full graph executor,
default dispatch, production claim, or push is authorized.

# Model5-8 Oracle Report

The package contains eight deterministic fixtures and 48 unique boundary tensors. Every isolated host cut replay passed shape, dtype, and byte equality.

Fixtures:

- F0: accepted Stage42 synthetic full-model input;
- F1/F2: deterministic full-range uint8 stress seeds 4301 and 4302;
- F3: zero-point-centered structured values;
- F4: edge/saturation-oriented values;
- F5-F7: canonical project photo, Ultralytics bus, and Ultralytics Zidane images using the recorded OpenCV letterbox pipeline.

## Oracle hierarchy refinement

Independent Q/DQ/Conv integer semantics are Level 0. Fixed host ORT remains Policy B authority, but Stage43 records two host session surfaces:

- semantic cut oracle: ORT 1.27.0 CPU EP, `ORT_DISABLE_ALL`;
- operational integration artifact: ORT 1.27.0 CPU EP, `ORT_ENABLE_ALL`.

The custom scalar path matches the semantic host cut exactly on all eight fixtures. `ORT_ENABLE_ALL` differs on F1, F2, F4, and one F7 element. A u8xs8 adjacent-pair int16 saturation model reproduces sampled optimized-ORT Conv codes exactly; this is an x86 MLAS optimization artifact and is not implemented in the K1X kernel because it does not represent the graph's exact DQ-Conv-Q semantics.

No tolerance is used. Both oracle surfaces and their hashes are preserved.

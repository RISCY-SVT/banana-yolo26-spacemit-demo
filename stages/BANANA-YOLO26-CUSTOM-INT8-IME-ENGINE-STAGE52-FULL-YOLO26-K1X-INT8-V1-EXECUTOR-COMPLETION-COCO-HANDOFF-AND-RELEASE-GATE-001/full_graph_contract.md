# Full graph contract

- Contract: `K1X_INT8_V1`
- Profile: `K1X_INT8_V1_YOLO26N_640_FULL_GRAPH_001`
- Model SHA-256: `30a94e4738606673b5e0a73499cbc977167f046f8fa8637d6040ce744f429c0c`
- Input: `images`, float32 NCHW `1x3x640x640`, RGB, values in `[0,1]`
- Direct RGB API input: already letterboxed `640x640x3`, interleaved RGB bytes
- Output: `output0`, float32 `1x300x6`
- Resident feature layout: `NCHWc8_SPATIAL_INNER_V1`
- Linear attention layout: `ROW_MAJOR_U8_V1`
- Integer operations: 215
- Package-defined integer boundaries: 215
- Arena: 8,192,000 bytes

The package freezes all tensor shapes, scales, zero points, layouts, operation
dependencies, Q62 assets, LUTs, static schedule IDs, and final head tie order.
The runtime parses and validates those descriptors once during prepare. The
run path uses a prepare-bound function-pointer schedule and contains no graph
name lookup, operator registry, ORT call, Python call, package parsing, or file
I/O.

TopK ordering is score descending, then stable point/index order. Final values
are derived from package-defined Q16 box and Q24 score tables. Legacy float-QDQ
ORT remains diagnostic and is not the exact integer authority.

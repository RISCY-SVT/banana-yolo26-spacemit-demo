# Pre-Registered Hypotheses

stage_id: `BANANA-YOLO26-CUSTOM-INT8-IME-ENGINE-STAGE23-MODEL4-CUT-CLOSED-BUCKET-ATTRIBUTION-AND-TARGETED-REPAIR-001`
start_head: `8350c57bd015f044a51800dcd318cb43976e534a`

## H1_runner_api_cut_closure

The real integrated C++ model4 C2f runner API can be driven with the same full-shape ONNX-cut input tensor(s) and can produce the ONNX-cut output boundary bit-exactly.

## H2_bucket_attribution

Stage22 `mean_total_us=225214` contains a large not-yet-attributed bucket. Stage23 must attribute at least 90% of total runtime into non-overlapping buckets.

## H3_output_quantize

The largest not-yet-attributed bucket is expected to be final `/model.4/cv2` output quantization `int32 -> uint8 NHWC`. A selected RVV or equivalent exact path should reduce that bucket by at least 3x, or Stage23 must classify why not.

## H4_rounding_robustness

RNE robustness must be part of the runner/API path or the FP-sensitive section must be eliminated. A bench-only scoped RNE guard is not sufficient as the final accepted runner behavior.

## H5_no_graph_expansion

No graph expansion is allowed in Stage23. If the ONNX-cut runner API proof fails, no performance repair may be accepted.

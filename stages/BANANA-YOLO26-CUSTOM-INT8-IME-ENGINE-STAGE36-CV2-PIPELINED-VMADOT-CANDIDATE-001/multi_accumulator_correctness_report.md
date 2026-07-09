# Multi-Accumulator Correctness Report

Stage36 strengthened correctness by running the real selected-cut path through the new accumulator layouts and comparing the final quantized ONNX-cut boundary.

Candidates:

- A1: four accumulator groups
- A2: six accumulator groups

Board smoke:

| candidate | affinity | runs | status | mismatches | max_abs_diff | sha_status |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| A1_cv2_pipelined4 | CPU0-3 | 1 | 0 | 0 | 0 | match |
| A2_cv2_pipelined6 | CPU0-3 | 1 | 0 | 0 | 0 | match |

Stable run:

| candidate | warmup | runs | repeats | status | mismatches | max_abs_diff | sha_status |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| A1_cv2_pipelined4 | 10 | 100 | 5 | 0 | 0 | 0 | match |
| A2_cv2_pipelined6 | 10 | 100 | 5 | 0 | 0 | 0 | match |

CPU0 single-thread selected candidate:

| candidate | affinity | status | mismatches | max_abs_diff | sha_status |
| --- | --- | ---: | ---: | ---: | --- |
| A1_cv2_pipelined4 | CPU0 | 0 | 0 | 0 | match |

Conclusion: all accumulator groups used by the selected A1 and diagnostic A2 paths contribute to byte-exact selected-cut output.

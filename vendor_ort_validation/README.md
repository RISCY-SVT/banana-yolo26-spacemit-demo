# SpacemiT ORT validation tools

This directory contains Stage63 and Stage64 helpers for immutable SpacemiT
ONNX Runtime evidence. It is intentionally separate from
`custom_int8_engine/`.

The tools never calculate model outputs and never treat provider logs as a
correctness oracle:

- `summarize_profile.py` converts ONNX Runtime profile events into
  provider-assignment and aggregate TSV files.
- `summarize_runner.py` converts one-sample repeat records from the immutable
  Stage46 runner into distribution statistics.
- `run_board_performance.sh` runs isolated canonical, stability, and thread
  scaling matrices against versioned runtime roots.
- `run_stage63_coco_matrix.sh` selects a version-bound predictor for each
  runtime; `run_coco_surface.sh` records the binary, model, and runtime
  identities for each isolated surface.
- `evaluate_coco.py` evaluates only the image IDs present in a predictor timing
  file and emits global and per-class COCO metrics.
- `inventory_onnx.py` records source and dumped-provider-subgraph operator
  inventories without inferring source-node assignment from transformed names.
- `summarize_fixed_outputs.py` validates fixed-fixture tensor structure and
  records finite-value and hash evidence.
- `stage64_make_fixtures.py` creates tiny signed/unsigned QDQ and ReduceMax
  controls with independent expected outputs.
- `stage64_onnx_audit.py` records QDQ signedness, scale granularity,
  QLinear counts, and explicit Conv `kernel_shape` coverage.
- `stage64_host_validate.py` validates the split FP32 and S8 pipelines on
  x86_64 before a model can reach the board.
- `stage64_validate_direct.py` applies the same finite-output and score-collapse
  gate to diagnostic direct-E2E XSlim models.
- `stage64_build_tables.py` derives compact review tables from immutable raw
  quantization, host, board, profile, timing, and COCO evidence.
- `stage64_two_stage_runner.cpp` and `stage64_two_stage_coco.cpp` execute a
  quantized inference graph followed by the explicitly CPU-only FP32 tail.
- `stage64_run_two_stage_board.sh` and `stage64_run_coco_board.sh` enforce
  task-local storage and explicit runtime binding for board measurements.

Large models, vendor archives, profiles, predictions, and raw logs remain
under the task-owned `/data` evidence roots.

`evaluate_coco.py` requires the accepted host evaluation environment with
`pycocotools==2.0.11` and NumPy. It does not install packages or access the
network.

The cross-built runtime runner and COCO predictor are compiled from the
accepted Stage46 sources. Neither contains custom IME instructions; provider
code is loaded only from the explicitly selected versioned runtime directory.

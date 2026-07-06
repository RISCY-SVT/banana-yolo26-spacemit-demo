# Model4 Fullshape Correctness Report

status: `pass`
selected_subset: `candidate_K_model4_c2f_representative_fullshape_synthetic`

## Board Correctness

Board run:

```text
taskset -c 0-3 ./bench_stage20_model4_fullshape_c2f 10 100 5
```

All candidates in `model4_fullshape_threading_matrix.tsv` reported:

```text
status=0
mismatches=0
checksum=-3094964234
affinity_ok=1
```

## Oracle Scope

The board timing tool uses a representative full-shape synthetic model4 C2f tensor pattern with real model4 shapes, weights, scales, zero-points, and correction formulas. The ONNX Runtime full-shape boundary extraction for the accepted Q/DQ model passed separately and is recorded in `fullshape_boundary_manifest.tsv`.

This is not a full YOLO26 inference oracle.

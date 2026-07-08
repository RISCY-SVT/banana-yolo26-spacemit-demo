# Low-Overhead Sliding Layout Report

stage_id: BANANA-YOLO26-CUSTOM-INT8-IME-ENGINE-STAGE32-MMT4D-OR-LOW-OVERHEAD-SLIDING-DATA-LAYOUT-DECISION-001
target_node: /model.4/m.0/cv1/conv/Conv
target_shape: 3x3, 80x80x32 -> 80x80x16
panel_gate_us: 7800
protocol: warmup=10 runs=100 repeats=5
board_affinity: taskset -c 0-3

## Raw Log

```text
/data/ncnn-logs/ai-team/2026-07-08_12-51-18/BANANA-YOLO26-CUSTOM-INT8-IME-ENGINE-STAGE32-MMT4D-OR-LOW-OVERHEAD-SLIDING-DATA-LAYOUT-DECISION-001/run_logs/stage32_layout_signedness_board_retry.log
```

## Layout-Only Candidate Matrix

| candidate | attachable_to_current_kernel | mean_us | stddev_us | cv_pct | checksum | gate_status |
|---|---:|---:|---:|---:|---:|---|
| B0_stage31_full_panel | 1 | 44704.7 | 6.01848 | 0.0134627 | -203239387 | fail |
| B1_row_cache_materialized | 1 | 47224.4 | 3.5138 | 0.00744063 | -203239387 | fail |
| B2_descriptor_only | 0 | 166.498 | 0.235209 | 0.141268 | 709376193 | pass-not-attachable |
| B3_interior_fast_path | 1 | 18447.2 | 9.50568 | 0.0515291 | -203239387 | fail |
| B4_row_cache_descriptor_model | 0 | 38781.2 | 10.6849 | 0.0275518 | 531533933 | fail |

## Gate Result

```text
stage32_direct_row_cache status=not_attempted layout_gate=fail buildtime_ime=1
```

No attachable layout candidate reached the required `panel_build_new_us <= 7800` gate.

`B2_descriptor_only` is intentionally not accepted as a direct/sliding path proof: it proves that a descriptor walk is cheap, but it does not feed the current `smt.vmadot1/2/3` kernel schedule without a later gather/pack cost. Counting it as a usable direct Conv layout would hide the missing data movement.

## Conclusion

The low-overhead sliding lane is rejected for now. The best attachable candidate, `B3_interior_fast_path`, remains `18447.2 us`, which is `2.365x` above the Stage32 gate and still too large to compete with the current MMT4D 4-thread node time.

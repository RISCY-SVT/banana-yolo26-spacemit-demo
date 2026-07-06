# Source Hygiene Report

stage_id: `BANANA-YOLO26-CUSTOM-INT8-IME-ENGINE-STAGE22-MODEL4-ONNX-CUT-ORACLE-AND-NEXT-BLOCK-GATE-001`

## Checks

```text
git_diff_check: pass
git_diff_cached_check: pass
symlink_scan: pass
secret_like_scan_changed_files: pass
large_generated_artifacts_staged: none
/data/ncnn_mutated_by_stage22: false
/data/banana-yolo11-spacemit-demo_mutated_by_stage22: false
```

## Notes

- Full-shape `.onnx`, `.npy`, and `.bin` artifacts were generated under `.deps/custom_int8_engine/stage22_onnx_cut/` and were not staged for git.
- `/data/ncnn` had pre-existing unrelated dirty files; Stage22 did not touch them.
- `/data/banana-yolo11-spacemit-demo` had pre-existing unrelated untracked files; Stage22 did not touch them.
- The changed-file ASCII/control scan reports Russian text in `STAGE*_SUMMARY_RU.md`; this is expected because user-facing summaries must be in Russian.

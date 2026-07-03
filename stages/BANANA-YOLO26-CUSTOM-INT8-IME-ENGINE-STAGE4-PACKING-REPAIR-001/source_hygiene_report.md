# Source Hygiene Report

secret_like_scan: pass
symlink_scan: pass
large_generated_blobs_staged: no
ncnn_source_mutated: false
xslim_used: false

Note: the broad first scan produced a self-referential false positive in
`commands.txt` because the logged `rg` command contained the words from the
pattern. A filtered scan excluding `commands.txt` found no secret-like content.

Changed files scanned:
- `custom_int8_engine/CMakeLists.txt`
- `custom_int8_engine/include/y26_k1x_conv_kernels.h`
- `custom_int8_engine/kernels/conv_mmt4d_prepack.cpp`
- `custom_int8_engine/tests/CMakeLists.txt`
- `custom_int8_engine/tests/test_stage4_packing_repair.cpp`
- `custom_int8_engine/tools/bench_stage4_packing.cpp`
- `stages/BANANA-YOLO26-CUSTOM-INT8-IME-ENGINE-STAGE4-PACKING-REPAIR-001/STAGE4_FINAL_REPORT.md`
- `stages/BANANA-YOLO26-CUSTOM-INT8-IME-ENGINE-STAGE4-PACKING-REPAIR-001/STAGE4_SUMMARY_RU.md`
- `stages/BANANA-YOLO26-CUSTOM-INT8-IME-ENGINE-STAGE4-PACKING-REPAIR-001/board_remote_dir.txt`
- `stages/BANANA-YOLO26-CUSTOM-INT8-IME-ENGINE-STAGE4-PACKING-REPAIR-001/commands.txt`
- `stages/BANANA-YOLO26-CUSTOM-INT8-IME-ENGINE-STAGE4-PACKING-REPAIR-001/conv1x1_stage4_report.md`
- `stages/BANANA-YOLO26-CUSTOM-INT8-IME-ENGINE-STAGE4-PACKING-REPAIR-001/conv3x3_stage4_report.md`
- `stages/BANANA-YOLO26-CUSTOM-INT8-IME-ENGINE-STAGE4-PACKING-REPAIR-001/log_dir.txt`
- `stages/BANANA-YOLO26-CUSTOM-INT8-IME-ENGINE-STAGE4-PACKING-REPAIR-001/microbench_stage4_report.md`
- `stages/BANANA-YOLO26-CUSTOM-INT8-IME-ENGINE-STAGE4-PACKING-REPAIR-001/packing_dataflow_repair_plan.md`
- `stages/BANANA-YOLO26-CUSTOM-INT8-IME-ENGINE-STAGE4-PACKING-REPAIR-001/real_conv_node_selection_refresh.md`
- `stages/BANANA-YOLO26-CUSTOM-INT8-IME-ENGINE-STAGE4-PACKING-REPAIR-001/run_ts.txt`
- `stages/BANANA-YOLO26-CUSTOM-INT8-IME-ENGINE-STAGE4-PACKING-REPAIR-001/shared-log-dir.txt`
- `stages/BANANA-YOLO26-CUSTOM-INT8-IME-ENGINE-STAGE4-PACKING-REPAIR-001/sliding_vmadot_ops_note.md`
- `stages/BANANA-YOLO26-CUSTOM-INT8-IME-ENGINE-STAGE4-PACKING-REPAIR-001/stage3_baseline_recheck.md`
- `stages/BANANA-YOLO26-CUSTOM-INT8-IME-ENGINE-STAGE4-PACKING-REPAIR-001/stage5_prompt.md`
- `stages/BANANA-YOLO26-CUSTOM-INT8-IME-ENGINE-STAGE4-PACKING-REPAIR-001/task-packet-caveat.txt`
- `stages/BANANA-YOLO26-CUSTOM-INT8-IME-ENGINE-STAGE4-PACKING-REPAIR-001/task_run_dir.txt`
- `stages/BANANA-YOLO26-CUSTOM-INT8-IME-ENGINE-STAGE4-PACKING-REPAIR-001/weight_prepack_format_v1.md`
- `stages/BANANA-YOLO26-CUSTOM-INT8-IME-ENGINE-STAGE4-PACKING-REPAIR-001/workspace_reuse_report.md`
- `stages/BANANA-YOLO26-CUSTOM-INT8-IME-ENGINE-STAGE4-PACKING-REPAIR-001/zero_point_requant_boundary_report.md`

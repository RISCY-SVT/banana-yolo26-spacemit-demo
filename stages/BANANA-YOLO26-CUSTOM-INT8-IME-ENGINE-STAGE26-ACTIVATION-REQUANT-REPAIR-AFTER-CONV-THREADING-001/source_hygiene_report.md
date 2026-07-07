# Source Hygiene Report

stage_id: BANANA-YOLO26-CUSTOM-INT8-IME-ENGINE-STAGE26-ACTIVATION-REQUANT-REPAIR-AFTER-CONV-THREADING-001

## Checks

```text
git diff --check: pass
git diff --cached --check: pass
symlink_scan: pass, no symlinks printed
secret_like_scan: pass, no findings in changed tracked files
path_scan: pass for changed tracked files; board-local path is recorded separately as raw evidence pointer
host_ctest: pass, 39/39
riscv_cross_build: pass
board_correctness: pass
result_packet_export: pending at report creation time
```

## Notes

Board deployment path is board-local evidence and is not a portable source path:

```text
/home/svt/yolo26-custom-int8-stage26/2026-07-07_10-38-26
```

No `/data/ncnn` mutation, XSlim use, vmadot1/2/3, vmadotn, FP/vfmadot, CPU4-7 IME, OpenMP/all-core dispatch, full engine, graph scheduler, COCO/mAP, camera/full-image path, or production/model-FPS claim was introduced.

# Stage21 Summary RU

classification: `stage21-model4-c2f-c2-integrated-ready-for-next-repair-decision`
stage_id: `BANANA-YOLO26-CUSTOM-INT8-IME-ENGINE-STAGE21-MODEL4-C2F-MERGE-REPAIR-INTEGRATION-001`
repo: `/data/banana-yolo26-spacemit-demo`
branch: `yolo26-custom-int8-engine`
start_head: `6ea3f0737c2063de94a7b4beac976180c4375872`
end_head: `d8025985bff6373aaf7082a47ad532a18bd64134`
pushed: `false`
full_engine_implemented: `false`
ncnn_source_mutated: `false`
production_claim_made: `false`

## Кратко

Stage21 перенес Stage20 repair `C2_split0_concat_lut_4t` из benchmark sidecar в реальный `/model.4` C2f runner как явный режим:

```text
Y26_STAGE16_MERGE_MODE_C2_SPLIT0_CONCAT_LUT
```

Глобальный backend/default dispatch не изменялся. Full YOLO26 engine не реализован.

## Proven

```text
host CTest: 36/36 pass
RISC-V cross build: pass
board correctness: pass
concat_mismatches: 0
model4_cv2_mismatches: 0
Stage20-compatible full-shape C2 mean_total_us: 116631
Stage20 +3% gate: pass
```

## Broken

```text
direct same-input full-shape ONNX cut proof for integrated runner: not closed in Stage21
```

## Unknown

```text
same-input ONNX cut result for the integrated runner
next dominant repair after /model.4/cv2 Conv is measured under the cut oracle
```

## Важно

Это не full YOLO26 FPS, не full-image/camera performance, не COCO/mAP и не production readiness claim.

## Next

```text
BANANA-YOLO26-CUSTOM-INT8-IME-ENGINE-STAGE22-MODEL4-ONNX-CUT-ORACLE-AND-NEXT-BLOCK-GATE-001
```

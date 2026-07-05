# Stage16A Representative Replay Correctness

subset: `candidate_I_model4_split_first_branch`
shape_class: `full_shape_model4_branch_entry`

```text
/model.4: 80x80x64
Split_output_1: 80x80x32
/model.4/m.0/cv1/conv/Conv: 80x80x16
```

## Stable Replay

| candidate | status | mismatches | checksum |
|---|---:|---:|---:|
| `scalar_reference_int8_lut` | 0 | 0 | 1324192976 |
| `stage17_IME_A2_rvv_f32_lut` | 0 | 0 | 1324192976 |

Board CPU0/1/2/3 compact Stage16 correctness smoke also passed with `mismatches=0`.

This is selected-subset evidence only. It is not full YOLO26 inference, full-image/camera performance, COCO/mAP, or model FPS.

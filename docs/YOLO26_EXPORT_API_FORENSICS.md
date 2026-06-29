# YOLO26 Export API Forensics

Raw evidence:

```text
/data/ncnn-logs/ort-logs/2026-06-29_16-03-40/
```

## Finding

The previous YOLO26 failure was a real Ultralytics package/API mismatch.

| Package | Default export | Explicit `end2end` | Canonical behavior |
|---|---|---|---|
| `8.3.233` | traditional `[1,84,N]` | rejected | false `refrigerator` |
| `8.4.0` | end-to-end `[1,300,6]` | rejected | sane default detections |
| `8.4.82` | end-to-end `[1,300,6]` | accepted | sane default and explicit paths |

`8.4.82` is the preferred R&D package for this repository because it matches the
live Ultralytics documentation: default YOLO26 is end-to-end and
`end2end=False` selects the traditional output.

## Output Contracts

- End-to-end: `[1,300,6]`, decoded as `[x1,y1,x2,y2,confidence,class_id]`.
- Traditional: `[1,84,8400]`, decoded as `cx,cy,w,h + class scores` with
  class-aware NMS.

## Evidence Tables

- `tables/ultralytics_version_export_arg_matrix.md`
- `tables/yolo26_onnx_cpu_decode_matrix.md`
- `YOLO26_CANONICAL_FALSE_REFRIGERATOR_ANALYSIS.md`

These tables live in the raw evidence directory above.

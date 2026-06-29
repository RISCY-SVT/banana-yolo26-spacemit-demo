# P0/P1/P2 Issue List

Run: `BANANA-YOLO26-INT8-RT204-OPERATOR-SUPPORT-FORENSICS-001`

Log directory:

```text
/data/ncnn-logs/ort-logs/2026-06-29_16-56-34/
```

## P0

No open P0 in the isolated YOLO26 R&D workspace.

The frozen YOLO11 production repo was verified read-only at
`production-2026-07-02` and was not modified.

## P1

No P1 blocking this R&D stage.

## P2 / R&D Blockers

- YOLO26 INT8 board EP is blocked: rt204 SpaceMIT EP fails to compile CPU-good
  Q/DQ INT8 subgraphs with `output_type not implemented for clip minmax`.
- Ultralytics `quantize=8` is not a usable YOLO26 INT8 path yet because CPU ORT
  returns zero detections for every tested Q/DQ candidate.
- YOLO11 rt204 direct tensor-probe results are promising, but require a
  separate full production adoption gate before any policy change.

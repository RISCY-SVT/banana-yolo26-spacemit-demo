# comparison_to_yolo11_report

Imported frozen YOLO11 production reference from prior final report, not rerun here:

| variant | AP | AP50 | AP75 | app forward-only FPS | app full-image FPS |
|---|---:|---:|---:|---:|---:|
| YOLO11 primary dynamic640 INT8 rt201 reference | 0.384006 | 0.539212 | 0.419599 | 5.31 | 4.35 |
| YOLO26 FP32 e2e rt204 measured | 0.404730 | 0.571221 | 0.435028 | 1.736486 | 1.931851 |
| YOLO26 FP16 keep-I/O rt204 measured | 0.404748 | 0.571417 | 0.435241 | 2.629069 | 2.528148 |

Decision: YOLO26n value is confirmed on accuracy (`AP +0.0207` over imported YOLO11 reference), while public vendor rt204 FP16 speed remains below the frozen YOLO11 INT8 production speed. This supports continuing custom-engine work only as R&D and does not create a production/default-backend claim.

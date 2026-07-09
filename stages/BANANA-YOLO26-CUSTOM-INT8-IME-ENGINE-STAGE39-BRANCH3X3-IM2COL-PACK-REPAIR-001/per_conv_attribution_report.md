# Per-Conv Attribution Report

stage_id: BANANA-YOLO26-CUSTOM-INT8-IME-ENGINE-STAGE39-BRANCH3X3-IM2COL-PACK-REPAIR-001
repo: /data/banana-yolo26-spacemit-demo
branch: yolo26-custom-int8-engine
start_head: 11675ccfbdf905bef92b5fd69f75d08a541a549c


| node | metric | Stage38 replay us | Stage39 fastpack us | speedup |
|---|---|---:|---:|---:|
| /model.4/m.0/cv1/conv/Conv | conv_total | 5991.510 | 5139.610 | 1.165752x |
| /model.4/m.0/cv1/conv/Conv | im2col_pack | 3589.480 | 3446.860 | 1.041377x |
| /model.4/m.0/cv1/conv/Conv | compute | 4089.010 | 3916.260 | 1.044111x |
| /model.4/m.0/cv2/conv/Conv | conv_total | 4299.750 | 3742.780 | 1.148812x |
| /model.4/m.0/cv2/conv/Conv | im2col_pack | 1968.590 | 1908.960 | 1.031237x |
| /model.4/m.0/cv2/conv/Conv | compute | 2778.930 | 2686.710 | 1.034325x |
| /model.4/cv2/conv/Conv | conv_total | 7868.580 | 7027.910 | 1.119619x |

Combined branch 3x3 conv total improved `1.158614x` in the measured attribution run and `1.179270x` in no-instrument timing.

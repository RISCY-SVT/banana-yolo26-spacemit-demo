# Conv IME Roofline Diagnostic

scope: selected-subset diagnostic only, not full-model roofline

| node | shape | kernel | MACs real-shape | Stage11 IME us | classification |
|---|---|---|---:|---:|---|
| `/model.0/conv/Conv` | `640x640x3 -> 320x320x16` | `3x3/s2/p1` | 44,236,800 | 68437.1 | conv/IME dominant |
| `/model.1/conv/Conv` | `320x320x16 -> 160x160x32` | `3x3/s2/p1` | 117,964,800 | 63761.7 | conv/IME dominant |
| `/model.2/cv1/conv/Conv` | `160x160x32 -> 160x160x32` | `1x1` | 26,214,400 | 25868.8 | conv/IME dominant |
| `/model.2/m.0/cv1/conv/Conv` | `160x160x16 -> 160x160x8` | `3x3/s1/p1` | 29,491,200 | 39434.4 | conv/IME dominant |
| `/model.2/m.0/cv2/conv/Conv` | `160x160x8 -> 160x160x16` | `3x3/s1/p1` | 29,491,200 | 30063.5 | conv/IME dominant |

The diagnostic uses selected-subset timing buckets. It does not estimate or claim YOLO26 full-model throughput.

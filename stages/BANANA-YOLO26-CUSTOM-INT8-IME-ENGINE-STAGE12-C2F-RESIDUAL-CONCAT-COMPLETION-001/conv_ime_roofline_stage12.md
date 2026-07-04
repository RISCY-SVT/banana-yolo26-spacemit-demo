# Conv IME Roofline Stage 12

scope: selected-subset diagnostic only
peak_reference: `2 TOPS / 1000 GMAC/s diagnostic reference`

## Stage 12 Covered Conv Nodes

| node | shape | kernel | MAC count | time_us | GMAC/s | rough_peak_pct | classification |
|---|---|---|---:|---:|---:|---:|---|
| `/model.0/conv/Conv` | `640x640x3 -> 320x320x16` | `3x3/s2/p1` | `44,236,800` | included in Stage 11 replay | n/a | n/a | carried from Stage 11 |
| `/model.1/conv/Conv` | `320x320x16 -> 160x160x32` | `3x3/s2/p1` | `117,964,800` | included in Stage 11 replay | n/a | n/a | carried from Stage 11 |
| `/model.2/cv1/conv/Conv` | `160x160x32 -> 160x160x32` | `1x1` | `26,214,400` | included in Stage 11 replay | n/a | n/a | carried from Stage 11 |
| `/model.2/m.0/cv1/conv/Conv` | `160x160x16 -> 160x160x8` | `3x3/s1/p1` | `29,491,200` | included in Stage 11 replay | n/a | n/a | carried from Stage 11 |
| `/model.2/m.0/cv2/conv/Conv` | `160x160x8 -> 160x160x16` | `3x3/s1/p1` | `29,491,200` | included in Stage 11 replay | n/a | n/a | carried from Stage 11 |
| `/model.2/cv2/conv/Conv` | `160x160x48 -> 160x160x64` | `1x1` | `78,643,200` | `50056.6` | `1.571` | `0.157%` | unclear / memory+overhead dominated |

## Decision

No Conv optimization or sliding `vmadot1/2/3` implementation is introduced in
Stage 12. Plain `smt.vmadot` MMT4D remains the implementation primitive.
The new `/model.2/cv2/conv/Conv` timing is selected-subset evidence only.

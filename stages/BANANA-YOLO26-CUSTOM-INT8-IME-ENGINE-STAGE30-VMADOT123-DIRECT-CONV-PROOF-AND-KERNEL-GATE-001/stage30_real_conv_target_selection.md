# Stage30 Real Conv Target Selection

Primary targets from the prompt:

| node | shape | kernel | MAC count | current Stage28 timing | current status |
| --- | --- | --- | ---: | ---: | --- |
| `/model.4/m.0/cv1/conv/Conv` | 80x80x32 -> 80x80x16 | 3x3 | 29,491,200 | 7,797.29 us | real 3x3 target |
| `/model.4/m.0/cv2/conv/Conv` | 80x80x16 -> 80x80x32 | 3x3 | 29,491,200 | 5,996.58 us | real 3x3 target |
| `/model.4/cv2/conv/Conv` | 80x80x96 -> 80x80x64 | 1x1 | 78,643,200 | 11,461.5 us | keep MMT4D in Stage30 |

Stage30 semantics result:

`vmadot1/2/3` are shifted-M tile instructions over an expanded A panel. They can potentially help a future packed/sliding Conv schedule by reusing a larger A panel across adjacent output-row tiles. They are not a standalone direct 3x3 Conv kernel.

Stage30 target decision:

- Do not attempt `/model.4/cv2/conv/Conv` 1x1 in this stage.
- Do not promote any `vmadot1/2/3` direct Conv into the runner in this stage.
- Recommend Stage31 as a bounded direct/sliding Conv applicability stage that starts from `/model.4/m.0/cv1/conv/Conv` and compares an expanded-A-panel schedule against current Stage28 threaded MMT4D.

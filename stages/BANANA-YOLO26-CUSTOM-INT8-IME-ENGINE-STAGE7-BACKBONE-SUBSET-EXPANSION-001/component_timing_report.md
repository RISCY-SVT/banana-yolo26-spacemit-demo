# Component Timing Report

All timings are board CPU0 selected-subset microbenchmarks only, not YOLO26 inference FPS.

## Stage 6 Replay

| bucket | us |
|---|---:|
| scalar total | `1.0121e+06` |
| IME total | `418971` |
| Conv0 IME | `67650.8` |
| activation/requant fallback | `286194` |
| Conv1 IME | `63994.2` |
| Stage5 Conv0 replay IME | `70105.5` |

## Stage 7 Candidate D

| bucket | scalar us | IME us |
|---|---:|---:|
| total | `1.22366e+06` | `593347` |
| Conv0 | `446505` | `68629.8` |
| Act0/requant | `287675` | `293585` |
| Conv1 | `281597` | `64868.9` |
| Act1/requant | `143229` | `143195` |
| Conv2 `/model.2/cv1/conv/Conv` | `63516.4` | `21933.2` |

Stage 7 speedup IME vs scalar: `2.0623x` for the selected subset. Activation/requant fallback: `73.6129%` of IME total.

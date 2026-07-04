# Stage 8 Component Timing Report

board_cpu: CPU0
selected_mode: `int8_lut`

| component | before us | after us |
|---|---:|---:|
| Conv0 IME | 68199.6 | 68356.5 |
| Act0/requant | 311321 | 128644 |
| Conv1 IME | 63955.9 | 63886.8 |
| Act1/requant | 154580 | 63924.9 |
| Conv2 IME | 21541.5 | 24143.5 |
| activation total | 465901 | 192568 |
| selected-subset IME total | 620735 | 350092 |

| metric | value |
|---|---:|
| activation reduction | 58.67% |
| total speedup vs Stage 8 baseline | 1.77307x |
| speedup vs scalar total | 3.58374x |
| activation share after | 55.0052% |

The minimum Stage 8 performance gate is met (`activation_total_us <= 220000`, `selected_subset_ime_total_us <= 400000`). The activation share remains above 40%, so graph expansion is deferred.

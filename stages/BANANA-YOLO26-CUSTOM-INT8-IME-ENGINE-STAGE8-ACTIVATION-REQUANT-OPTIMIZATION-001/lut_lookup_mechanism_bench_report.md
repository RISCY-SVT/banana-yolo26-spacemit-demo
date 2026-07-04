# LUT Lookup Mechanism Bench Report

scope: microbench only, selected-subset Act0 code stream
board_cpu: CPU0

| mechanism | us | checksum |
|---|---:|---:|
| scalar table lookup | 9057.34 | -580533438 |
| scalar unrolled4 table lookup | 8401.9 | -580533438 |
| RVV `vluxei` indexed load | not implemented | n/a |
| RVV `vrgather` register gather | not implemented | n/a |

Conclusion: simple scalar lookup is not the remaining dominant cost by itself. Remaining time is primarily conv-output code quantization plus table lookup/write. No RVV gather performance claim was made.

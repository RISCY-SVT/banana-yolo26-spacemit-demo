# RT204 versus RT205 repro difference

| Surface | RT204 | RT205 |
|---|---|---|
| explicit kernel_shape Q/DQ Conv | clip-minmax exception | same error then abort |
| Q/DQ Conv without kernel_shape | pass | pass, exact |
| QLinearConv | pass | core dump |
| QLinearMatMul | pass | SIGILL (132) |
| primary Q/DQ full model | first-Conv compile failure | first-Conv compile failure then abort |
| broad historical CPU filter | executes CPU-heavy | SIGILL/core dump |

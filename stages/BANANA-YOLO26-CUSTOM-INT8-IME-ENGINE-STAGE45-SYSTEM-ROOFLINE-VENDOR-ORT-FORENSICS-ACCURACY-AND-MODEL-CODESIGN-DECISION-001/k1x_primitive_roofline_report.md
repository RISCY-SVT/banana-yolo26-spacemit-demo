# K1X primitive roofline

Measured on CPU0 under `steady_clock`; no `rdcycle` was used. At 32 MiB, memcpy
delivered `5.045716 GB/s` under a read+write byte contract and sequential writes
`7.341689 GB/s`. The scalar checksum read probe (`0.866986 GB/s`) is instruction
limited and is not claimed as LPDDR peak. Existing scalar layout/activation
primitives are costly: NCHW->NHWC `9326.987 us`, uint8 LUT `4684.663 us`, exact
fixed requant `5454.583 us`, and stride2 panel work `12139.443 us` on the stated
surfaces. This supports resident INT8 layouts and fused integer post-processing.

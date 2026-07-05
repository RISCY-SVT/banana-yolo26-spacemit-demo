# Stage16A Representative Component Timing

Protocol: `warmup=10 runs=100 repeats=5`

| candidate | mean_total_us | stddev_total_us | min_total_us | max_total_us | cv_total_pct | mean_conv_us | mean_activation_requant_us | mean_split_us | mean_correction_us | conv_share_pct | activation_share_pct |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `scalar_reference_int8_lut` | 90950.820736 | 37.846523 | 90918.894880 | 91013.761960 | 0.041612 | 50738.654554 | 40039.630260 | 170.835002 | 215.210720 | 55.786912 | 44.023385 |
| `stage17_IME_A2_rvv_f32_lut` | 25670.974780 | 345.670192 | 25422.612850 | 26158.283980 | 1.346541 | 20458.001284 | 5035.606050 | 175.701976 | 216.976304 | 79.693122 | 19.615952 |

Stage17 stable replay differs from the Stage16A one-shot by less than 1% for the IME total and therefore confirms the earlier conclusion with a proper repeat protocol.

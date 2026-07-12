# Graph Shape Class Report

- Compute nodes: `106`
- Conv nodes: `102`
- MatMul/Gemm nodes: `4`
- Static MACs: `2740153600`
- Shapes, divisibility, groups, and bytes are derived from the accepted ONNX bytes.
- `m12n16_fullshape` is a candidate mapping, not measured proof until board results are joined.

| class | nodes | MACs | MAC share |
|---|---:|---:|---:|
| 1x1_high_resolution | 12 | 481689600 | 17.578927% |
| 1x1_low_resolution | 40 | 718643200 | 26.226384% |
| 3x3_stride1 | 31 | 564940800 | 20.617122% |
| 3x3_stride2 | 7 | 869990400 | 31.749695% |
| grouped_or_depthwise_conv | 8 | 13420800 | 0.489783% |
| matmul_attention | 4 | 61440000 | 2.242210% |
| small_n_head_conv | 4 | 30028800 | 1.095880% |

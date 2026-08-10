# Graphwise hotspots

Thresholds are diagnostic: SNR >= 0.1 and cosine < 0.99. They do not establish causality.

| lane | pyramid | branch | op | tensor | SNR | cosine |
|---|---|---|---|---|---:|---:|
| B6 |  |  | /model.22/Concat[Concat] | /model.22/Split_output_0 | 0.319 | 0.8336 |
| B6 |  |  | /model.10/m/m.0/attn/proj/conv/Conv[Conv] | /model.10/m/m.0/attn/proj/conv/Conv_output_0 | 0.3128 | 0.839 |
| B6 |  |  | /model.10/m/m.0/ffn/ffn.1/conv/Conv[Conv] | /model.10/m/m.0/ffn/ffn.1/conv/Conv_output_0 | 0.3091 | 0.8335 |
| B6 |  |  | /model.22/cv1/act/Mul[Mul] | /model.22/cv1/act/Mul_output_0 | 0.301 | 0.8405 |
| B6 |  |  | /model.10/m/m.0/attn/Add[Add] | /model.10/m/m.0/attn/Reshape_1_output_0 | 0.2949 | 0.8464 |
| B6 |  |  | /model.10/m/m.0/attn/Add[Add] | /model.10/m/m.0/attn/Add_output_0 | 0.2852 | 0.8518 |
| B5 |  |  | /model.22/Concat[Concat] | /model.22/Split_output_0 | 0.2844 | 0.8511 |
| B6 |  |  | /model.10/m/m.0/attn/MatMul_1[MatMul] | /model.10/m/m.0/attn/MatMul_1_output_0 | 0.2742 | 0.8573 |
| B3 | P5 | confidence | /model.23/one2one_cv3.2/one2one_cv3.2.1/one2one_cv3.2.1.1/conv/Conv[Conv] | /model.23/one2one_cv3.2/one2one_cv3.2.1/one2one_cv3.2.1.1/conv/Conv_output_0 | 0.2626 | 0.8894 |
| B1 | P5 | confidence | /model.23/one2one_cv3.2/one2one_cv3.2.1/one2one_cv3.2.1.0/conv/Conv[Conv] | /model.23/one2one_cv3.2/one2one_cv3.2.1/one2one_cv3.2.1.0/conv/Conv_output_0 | 0.2598 | 0.8604 |
| B6 |  |  | /model.8/m.0/cv2/act/Mul[Mul] | /model.8/m.0/cv2/act/Mul_output_0 | 0.2538 | 0.8803 |
| B5 | P4 | confidence | /model.23/one2one_cv3.1/one2one_cv3.1.1/one2one_cv3.1.1.1/conv/Conv[Conv] | /model.23/one2one_cv3.1/one2one_cv3.1.1/one2one_cv3.1.1.1/conv/Conv_output_0 | 0.2531 | 0.8697 |
| B6 |  |  | /model.9/cv1/conv/Conv[Conv] | /model.9/cv1/conv/Conv_output_0 | 0.2525 | 0.8704 |
| B6 |  |  | /model.10/cv2/act/Mul[Mul] | /model.10/cv2/act/Mul_output_0 | 0.2487 | 0.87 |
| B5 |  |  | /model.22/cv1/act/Mul[Mul] | /model.22/cv1/act/Mul_output_0 | 0.2468 | 0.8703 |
| B6 |  |  | /model.10/m/m.0/Add[Add] | /model.10/m/m.0/Add_output_0 | 0.2453 | 0.8725 |
| B6 |  |  | /model.10/m/m.0/Add_1[Add] | /model.10/m/m.0/Add_1_output_0 | 0.2452 | 0.8723 |
| B3 | P5 | confidence | /model.23/one2one_cv3.2/one2one_cv3.2.1/one2one_cv3.2.1.0/conv/Conv[Conv] | /model.23/one2one_cv3.2/one2one_cv3.2.1/one2one_cv3.2.1.0/conv/Conv_output_0 | 0.2447 | 0.8696 |
| B6 |  |  | /model.8/m.0/cv2/conv/Conv[Conv] | /model.8/m.0/cv2/conv/Conv_output_0 | 0.244 | 0.8736 |
| B6 |  |  | /model.10/m/m.0/attn/pe/conv/Conv[Conv] | /model.10/m/m.0/attn/Reshape_2_output_0 | 0.243 | 0.8726 |
| B6 | P5 | confidence | /model.23/one2one_cv3.2/one2one_cv3.2.1/one2one_cv3.2.1.1/conv/Conv[Conv] | /model.23/one2one_cv3.2/one2one_cv3.2.1/one2one_cv3.2.1.1/conv/Conv_output_0 | 0.2421 | 0.8808 |
| B5 | P5 | confidence | /model.23/one2one_cv3.2/one2one_cv3.2.1/one2one_cv3.2.1.1/conv/Conv[Conv] | /model.23/one2one_cv3.2/one2one_cv3.2.1/one2one_cv3.2.1.1/conv/Conv_output_0 | 0.2418 | 0.8901 |
| B5 |  |  | /model.10/m/m.0/attn/proj/conv/Conv[Conv] | /model.10/m/m.0/attn/proj/conv/Conv_output_0 | 0.2394 | 0.8764 |
| B6 | P4 | confidence | /model.23/one2one_cv3.1/one2one_cv3.1.1/one2one_cv3.1.1.1/conv/Conv[Conv] | /model.23/one2one_cv3.1/one2one_cv3.1.1/one2one_cv3.1.1.1/conv/Conv_output_0 | 0.2389 | 0.8776 |
| B6 |  |  | /model.10/m/m.0/attn/MatMul_1[MatMul] | /model.10/m/m.0/attn/Split_output_2 | 0.2283 | 0.8809 |
| B6 | P5 | confidence | /model.23/one2one_cv3.2/one2one_cv3.2.1/one2one_cv3.2.1.0/conv/Conv[Conv] | /model.23/one2one_cv3.2/one2one_cv3.2.1/one2one_cv3.2.1.0/conv/Conv_output_0 | 0.2261 | 0.8799 |
| B5 |  |  | /model.2/cv1/conv/Conv[Conv] | /model.2/cv1/conv/Conv_output_0 | 0.2253 | 0.8865 |
| B5 |  |  | /model.0/conv/Conv[Conv] | /model.0/conv/Conv_output_0 | 0.2237 | 0.8956 |
| B6 |  |  | /model.13/m.0/cv2/act/Mul[Mul] | /model.13/m.0/cv2/act/Mul_output_0 | 0.2197 | 0.8841 |
| B6 |  |  | /model.10/Concat[Concat] | /model.10/Concat_output_0 | 0.2188 | 0.8873 |

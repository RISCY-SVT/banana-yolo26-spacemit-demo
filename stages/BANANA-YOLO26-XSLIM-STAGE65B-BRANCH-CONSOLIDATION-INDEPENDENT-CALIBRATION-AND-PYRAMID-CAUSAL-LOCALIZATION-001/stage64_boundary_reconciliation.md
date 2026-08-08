# Stage64 boundary reconciliation

Stage64 recorded the P5-confidence observation summarized by the launch
contract: cosine approximately `0.987823`, normalized MAE approximately
`0.555`, FP32 maximum above the quantized ceiling on `79/100` images, S8
maximum at the ceiling on `80/100`, and mean-logit bias approximately
`+1.546`.

Stage65B did not generate an independent model or run the required 500-image
boundary audit. Therefore these historical correlations were not reconciled
against an independent corpus and no causal conclusion is made.

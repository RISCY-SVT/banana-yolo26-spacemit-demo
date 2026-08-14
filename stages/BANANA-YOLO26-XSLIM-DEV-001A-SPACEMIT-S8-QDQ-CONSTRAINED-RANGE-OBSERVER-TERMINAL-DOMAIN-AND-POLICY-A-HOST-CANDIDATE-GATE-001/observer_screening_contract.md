# Observer Screening Contract

The screening surface was frozen before reading any A1-A6 task metric. It uses the exact Stage65B-R3 source/QDQ correspondence for R0 and R7 and the exact Stage65B-R1 terminal domains for T6.

For each selected tensor, 500 H500 activations were sampled with the accepted project preprocessing. R0/R7 hashes were reconciled against the accepted four-thread R3 oracle; T6 hashes were independently reconciled against the accepted one-thread R1 oracle. The raw sample arrays remain under the stage raw root and are not tracked.

The bounded methods were `default`, `minmax`, percentiles `0.999`, `0.9995`, and `0.9999`, `mse`, `kl`, and `constrained-mse`. Every non-default trial preserved real zero, the evidence-derived positive headroom, and the configured SiLU floor for activation-prefix tensors. T6 additionally preserved its observed negative and positive terminal bounds. The numeric simulator used signed per-tensor INT8 (`[-128, 127]`), round-to-nearest-even, and rail saturation.

Selection was deterministic: reject any constraint failure, then minimize mean normalized MAE, clipping fraction, and method name. `kl` won all four groups. This is a qparam/proxy decision only; H500 and full-val task metrics remain independent later gates.

The first preparation attempt was isolated before selection because it compared four-thread samples directly with a one-thread T6 oracle. The successful `observer-preparation-v2` run explicitly validated both hash surfaces and produced the frozen manifest recorded in `policy_target_manifest.sha256`.

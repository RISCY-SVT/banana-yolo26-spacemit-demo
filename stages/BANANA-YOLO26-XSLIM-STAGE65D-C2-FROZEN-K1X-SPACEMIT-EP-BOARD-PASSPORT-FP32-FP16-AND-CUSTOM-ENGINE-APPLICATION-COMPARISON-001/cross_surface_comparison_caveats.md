# Cross-Surface Comparison Caveats

The accepted custom engine and C2 SpaceMIT EP are application-level surfaces.
They do not share one exported/quantized model or one backend contract.

Consequently:

- accuracy deltas are application-level observations, not quantizer-only
  effects;
- latency deltas are application-level context, not engine-only speedups;
- each row retains its model SHA, export lineage, quantization format, runtime,
  preprocessing and evaluator identity;
- native accepted configurations are reported separately from any symmetric
  affinity context;
- no result authorizes replacing the custom executor or promoting C2.

A true backend comparison requires a separately authorized same-source
K1X_INT8_V2 custom-executor conversion.

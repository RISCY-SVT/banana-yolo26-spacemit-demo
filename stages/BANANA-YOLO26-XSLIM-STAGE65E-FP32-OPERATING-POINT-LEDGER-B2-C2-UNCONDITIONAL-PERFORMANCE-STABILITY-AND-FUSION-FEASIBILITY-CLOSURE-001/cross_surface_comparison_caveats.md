# Cross-surface comparison caveats

The accepted custom executor and the B2/C2 vendor lane use different model/export lineages, quantization formats, runtime backends, and output implementations. Their same-boot fixed-input timings and accepted COCO metrics are application-level context, not an engine-only or quantizer-only comparison. The frozen vendor timing contract uses CPU0-3, while the accepted custom low-latency profile natively uses CPU0-4; rows remain separate and no direct speedup ratio is claimed. No custom model or executable was rebuilt.

# XSlim policy versus exported graph

## Source policy

The inspected XSlim policy selects signed INT8 ranges `[-128, 127]`.
Activations use asymmetric per-tensor quantization. Conv weights use symmetric
per-channel quantization on output-channel axis 0. MatMul/Gemm weights use
symmetric per-tensor quantization, including the configured power-of-two
policy. Bias remains float32 in the measured configurations.

## Official 2.1.1 project-exact export

| Property | Measured export |
|---|---:|
| graph nodes | 1,195 |
| Conv | 102 |
| MatMul | 4 |
| QuantizeLinear | 354 |
| DequantizeLinear | 458 |
| all Q/DQ nodes | 812 |
| QLinear operators | 0 |
| UINT8 zero points | 0 |
| signed symmetric weight sites | 102 |
| signed activation sites | 710 |
| Conv with explicit valid `kernel_shape` | 102 / 102 |

All weight zero points are signed INT8 zero. Activation zero points are signed
INT8 scalars; the input site is `-128`, and both zero and nonzero signed values
occur elsewhere. This reconciles the vendor wording: "asymmetric" describes
the quantizer capability/policy and does not require every observed activation
zero point to be nonzero.

The exported ONNX tensors, not source defaults, are the conformance authority.

# Custom Model4 Cut Contract

The custom block remains the selected `/model.4` same-input ONNX cut only.

```text
input boundary:  /model.4/cv1/conv/Conv_output_0_QuantizeLinear_Output
input layout:    NCHW in ONNX, NHWC uint8 bin for C++ runner handoff
input shape:     ONNX 1x64x80x80, runner bin 1x80x80x64
output boundary: /model.4/cv2/conv/Conv_output_0_QuantizeLinear_Output
output layout:   NCHW in ONNX, NHWC uint8 bin from C++ runner
output shape:    ONNX 1x128x80x80, runner bin 1x80x80x128
selected mode:   Y26_STAGE16_MERGE_MODE_STAGE39_BRANCH3X3_FAST_PACK
output quantize: Y26_STAGE16_OUTPUT_QUANTIZE_STAGE38_RVV_DIRECT_STORE
threading:       explicit CPU0-3 only
```

Stage40 board run used the prefix-generated input fixture:

```text
input_nhwc_sha256:  94dcc2e954f7b9b112bc3f11cfe4a147dfe91c52b37f94058e2dd09b4e08b1b8
output_nhwc_sha256: 517db620fca8465888ec387673f888d5e7c43c86d613c88cbf4bb5ffcbe4cd91
mismatches:         0
max_abs_diff:       0
frm_sweep:          pass
affinity_ok:        1
```

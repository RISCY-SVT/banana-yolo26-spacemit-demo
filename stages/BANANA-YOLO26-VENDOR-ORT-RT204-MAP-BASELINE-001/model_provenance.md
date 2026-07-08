# model_provenance

| model | path | sha256 | status |
|---|---|---|---|
| YOLO26n checkpoint | .deps/models/yolo26/yolo26n.pt | 9b09cc8bf347f0fc8a5f7657480587f25db09b34bf33b0652110fb03a8ad4fef | matches expected |
| YOLO26n FP32 e2e 640 | .deps/models/yolo26/fp32_fp16_xslim_effect_matrix/yolo26n_640_e2e_fp32.onnx | d71286588abe691ede49faa5ca9a471b7e9e5257669953ee59abbc2e9d115fc2 | measured full COCO |
| YOLO26n FP16 body/head keep-I/O 640 | .deps/models/yolo26/fp32_fp16_xslim_effect_matrix/yolo26n_640_e2e_native_fp16_body_headfp32_keep_io.onnx | f515005e1cfeb19ce320408d44f79d8439774f4deaff188eed3ca70b258ff6d4 | measured full COCO, best speed |
| YOLO26n FP16 full-I/O 640 | .deps/models/yolo26/fp32_fp16_xslim_effect_matrix/yolo26n_640_e2e_native_fp16_body_headfp32_full_io.onnx | 91202757f0971db1f1e866d69d1d2e7f6261cfc85d5896fc11e0e72a27a1db3f | not rerun; prior accepted secondary |

Export/API line preserved: Ultralytics current e2e output contract `[1,300,6]`.

# Vendor message transcript

The following public comments were captured through the GitHub API before the
Stage64 tests. They are preserved as claims to test, not treated as measured
results.

## SpacemiT ONNX Runtime issue 1

- Repository: `spacemit-com/onnxruntime`
- Issue: `1`
- Comment ID: `5125717846`
- Author: `alex-spacemit`
- Created: `2026-07-30T02:36:37Z`

Verbatim body:

> Here is the current quantization format support status of the spacemit-ep provider:
>
> 1. For feature activations: We support asymmetric int8 per-tensor quantization represented by QDQ nodes. The model you provided uses symmetric uint8 per-tensor quantization, which is not compatible with our current implementation.
> 2. For weights: We support symmetric int8 per-tensor or per-channel quantization represented by QDQ nodes, while your model adopts symmetric uint8 per-tensor quantization. The root cause of the incompatibility for both points above is that uint8 zero-point is not yet supported.
> 3. QLinearOperator was supported in the earlier 1.x releases. Starting from the 2.x versions, we have standardized on QDQ Operator as the unified quantization format.
>
> For detailed quantization format specifications, please refer to:
> `https://github.com/spacemit-com/xslim/blob/9a33f2f770d00fd02ff8bc0f1907135e9bf47f8c/src/xslim/quantizer/xslim.py#L232`

## XSlim issue 45

- Repository: `spacemit-com/xslim`
- Issue: `45`
- Comment ID: `4955022265`
- Author: `alex-spacemit`
- Created: `2026-07-13T06:21:49Z`

Verbatim body:

> For the YOLO26 model, you can use truncate_var_names to split the model into two parts. Only the inference part should be quantized, while the post-processing part should remain unquantized.
> For YOLO models, it is also recommended not to concatenate the bbox and confidence branches before quantization. With per-tensor quantization, the scales of the confidence and bbox outputs differ significantly, and combining them can cause a substantial accuracy drop.
> You can refer to the following configuration:
>
> ```json
> {
>   "model_parameters": {
>     "onnx_model": "yolo26n_e2e.onnx",
>     "working_dir": "output"
>   },
>   "calibration_parameters": {
>     "calibration_step": 50,
>     "input_parameters": [
>       {
>         "mean_value": [0, 0, 0],
>         "std_value": [255, 255, 255],
>         "color_format": "rgb",
>         "data_list_path": "calib_img_list.txt"
>       }
>     ]
>   },
>   "quantization_parameters": {
>     "truncate_var_names": [
>       "/model.23/one2one_cv2.0/one2one_cv2.0.2/Conv_output_0",
>       "/model.23/one2one_cv3.0/one2one_cv3.0.2/Conv_output_0",
>       "/model.23/one2one_cv2.1/one2one_cv2.1.2/Conv_output_0",
>       "/model.23/one2one_cv3.1/one2one_cv3.1.2/Conv_output_0",
>       "/model.23/one2one_cv2.2/one2one_cv2.2.2/Conv_output_0",
>       "/model.23/one2one_cv3.2/one2one_cv3.2.2/Conv_output_0"
>     ]
>   }
> }
> ```
>
> After truncating the model with truncate_var_names, the issue you encountered will be bypassed. However, the issue related to the Reduce operators still remains. I will open a PR to address it.

## Interpretation boundary

Stage64 tests these claims at four distinct layers:

1. emitted ONNX representation and signedness;
2. tiny operator correctness and placement;
3. complete split-pipeline provider placement and correctness;
4. stability, timing, and COCO task accuracy.

The comments do not establish that the official XSlim 2.1.1 release contains
the later ReduceMax source repair. That question is tested independently.

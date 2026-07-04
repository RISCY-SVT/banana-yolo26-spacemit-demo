# Residual Add Contract Report

status: `deferred`

## ONNX Contract

`/model.2/m.0/Add` inputs:

- `/model.2/Split_output_1_DequantizeLinear_Output`
- `/model.2/m.0/cv2/act/Mul_output_0`

The Add output feeds `/model.2/Concat` before a later Concat Q/DQ boundary.

## Decision

Residual Add was not implemented in Stage 11. The Add is float-domain in the accepted Q/DQ ONNX graph and does not expose a clean integer Add output contract before Concat. Implementing it now would either introduce a hidden float path into the branch merge or force a broader Concat/QDQ contract, which is Stage 12 scope.

recommended_next: define `/model.2/m.0/cv2` activation + Add + Concat boundary together with ONNX CPU oracle.

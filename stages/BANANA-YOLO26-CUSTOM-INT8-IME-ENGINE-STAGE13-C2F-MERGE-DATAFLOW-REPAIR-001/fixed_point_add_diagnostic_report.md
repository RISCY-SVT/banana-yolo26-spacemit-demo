# Fixed Point Add Diagnostic Report

`A5_fixed_point_int_domain_add` was not attempted.

Reason:

- Stage 12 proved `/model.2/m.0/Add` is float-domain in the accepted Q/DQ graph.
- Stage 13 forbids accepting an integer-domain Add shortcut without an ONNX-equivalent oracle.
- A0/A1/A2 already preserved exact oracle behavior with `mismatches=0`.

Future integer Add work requires a separate proof that Add, Concat Q/DQ, and
`/model.2/cv2/conv/Conv` corrected int32 output all remain exact.

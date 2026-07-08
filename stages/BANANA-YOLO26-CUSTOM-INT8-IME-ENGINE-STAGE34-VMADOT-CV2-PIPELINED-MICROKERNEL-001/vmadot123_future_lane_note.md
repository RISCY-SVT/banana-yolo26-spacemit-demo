# VMADOT123 Future Lane Note

Stage31 proved `smt.vmadot1/2/3` semantics but the attachable direct/sliding real-node sidecar was dominated by panel-build overhead.

Stage32 rejected low-overhead sliding layout for now.

Stage34 does not implement `smt.vmadot1/2/3`, does not integrate direct/sliding Conv, and does not expand graph coverage.

Future `vmadot1/2/3` work should reopen only if a source-backed layout proof removes panel-build overhead and beats current threaded MMT4D on a real node under the same-input ONNX-cut gate.

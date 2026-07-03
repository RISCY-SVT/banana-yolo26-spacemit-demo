# Custom INT8 Engine Tools

Stage 0 tools are converter/oracle helpers only. They may depend on Python
packages such as `onnx`, `numpy`, or `onnxruntime`; the runtime library must not.

Use `inspect_onnx.py` for graph inventory and `dump_qdq_metadata.py` for Q/DQ
scale and zero-point recovery. Do not use xslim-derived YOLO26 artifacts as
authority for this engine path.

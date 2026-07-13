# Arena lifetime

Tensor first/last operation and arena offsets are recorded in `model4_8_tensor_manifest.tsv`. The generated schedule reuses non-overlapping ranges and preserves branch values through their last consumer. Host tests cover arena aliasing; runtime Conv APIs reject input/output overlap unless explicitly represented as a view.

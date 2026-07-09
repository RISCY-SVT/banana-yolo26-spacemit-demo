# Stage38 Hypotheses

stage_id: BANANA-YOLO26-CUSTOM-INT8-IME-ENGINE-STAGE38-POST-3X3-PIPELINED-BOTTLENECK-GATE-001

H1: Stage37 A1 replay remains byte-exact against same-input ONNX cut.

H2: im2col/pack is a material hidden sub-bucket inside at least one branch 3x3 Conv path.

H3: output QuantizeLinear remains a material bucket around 18-22% and may be the cheapest next local repair if replay confirms it.

H4: cluster1 non-IME offload may hide part of output_quantize/activation/merge/input-adapter work, but shared LPDDR may cap or reverse the win.

H5: only one repair lane may be selected in Stage38; no multi-lane stacking.

# Full-graph correctness

The frozen `K1X_INT8_V1_YOLO26N_640_FULL_GRAPH_001` profile contains 215 static integer operations and 215 package-defined integer tensor boundaries. The fixed detector selector converts those boundaries to the deterministic `1x300x6` output.

The independent Python audit checked the package and every operator class without ONNX Runtime arithmetic: 61,404 dense-Conv accumulator vectors across 8,772 output channels, 87 integer LUT assets, four MatMul descriptors, two Q48 Softmax surfaces, two frozen Resize branches, and the TopK/Gather selector contract. All checks passed.

For F0-F7, portable host scalar, host optimized, board scalar, and board optimized execution produced byte-identical files at all 215 boundaries. The full board state probe repeated F0 under ambient FRM RNE, RTZ, RDN, RUP, and RMM; every run produced `0xd43f5e018b415631`, restored FRM and `vcsr`, retained CPU affinity, and reported zero IME execution on CPU4-7.

The bus and Zidane public-image fixtures pass all host boundaries; board copies are retained in the raw evidence. The complete COCO prediction JSON was regenerated twice and was byte-identical with SHA-256 `cda5c8c7a46d61d9c90f6292001eea190cb8f6617efe647a33dc6134dd57ccda`.

Legacy float-QDQ ORT is not the exact integer authority. Its output remains an accuracy and integration diagnostic, as defined in `full_graph_legacy_qdq_diagnostic.tsv`.

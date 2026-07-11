# COCO2017 surface contracts

All accuracy rows use the official 5000-image COCO val2017 set and matching
instances annotations. Preprocessing is OpenCV linear letterbox with value 114,
RGB order, NCHW float32, and division by 255. The e2e output contract is
`output0` float32 `1x300x6`; confidence threshold is 0.001 and maximum detections
is 300. Host rows use ORT 1.27.0 CPUExecutionProvider. Board rows use matched
RT204/RT205 package CPU sessions, intra=4/inter=1, on CPU0-3.

Host prediction environment: Python 3.12.3, OpenCV 4.13.0, NumPy 2.5.0,
Pillow 12.2.0. Evaluation environment: Python 3.12.3, NumPy 2.5.1,
pycocotools 2.0.11. The board C++ predictor uses the repository's frozen
letterbox/decode implementation and stage-owned OpenCV shared libraries.

SpacemiT EP rows are `not-runnable`, not zero-accuracy measurements. The first
quantized Conv fails before a full output. CPU-bad negative-control models are
not promoted into the full evaluation matrix.

# Accuracy surface contracts

All three host surfaces used the same deterministic first 500 COCO val2017 images,
annotations, e2e 300x6 output decoder, confidence 0.001, class mapping, and
letterbox/RGB/NCHW `/255` preprocessing.

- FP32 operational: model `d71286588abe691ede49faa5ca9a471b7e9e5257669953ee59abbc2e9d115fc2`, host ORT 1.27.0 CPU EP, ENABLE_ALL.
- INT8 semantic: model `30a94e4738606673b5e0a73499cbc977167f046f8fa8637d6040ce744f429c0c`, host ORT 1.27.0 CPU EP, DISABLE_ALL.
- INT8 operational: same INT8 bytes/runtime, ENABLE_ALL.

This is a 500-image directional audit. It is not full val2017 and is not custom
board-engine mAP because no full custom engine exists.

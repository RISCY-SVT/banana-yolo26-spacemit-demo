# Preprocessing contract

Every calibration, holdout, Graphwise, boundary, scout, and COCO surface uses
the accepted project-exact 640x640 transform:

1. OpenCV color-image decode.
2. Aspect-preserving resize using min(640 / width, 640 / height).
3. Symmetric letterbox padding with value 114.
4. BGR to RGB conversion.
5. Float32 conversion and division by 255.
6. Contiguous NCHW 1x3x640x640 input.

The source implementation is
vendor_ort_validation/stage64_preprocess.py. The XSlim PTQ callback and the
one-path boundary-audit adapter both call the same letterbox_rgb_nchw
function. The adapter parity test is byte-exact.

The 1,515-image union of C1000, C500-size-balanced, and H500 was processed
twice. The complete reports were byte-identical with SHA-256
dedb7052a2ac19c75d5ab47c075c8d9a64f3a286198e4b4cb4ca785e5c71476d.

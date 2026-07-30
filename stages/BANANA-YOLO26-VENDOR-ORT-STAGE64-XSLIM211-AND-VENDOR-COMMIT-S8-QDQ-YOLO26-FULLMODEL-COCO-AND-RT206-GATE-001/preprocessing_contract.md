# Preprocessing and calibration contract

## Corpus

The deterministic calibration list contains 50 sorted images. The disjoint
holdout list contains 100 images; path overlap is zero. Every file is covered
by a SHA-256 manifest.

The available calibration corpus is a 2,015-image subset of COCO val2017.
Consequently, the 50 calibration images also occur in the final 5,000-image
COCO evaluation set. This leakage is explicit: final COCO measures the
requested vendor reproduction but is not an independent calibration/evaluation
split. A promotion review needs a separately licensed calibration corpus.

## Vendor-literal lane

The release implementation reads with OpenCV, converts BGR to RGB, directly
resizes to `640x640`, converts to `float32`, and applies division by 255 from
the supplied `std_value`. It does not preserve aspect ratio or apply the
project letterbox.

## Project-exact lane

The stage-local callback performs:

```text
decode
aspect-preserving resize
640x640 symmetric letterbox
pad value 114
BGR to RGB
float32 / 255
NCHW 1x3x640x640
```

It matches the accepted project bytes on F0, bus, Zidane, the canonical
fixture, ten calibration images, and ten holdout images: shape and dtype match,
mismatch count is zero, and maximum absolute difference is zero.

The vendor-literal and project-exact lanes are separate measured calibration
surfaces. Their results are never averaged.

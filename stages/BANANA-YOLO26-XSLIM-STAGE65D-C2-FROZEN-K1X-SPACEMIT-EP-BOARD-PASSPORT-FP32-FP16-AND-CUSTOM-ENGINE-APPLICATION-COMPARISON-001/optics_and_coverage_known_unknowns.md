# Optics And Coverage Known/Unknowns

## Known

- The evaluated model input is 640x640 under the frozen preprocessing contract.
- Stage65D task metrics use COCO pixel-area size bins.
- Vendor B2/C2 and the accepted custom-engine package are different model and
  runtime surfaces; their application table is not an engine-only comparison.

## Unknown

- Critical production classes and confusion costs.
- Target physical dimensions and distance distribution.
- Required horizontal and vertical field of view.
- Sensor resolution, pixel pitch, lens, focus and zoom behavior.
- Minimum target pixel dimensions after resize/letterbox.
- Lighting, motion blur, occlusion and weather envelope.
- Camera placement, vibration and bandwidth constraints.
- Labeled application-camera false-negative and false-positive rates.
- Time-to-first-detection and track-continuity requirements.

These measurements are required before choosing fixed telephoto, dual-camera,
high-resolution ROI, or PTZ coverage.

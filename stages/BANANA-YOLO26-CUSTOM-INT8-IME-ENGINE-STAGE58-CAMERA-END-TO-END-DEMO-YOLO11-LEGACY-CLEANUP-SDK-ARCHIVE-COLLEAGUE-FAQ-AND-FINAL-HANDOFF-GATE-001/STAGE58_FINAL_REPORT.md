# Stage58 Final Report

Classification: `stage58-camera-handoff-complete-with-hardware-limitations`.

The active YOLO11/vendor runtime surface is removed, ABI1 release 0.9.1 is additive-compatible, and the real BPI-F3 camera demo is measured at 5.916864 processed/displayed FPS for 1280x720@60.000 MJPG.

The model remains fixed at 640x640 letterboxed input. COCO remained 5000/5000 and byte-identical at mAP50-95 0.3707408944391919.

The connected camera was physically fixed on one wall scene, so the requested five-class/three-scene real-camera envelope could not be completed remotely; COCO size-bin evidence and the exact scene limitation are explicit.

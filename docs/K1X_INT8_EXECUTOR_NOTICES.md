# K1X INT8 Executor Notices

This handoff contains project-built executor binaries, public headers, a frozen
integer model package, documentation, and safe smoke fixtures. It does not
contain ONNX Runtime, COCO data, vendor runtime archives, private logs, or build
trees.

The camera demo dynamically links against the five OpenCV runtime libraries
included under `opencv/lib`. Their license is preserved under `licenses/`.
The core executor libraries do not depend on OpenCV.

Stage62 adds the requested AGPL-3.0-or-later technical complete-source surface.
`LEGAL_STATUS.md` records that no Enterprise agreement, ownership certification,
or complete model-rights clearance was found, so legal clearance is not
certified. This is not an extra restriction on AGPL rights.

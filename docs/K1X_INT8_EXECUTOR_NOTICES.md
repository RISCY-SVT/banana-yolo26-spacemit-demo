# K1X INT8 Executor Notices

This handoff contains project-built executor binaries, public headers, a frozen
integer model package, documentation, and safe smoke fixtures. It does not
contain ONNX Runtime, COCO data, vendor runtime archives, private logs, or build
trees.

The camera demo dynamically links against the five OpenCV runtime libraries
included under `opencv/lib`. Their license is preserved under `licenses/`.
The core executor libraries do not depend on OpenCV.

The source repository does not currently declare a top-level redistribution
license. Treat this bundle as an internal engineering handoff until the project
owner supplies release licensing. Model redistribution remains subject to the
model source and training-data terms selected by the project owner.

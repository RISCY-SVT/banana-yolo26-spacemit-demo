# Stage58 Demo Architecture

OpenCV V4L2 capture/image/video input feeds exact 640x640 letterbox preprocessing, the frozen C ABI RGB path, output deletterboxing, deterministic rendering, and an optional GUI or recording sink.

Latest-frame mode uses one capture thread and a replaceable queue of depth one; sequential mode intentionally processes each returned frame.

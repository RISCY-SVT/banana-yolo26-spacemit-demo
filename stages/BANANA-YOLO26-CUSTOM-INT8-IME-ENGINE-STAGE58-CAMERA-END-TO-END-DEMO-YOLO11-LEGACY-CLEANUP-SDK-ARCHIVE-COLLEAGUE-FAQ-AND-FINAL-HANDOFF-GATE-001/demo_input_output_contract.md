# Demo Input and Output Contract

BGR camera pixels are aspect-resized, padded with 114, converted to interleaved RGB8, and passed to `y26_executor_run_rgb`. The executor output is 300 rows of x1,y1,x2,y2,confidence,class in letterbox coordinates.

The demo performs confidence/finite/class validation and deletterboxing, but never runs a second NMS.

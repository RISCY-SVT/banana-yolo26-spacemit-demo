# Stage41 Hypotheses

H1: A native in-process C++ runner with custom `/model.4` and ORT CPU fallback can reproduce full ORT CPU `output0` byte-exactly without Python or file handoff in the runtime path.

H2: In-process C++ tensor handoff will expose real runtime overheads hidden by the Stage40 Python/file skeleton and is the correct foundation for maximum-speed YOLO26 INT8 runtime.

H3: Whole-suffix ranking will show a dominant post-`/model.4` target or block group, and that target may not be the immediate `/model.5` node.

H4: The next custom block must be selected by board-side in-process timing and clean Q/DQ boundary contracts, not graph order alone.

H5: The final high-speed engine must keep quantized tensors resident in memory as u8/int8 plus scale/zp metadata, with no Python, no file I/O, no repeated NCHW/NHWC boundary churn, and no unnecessary float round-trip.

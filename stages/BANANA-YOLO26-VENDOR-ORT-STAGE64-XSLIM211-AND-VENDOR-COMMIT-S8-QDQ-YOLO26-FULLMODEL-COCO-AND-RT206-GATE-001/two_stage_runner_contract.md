# Two-stage runner contract

The Stage64 runner creates two independent ONNX Runtime sessions:

1. quantized inference graph, selected as CPU EP or SpacemiT EP;
2. floating-point post-processing graph, always CPU EP.

The inference graph has one `float32 1x3x640x640` input and exactly six
floating-point outputs. The tail accepts those six tensors by exact name and
returns `float32 1x300x6`.

The runner verifies:

- six-name and shape pairing;
- finite boundary and final values;
- explicit runtime/provider identities;
- separate session-create, first-run, inference, tail, and total timings;
- boundary and final hashes;
- one output hash per measured run;
- task-local profile, temp, cache, and output paths.

The board timing wrapper leaves ORT profiling disabled by default. Placement
arms opt in with `--enable-profiling`; timing arms omit it so fused-provider
and per-node CPU profile event costs are not silently mixed into the primary
steady-state comparison.

The COCO runner applies the accepted project letterbox, runs the same two
sessions, de-letterboxes final boxes, and writes timing and prediction files.
The intentionally CPU-only tail is reported separately and is not counted as
unexpected fallback.

Neither runner contains IME instructions. The provider is loaded from the
explicit immutable ORT 2.0.6 runtime root.

# Persistent Island Lifetime Plan

Prepared state owns four worker-local prepacked model5 weight sets, persistent MMT4D workspaces, a model4 postactivation buffer, corrected model5 int32 storage, two 256-entry LUTs, and 128 fixed-requant records.

The accounted persistent allocation/prepack footprint is `4,062,720` bytes for the four-worker configuration. Caller-owned input/output buffers are excluded.

Per-run order:

1. Borrow model4 NHWC uint8 preactivation output.
2. Fill persistent model4 postactivation NHWC signed buffer.
3. Run persistent CPU0-3 model5 Conv workers into persistent corrected int32 storage.
4. Apply prepared fixed requant and LUT into caller-owned NHWC model5 output.
5. Retain NHWC output for a future model6 consumer or convert once at island exit.

There is no custom heap allocation, graph-name lookup, weight packing, Python call, or file I/O in this run path.

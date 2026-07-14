# Unified resident arena

The Stage49 resident executor is retained as a kernel/schedule facade but binds directly to the FullExecutor activation arena and tensor offsets. It creates no second arena and no second worker pool in the selected safe route.

Headline execution binds the model4 entry and six model4-model9 live tensors by pointer. The old load_tensor, live-out copy, and unconditional diagnostic snapshots are absent. Capture mode may still allocate diagnostic snapshots outside headline timing.

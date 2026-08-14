# Selector contract

- Exact `tensor_names` are resolved against the post-fusion graph.
- A bounded selector requires both nonempty input and output names and a
  complete path between them.
- Strict missing matches are errors.
- A selected tensor with no quantization root, multiple roots, an FP32 root,
  or per-channel activation root is an error.
- Conflicting assignments to one final root are errors; equivalent overlap is
  deterministically deduplicated.
- Configuration order does not change root assignment.
- The manifest records policy name, final root, matched tensor names,
  observer, selected qparams, representable range, and lock state.

No private YOLO tensor name is embedded in XSlim source. Model-specific names
exist only in stage configuration/evidence.

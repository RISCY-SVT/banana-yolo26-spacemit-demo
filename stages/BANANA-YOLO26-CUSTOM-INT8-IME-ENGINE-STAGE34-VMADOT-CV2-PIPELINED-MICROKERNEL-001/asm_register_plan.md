# ASM Register Plan

Stage34 throughput diagnostic used named `smt.vmadot` only.

## Accepted Existing Wrapper

```text
inputs: v0, v1
accumulator: v28/v29 (m2)
sequence: vsetvli e32 -> zero accumulator -> vsetvli e8 -> load A/B -> smt.vmadot -> vsetvli e32 -> store
status: board pass through existing bench_vmadot_microkernel
```

## Rejected Diagnostic Shapes

```text
dependent: v28
2 accumulator low: v24/v26
2 accumulator high: v28/v30
4 accumulator: v20/v22/v24/v26
6 accumulator: v20/v22/v24/v26/v28/v30
```

All rejected diagnostic shapes assembled and disassembled symbolically, but trapped on board under CPU0 `taskset`.

Conclusion: do not use these accumulator plans for a selected `/model.4/cv2` microkernel without a separate proof stage.

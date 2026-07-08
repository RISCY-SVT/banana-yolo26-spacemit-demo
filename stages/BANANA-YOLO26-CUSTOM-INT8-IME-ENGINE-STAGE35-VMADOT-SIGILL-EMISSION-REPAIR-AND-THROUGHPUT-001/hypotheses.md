# Stage35 Hypotheses

stage_id: `BANANA-YOLO26-CUSTOM-INT8-IME-ENGINE-STAGE35-VMADOT-SIGILL-EMISSION-REPAIR-AND-THROUGHPUT-001`

H1: Stage34 failed because its benchmark emitted or executed `smt.vmadot` differently from the proven helper path.

H2: The failure is either named-asm encoding, missing/incorrect target feature, `vtype`/AVL/LMUL sequence, inline asm clobber/register allocation, or unsupported register shape.

H3: If the benchmark uses the exact same proven emission route and `vtype` sequence as the helper, `exact_single_wrapper_shape` must become board-executable on CPU0.

H4: Only after H3 passes may throughput and pipelined `/model.4/cv2/conv/Conv` microkernel results be considered meaningful.

H5: If the existing helper remains executable but a standalone same-word case traps, Stage35 must classify the root cause as an execution-context or register/vtype limitation, not as a hardware throughput ceiling.

H6: If the raw same-word path repairs the SIGILL and independent accumulator groups improve cycles per `smt.vmadot`, only then may Stage35 attempt one bounded `cv2` signed-storage `s8xs8` pipelined sidecar. No `vmadotus`, `vmadotn`, FP/vfmadot, graph expansion, or full-engine work is in scope.

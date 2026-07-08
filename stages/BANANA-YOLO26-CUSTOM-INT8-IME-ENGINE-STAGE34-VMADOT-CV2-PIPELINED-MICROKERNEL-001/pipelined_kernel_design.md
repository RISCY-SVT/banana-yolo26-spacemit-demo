# Pipelined Kernel Design

Stage34 design target was `/model.4/cv2/conv/Conv` only:

```text
1x1, 80x80, C_in=96, C_out=128
current storage: signed int8 activation storage
current instruction: smt.vmadot s8 x s8 -> s32
current correction: explicit u8-as-s8 correction after raw dot
```

Candidate family considered:

```text
A0_current_baseline
A1_unrolled_address_arithmetic_reduced
A2_pipeline_2_accumulator_tiles
A3_pipeline_4_accumulator_tiles
A4_pipeline_4_accumulators_prefetch_distance_2
A5_pipeline_4_accumulators_prefetch_distance_3
```

Step 0 throughput gate rejected implementation of A1-A5 in Stage34 because the local inline/register-blocked `smt.vmadot` shapes trapped on board with `SIGILL`. The accepted wrapper path remains executable, but it is not a viable substrate for a new software-pipelined register-blocked kernel without a separate assembler/semantics proof for the exact loop shape.

No `/model.4/cv2` runtime mode was changed. The selected runner remains the Stage26 branch1/add LUT path with Stage25 threaded MMT4D policy.

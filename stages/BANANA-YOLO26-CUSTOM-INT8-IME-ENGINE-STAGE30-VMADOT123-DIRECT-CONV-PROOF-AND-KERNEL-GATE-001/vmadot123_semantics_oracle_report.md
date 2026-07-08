# vmadot123 Semantics Oracle Report

Oracle method:

1. Use named asm for `smt.vmadot1`, `smt.vmadot2`, and `smt.vmadot3`.
2. Load an 8x8 signed-int8 A window into a non-overlapping vector register group.
3. Load a 4x8 transposed signed-int8 B tile into a separate vector register.
4. Derive an independent scalar bilinear map with impulse inputs over all 64 A positions and 32 B positions.
5. Validate the derived scalar map against independent fixtures:
   - all-zero
   - single-one / impulse
   - ramp
   - alternating signs
   - edge values -128/-127/-1/0/1/127
   - random deterministic seed
   - accumulate random deterministic seed

Result:

| variant | derived map entries | CPU0 mismatches | CPU1 mismatches | CPU2 mismatches | CPU3 mismatches | semantics status |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `smt.vmadot1` | 128 | 0 | 0 | 0 | 0 | oracle-proven |
| `smt.vmadot2` | 128 | 0 | 0 | 0 | 0 | oracle-proven |
| `smt.vmadot3` | 128 | 0 | 0 | 0 | 0 | oracle-proven |

Recovered semantics summary:

- `smt.vmadot1` computes the 4x4x8 dot tile from A rows shifted by +1 relative to base `smt.vmadot`.
- `smt.vmadot2` computes the 4x4x8 dot tile from A rows shifted by +2.
- `smt.vmadot3` computes the 4x4x8 dot tile from A rows shifted by +3.
- This explains the initial failed 32-byte probe: it did not provide the extra shifted A rows and overlapped implicit A use with the B vector register.

Acceptance conclusion:

`vmadot1/2/3` are parser-visible, assembler-visible, disassembly-visible, board-executable on CPU0-3, and independently scalar-oracle proven in Stage30.

This proves semantics, not full Conv value. Direct-conv integration remains a separate gated step.

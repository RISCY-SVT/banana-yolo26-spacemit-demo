# Stage30 Final Report

classification: `stage30-vmadot123-semantics-proven-but-no-speed-win`
stage_id: `BANANA-YOLO26-CUSTOM-INT8-IME-ENGINE-STAGE30-VMADOT123-DIRECT-CONV-PROOF-AND-KERNEL-GATE-001`
repo: `/data/banana-yolo26-spacemit-demo`
branch: `yolo26-custom-int8-engine`
start_head: `8c3d64735c3460174e709c780b6a179724a807e8`
end_head: `921c1d75ab5161bb9e3e732516047cfe058e3b16`
pushed: false
full_engine_implemented: false
ncnn_source_mutated: false
production_claim_made: false

## Summary

Stage30 opened the first narrow `vmadot1/2/3` proof lane. It recovered prior W1 evidence, built current named-asm probes, verified symbolic disassembly, executed the probes on board CPU0-3, and built an independent scalar oracle from impulse-derived bilinear maps.

## Results

| instruction | parser/assembler | disassembly | board CPU0-3 | independent oracle | implementation status |
| --- | --- | --- | --- | --- | --- |
| `smt.vmadot1` | pass | pass | pass | pass | micro-semantics proven |
| `smt.vmadot2` | pass | pass | pass | pass | micro-semantics proven |
| `smt.vmadot3` | pass | pass | pass | pass | micro-semantics proven |
| `smt.vmadotn` | not used | not used | not used | not proven | rejected/not authorized |

The first 32-byte A-window probe executed but failed validation. That failure identified the real semantics: `vmadot1/2/3` require shifted A rows beyond the base 4x8 tile. The corrected proof uses an 8x8 A window and non-overlapping vector registers.

Derived oracle validation:

```text
CPU0: status_failures=0 traps=0 validation_mismatches=0
CPU1: status_failures=0 traps=0 validation_mismatches=0
CPU2: status_failures=0 traps=0 validation_mismatches=0
CPU3: status_failures=0 traps=0 validation_mismatches=0
```

## Direct Conv Decision

No real direct Conv sidecar was accepted in Stage30. The instructions are now semantics-proven, but a useful 3x3 Conv kernel requires an expanded-A-panel schedule and duplicate-row/output policy. That is a bounded Stage31 integration task.

## Validation

- Host build: pass.
- Host CTest: pass.
- RISC-V cross build with IME: pass.
- Board CPU0-3 micro-oracle: pass.
- No CPU4-7 IME execution.
- No full engine, graph scheduler, camera/full-image, COCO/mAP, or production claim.

## Next

Recommended next stage:

`BANANA-YOLO26-CUSTOM-INT8-IME-ENGINE-STAGE31-VMADOT123-DIRECT-CONV-INTEGRATION-001`

## Non-Claims

This is not full YOLO26 inference. This is not model FPS. This is not full-image/camera performance. This is not COCO/mAP. This is not production/default-backend readiness.

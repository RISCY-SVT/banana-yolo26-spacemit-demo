# STAGE31_FINAL_REPORT

classification: stage31-direct-conv-correct-but-no-speed-win
stage_id: BANANA-YOLO26-CUSTOM-INT8-IME-ENGINE-STAGE31-VMADOT123-DIRECT-CONV-INTEGRATION-001
repo: /data/banana-yolo26-spacemit-demo
branch: yolo26-custom-int8-engine
start_head: 921c1d75ab5161bb9e3e732516047cfe058e3b16
end_head: pending-local-commit-see-final-response
pushed: false
full_engine_implemented: false
ncnn_source_mutated: false
production_claim_made: false

## Summary

Stage31 implemented one explicit direct/sliding 3x3 Conv sidecar for:

`/model.4/m.0/cv1/conv/Conv`

The sidecar uses the Stage30-proven `smt.vmadot1`, `smt.vmadot2`, and `smt.vmadot3` shifted-A-row instructions, while keeping the accepted MMT4D runner path unchanged.

## Correctness

| Check | Status |
| --- | --- |
| Stage30 vmadot1/2/3 replay CPU0-3 | pass |
| Primary direct Conv CPU0-3 | pass |
| Host CTest | pass |
| RISC-V cross build | pass |
| CPU4-7 IME execution | not used |

Primary direct Conv result:

- mismatches: 0
- max_abs_diff: 0
- checksum_direct: 1324192976
- checksum_expected: 1324192976

## Performance

Protocol:

- `taskset -c 0-3`
- warmup: 10
- runs: 100
- repeats: 5

| Candidate | mean_us | stddev_us | Result |
| --- | ---: | ---: | --- |
| Direct/sliding vmadot123 sidecar | 56980.9 | 11.5521 | correctness-only; speed fail |
| MMT4D 1-thread | 20544.9 | 82.4099 | faster |
| MMT4D 4-thread | 5437.09 | 29.5962 | current best |

Direct sidecar buckets:

| Bucket | mean_us |
| --- | ---: |
| panel_build | 38901.3 |
| kernel_compute | 15795.9 |
| correction | 201.322 |
| writeback | 1275.35 |

Speed gate:

- same-thread gate: fail (`0.360558x`, required >= `1.20x`)
- best-threaded gate: fail (`0.0954194x`, required >= `1.15x`)

## vmadotn

`vmadotn` remains rejected/not authorized. Candidate mnemonics `smt.vmadotn`, `smt.vmadot.n`, and `smt.vmadot4` were rejected by the assembler.

## Signedness Family

Unsigned and mixed signedness dot variants are parser/disassembly-visible, but Stage31 did not prove their independent semantics and did not implement them.

## Non-Claims

This is not full YOLO26 inference.
This is not model FPS.
This is not full-image/camera performance.
This is not COCO/mAP.
This is not production/default-backend readiness.

## Final-head Traceability

The repo-local report cannot contain its own final commit hash before the commit is made. The result packet will include a `.with-final-head.md` copy after commit.

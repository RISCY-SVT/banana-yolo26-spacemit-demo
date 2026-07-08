# vmadot123 Evidence Recovery

Sources inspected:

- `/exchange/results/archive/W1-IME-INT8-VMADOT-TOOLCHAIN-HARDWARE-PROOF-001/`
- `/data/lab/task-runs/W1-IME-INT8-VMADOT-TOOLCHAIN-HARDWARE-PROOF-001/`
- `/control/specs/drafts/` references found by local search
- `/data/ncnn-logs/ai-team/` stage reports
- `/data/ncnn-logs/ort-logs/` Track B context

Recovered prior facts:

| instruction | spec-visible | assembler-visible | disassembly-visible | board-executable | oracle-proven before Stage30 | implementation-authorized before Stage30 |
| --- | --- | --- | --- | --- | --- | --- |
| `smt.vmadot1` | yes | prior route reported accepted | prior route reported accepted | prior route reported accepted | no | no |
| `smt.vmadot2` | yes | prior route reported accepted | prior route reported accepted | prior route reported accepted | no | no |
| `smt.vmadot3` | yes | prior route reported accepted | prior route reported accepted | prior route reported accepted | no | no |
| `smt.vmadotn` | no accepted current route | rejected on tested routes | rejected on tested routes | rejected on tested routes | no | no |

Stage30 interpretation:

- Prior board-executable evidence is useful but not enough for engine use.
- Stage30 performs a current parser/disassembly/board/oracle regression.
- `vmadotn` remains rejected/not authorized and is not implemented.

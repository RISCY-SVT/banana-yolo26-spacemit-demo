# Recovery after host restart

## Recovered state

The Ubuntu execution host restarted while the Stage65C-R1 board hash smoke was being
introduced. On resume, no Stage65C-R1, ONNX Runtime, SSH runner, bootstrap, evaluator, or
dataset process remained on the host. The board retained boot ID
`0a0691d1-7502-44c3-903b-444dba83c1d9` and had no active Stage runner.

The only incomplete board surface was `hashing-smoke`: its status file contained a header and
no accepted rows. One B2 CPU inference had completed and produced valid frozen output/boundary
hashes, but the orchestration stopped before recording acceptance. No full-val directory or
other decision surface had been opened.

## Root cause and isolation

The first clean reproduction failed after inference because the board `awk` implementation
rejects `index` as a scalar loop variable. The failure was in status parsing, not model/runtime
execution. Both incomplete roots were renamed under the stage-owned NVMe root and copied to raw
evidence as `recovery-smoke/interrupted` and `recovery-smoke/awk-failure`; neither contributes to
an accepted metric.

The parser variable was renamed to `i`, the script was syntax-checked and redeployed, and a new
four-surface smoke passed with exact frozen output and six-boundary identities. A subsequent
2-case x 4-surface x 100-run matrix passed. It was cross-checked against the accepted 10 clean
session recreations per surface. Every surface had one output hash and one boundary manifest.

## Experimental disposition

- Incomplete smoke data: isolated, retained as raw recovery evidence, excluded from decisions.
- Accepted determinism data: generated in fresh, fail-on-existing directories after recovery.
- Board/runtime state: unchanged and deterministic.
- Full val2017: started only after all reconstructed determinism and tail-replay gates passed.

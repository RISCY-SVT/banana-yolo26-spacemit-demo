# Source hygiene report

The final prospective Stage54 change set passed `git diff --check`; the staged
evidence set passed `git diff --cached --check` after normalization.

```text
prospective changed paths:       152
stage symlinks:                  0
changed symlinks:                0
changed files larger than 5 MiB: 0
vendor/model/dataset artifacts:  0
secret-like matches:             0
private-path matches:            0
tracked repository symlinks:     0
```

Model packages, executables, libraries, fixtures, COCO predictions, build
trees, and raw board logs remain outside Git under the recorded `/data`
evidence roots.

The largest new report is `same_shape_route_performance_raw.tsv` at 5,028,107
bytes. It contains 58,228 structured full-operation timing observations from
the mandatory controlled same-shape experiment and remains below the 5 MiB
large-file gate.

`/data/ncnn` was not modified. Its final state matches the pre-existing
Stage54 entry state:

```text
HEAD:
  a245a70c641a1f20f357c65d103e5f9e50fe84a1

pre-existing modified files:
  src/layer/riscv/convolution_1x1_int8_xsmtvdot.S
  src/layer/riscv/convolution_1x1_int8_xsmtvdot.cpp
  src/layer/riscv/convolution_1x1_int8_xsmtvdot.h

binary diff SHA-256:
  2bf1cc38885018a02478aa7542581639786c79bca5ce11a6e827d24bcc5f4eca
```

The raw changed-path inventory and scan outputs are under the Stage54 log
root's `artifacts/` directory. The complete command and exit-code record is in
`command-ledger.tsv`.

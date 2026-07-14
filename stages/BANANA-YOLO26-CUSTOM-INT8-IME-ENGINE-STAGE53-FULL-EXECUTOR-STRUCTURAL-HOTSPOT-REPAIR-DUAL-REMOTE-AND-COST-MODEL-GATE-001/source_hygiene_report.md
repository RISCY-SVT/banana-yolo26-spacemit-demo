# Source hygiene report

The final prospective Stage53 change set passed `git diff --check` and the
staged evidence set passed `git diff --cached --check` after normalization.

```text
prospective changed paths:       89
stage symlinks:                  0
changed symlinks:                0
changed files larger than 5 MiB: 0
vendor/model/dataset artifacts:  0
secret-like matches:             0
private-path matches:            0
tracked repository symlinks:     0
```

The largest new report is `full_operation_profile_raw.tsv` at 826803 bytes.
It is a structured 10,400-observation timing table, not a binary dump. Model
packages, executables, libraries, fixtures, COCO predictions, build trees, and
raw board logs remain outside Git under the recorded `/data` evidence roots.

`/data/ncnn` was not modified. Its final state matches the pre-existing Stage53
entry state:

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

The raw changed-path inventory and scan outputs are under the Stage53 log
root's `artifacts/` directory. The complete command and exit-code record is in
`command-ledger.tsv`.

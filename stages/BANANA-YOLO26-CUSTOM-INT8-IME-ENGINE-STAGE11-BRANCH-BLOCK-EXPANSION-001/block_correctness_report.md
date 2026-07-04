# Block Correctness Report

## Host

Host-native build: pass.

CTest:

- total: `28`
- passed: `28`
- failed: `0`
- new test: `test_stage11_branch_block_runner`

## Board

Board target: `svt@banana`

CPU affinity:

- CPU0: pass
- CPU1: pass
- CPU2: pass
- CPU3: pass

For each CPU:

- Stage 10 replay: pass
- Stage 10 RNE regression: pass
- Stage 11 scalar A0: pass
- Stage 11 scalar A2: pass
- Stage 11 IME A2: pass
- branch cv1 activation mismatches: `0`
- branch cv2 corrected-int32 mismatches: `0`

No IME test was run on CPU4-7.

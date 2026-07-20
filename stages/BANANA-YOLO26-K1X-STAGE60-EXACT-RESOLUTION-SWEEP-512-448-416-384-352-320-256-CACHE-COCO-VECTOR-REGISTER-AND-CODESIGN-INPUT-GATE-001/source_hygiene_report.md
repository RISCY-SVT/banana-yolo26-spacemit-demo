# Source Hygiene Report

## Scope

The final scan covers every file changed from frozen source commit
`175c1d939cc93fba0e730dba3f1281704e8f25b9` and every Stage60 report file.
Large generated packages, predictions, benchmark samples, traces, and COCO
artifacts remain under the task-local NVMe `/data` roots and are not committed.

## Results

| Check | Result |
| --- | --- |
| `git diff --check` | pass |
| `git diff --cached --check` | pass |
| unresolved placeholder scan | 0 |
| report symlink scan | 0 |
| report files larger than 10 MiB | 0 |
| committed model/binary artifact scan | 0 |
| credential/private-key signature scan | 0 |
| non-exportable private-path scan | 0 |
| cross-binary RPATH/RUNPATH audit | none |
| board eMMC project writes | 0 |

The three pre-existing modified files in `/data/ncnn` were not touched by
Stage60. Their final SHA-256 values equal the hashes captured before Stage60:

| File | SHA-256 |
| --- | --- |
| `convolution_1x1_int8_xsmtvdot.S` | `b50bd3355fea15adc142d7ae70e8916ef78781563d1aeccdf500d014f91c5229` |
| `convolution_1x1_int8_xsmtvdot.cpp` | `f36b8d1a40ac905ba744edf9c575b64b0ff109a15cf8418af98ca115514acdb6` |
| `convolution_1x1_int8_xsmtvdot.h` | `fe136f6470d52d3ad1cf580a8a4d53393c669b1191a19ae3600ff6df118ce359` |

## Validation Summary

- Host CTest: 50/50 pass.
- ASan/UBSan CTest: 51/51 pass.
- TSan: full suite pass, Stage48 pass, and 20 repeated startup runs pass with
  no race report.
- Python source compilation: pass.
- RISC-V release and Stage60 tool cross-builds: pass.
- Board loader, capability checks, all exact fixtures, full COCO, finalist
  soaks, and system-profile rollback: pass.

One earlier hygiene invocation had shell-quoting and path errors. It produced
no result and is retained in the command ledger. The corrected
`final-source-hygiene-scans-v3` invocation is the authority for this report.

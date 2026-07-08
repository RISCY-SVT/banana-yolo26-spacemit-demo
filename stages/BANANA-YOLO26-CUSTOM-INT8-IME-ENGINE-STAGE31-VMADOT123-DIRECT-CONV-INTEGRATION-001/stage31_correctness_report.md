# Stage31 Correctness Report

Primary node:

`/model.4/m.0/cv1/conv/Conv`

Output boundary:

Corrected int32 NHWC output, compared against existing accepted full-shape branch-entry reference.

Board correctness:

| CPU | direct_status | direct_mismatches | direct_max_abs_diff | checksum_direct | checksum_expected |
| --- | ---: | ---: | ---: | ---: | ---: |
| CPU0 | 0 | 0 | 0 | 1324192976 | 1324192976 |
| CPU1 | 0 | 0 | 0 | 1324192976 | 1324192976 |
| CPU2 | 0 | 0 | 0 | 1324192976 | 1324192976 |
| CPU3 | 0 | 0 | 0 | 1324192976 | 1324192976 |

Host test:

`test_stage31_vmadot123_direct_conv` passed as part of `41/41` host CTest.

Conclusion:

The primary direct/sliding sidecar is exact for the corrected int32 node boundary on CPU0-3.

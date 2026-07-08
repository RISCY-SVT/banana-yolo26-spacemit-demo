# MMT4D Mixed Signedness Oracle Report

host_test: `test_stage33_mixed_signedness_oracle`

The host oracle compares the existing baseline formula against the mixed signedness formula on the real compact model4 C2f fixture after the runner has produced the concat input for `/model.4/cv2/conv/Conv`.

```text
baseline: smt.vmadot-style signed storage formula
candidate: u8 activation code * s8 weight + adjusted bias
```

Host output:

```text
stage33_mixed_signedness_oracle fixture=synthetic_seeded status=0 activation_zero_point_u8=15 input_storage_zero_point_s8=-113 output_elements=128 mismatches=0 max_abs_diff=0 checksum=-143848
```

Board CPU0-3 smoke:

```text
CPU0: mismatches=0 max_abs_diff=0
CPU1: mismatches=0 max_abs_diff=0
CPU2: mismatches=0 max_abs_diff=0
CPU3: mismatches=0 max_abs_diff=0
```

Gate B result: `pass`

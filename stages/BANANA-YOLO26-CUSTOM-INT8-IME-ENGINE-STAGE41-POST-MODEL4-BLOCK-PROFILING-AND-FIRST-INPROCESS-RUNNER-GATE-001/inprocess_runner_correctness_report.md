# In-Process Runner Correctness Report

## Host C++ Exact Scaffold

Host C++ in-process scaffold using host ORT 1.27.0 and scalar custom `/model.4`:

```text
full_ort_vs_expected_output0: mismatches=0 max_abs_diff=0 byte_equal=1
custom_model4_vs_ort_model4: mismatches=0 max_abs_diff=0 byte_equal=1
custom_model4_through_suffix_vs_full_ort_output0: mismatches=0 max_abs_diff=0 byte_equal=1
custom_model4_nhwc_sha256: 517db620fca8465888ec387673f888d5e7c43c86d613c88cbf4bb5ffcbe4cd91
inprocess_output0_bin_sha256: 8ddc0e17ab7307ac7fc1f91d9145acf3f88647d7528e73183b8e6d723c41ebac
```

This proves the C++ in-memory scaffold wiring can close against the accepted Stage40 ORT oracle without Python or per-block file handoff in the measured path.

## Board Selected-Mode Blocker

Board C++ in-process selected mode with SpacemiT ORT 2.0.1:

```text
full_board_ort_vs_host_expected_output0: mismatches=1597 max_abs_diff=635.707 byte_equal=0
custom_model4_vs_board_ort_model4: mismatches=78351 max_abs_diff=2 byte_equal=0
custom_model4_through_board_suffix_vs_board_full_ort_output0: mismatches=1508 max_abs_diff=635.707155 byte_equal=0
affinity_ok=1
```

Upstream board ORT smoke did not fix the contract:

```text
full_board_ort_vs_host_expected_output0: mismatches=1660 max_abs_diff=637.236
custom_model4_vs_board_ort_model4: mismatches=78311 max_abs_diff=2
```

Conclusion: board selected-mode hard gate is blocked by board ORT CPU reference/runtime contract mismatch. It is not accepted as a full output0 closure.

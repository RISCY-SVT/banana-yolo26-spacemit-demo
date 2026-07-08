# Same-Input ONNX Cut Report

The baseline and Stage33 candidate were both run through the real `y26_stage16_model4_c2f_run_cut_u8_output` runner API with the same Stage22 ONNX-cut fixture input.

fixture_dir: `.deps/custom_int8_engine/stage22_onnx_cut/model4_c2f_synthetic_seeded`

## Candidate Correctness

```text
mode: ime_threaded
merge_repair: branch1_add_lut_mixed_cv2
status: 0
mismatches: 0
max_abs_diff: 0
checksum: 106597930
expected_checksum: 106597930
affinity_ok: 1
output_sha256: 70adcf083e85ea95d5fb42e57ea26f51255944aaa442cde4d919f5ae551b3433
```

FRM sweep:

```text
RNE: mismatches=0 after_frm=0
RTZ: mismatches=0 after_frm=1
RDN: mismatches=0 after_frm=2
RUP: mismatches=0 after_frm=3
RMM: mismatches=0 after_frm=4
```

Gate result: `pass`

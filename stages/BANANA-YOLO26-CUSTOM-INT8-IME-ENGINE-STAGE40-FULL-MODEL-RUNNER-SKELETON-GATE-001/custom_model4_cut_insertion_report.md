# Custom Model4 Cut Insertion Report

Board command used the real selected `/model.4` runner path:

```text
taskset -c 0-3 ./bench_stage23_model4_runner_cut \
  --fixture-dir fixture \
  --mode ime_threaded \
  --output-quantize rvv_direct \
  --merge-repair branch3x3_fastpack \
  --thread-branch0 4 \
  --thread-branch1 4 \
  --thread-model4-cv2 4 \
  --warmup 10 \
  --runs 100 \
  --repeats 5 \
  --frm-sweep \
  --dump-actual custom_model4_cv2_q_u8_nhwc.bin
```

Result:

```text
status: 0
mismatches: 0
max_abs_diff: 0
checksum: 106624945
expected_checksum: 106624945
affinity_ok: 1
FRM sweep: pass RNE/RTZ/RDN/RUP/RMM
custom output SHA256: 517db620fca8465888ec387673f888d5e7c43c86d613c88cbf4bb5ffcbe4cd91
```

The custom output was then fed into the ORT CPU suffix cut. Final `output0` matched the full ORT CPU reference exactly:

```text
custom_model4_skeleton_final_vs_full_reference: pass
mismatches: 0
max_abs_diff: 0
```

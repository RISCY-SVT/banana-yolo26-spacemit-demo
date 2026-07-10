# In-Process Runner API

New tool:

```text
custom_int8_engine/tools/bench_stage41_inprocess_runner.cpp
```

Build is conditional on explicit CMake variables:

```text
Y26_K1X_ORT_INCLUDE_DIR
Y26_K1X_ORT_LIBRARY
```

Important CLI options:

```text
--model <full ONNX>
--cut-dir <prefix/model4/suffix cuts>
--input-npy <images.npy>
--expected-output-npy <output0.npy>
--custom-mode scalar|ime_threaded
--profile-cuts-tsv <suffix cumulative cuts>
--warmup <n>
--runs <n>
--repeats <n>
```

Host exactness was checked with:

```text
custom_mode: scalar
ORT library: host onnxruntime wheel 1.27.0
```

Board selected-mode was checked with:

```text
custom_mode: ime_threaded
ORT library: /home/svt/spacemit-ort.riscv64.2.0.1/lib
affinity: taskset -c 0-3
```

The tool keeps the existing custom `/model.4` runner mode explicit and does not alter default dispatch.

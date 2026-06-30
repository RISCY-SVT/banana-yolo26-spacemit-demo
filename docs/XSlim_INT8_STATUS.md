# XSlim INT8 Status

Raw evidence:

```text
/data/ncnn-logs/ort-logs/2026-06-30_08-45-33/
/data/ncnn-logs/ort-logs/2026-06-30_09-38-36/
```

## Current Decision

```text
XSLIM_STATIC_YOLO26_INT8_NEEDS_UPSTREAM_FIX
```

XSlim remains useful for diagnostics, but it did not produce an accepted
YOLO26n static INT8 path for rt204.

## What Was Tested

| Path | Contract | Result |
| --- | --- | --- |
| Static PTQ e2e/default | `[1,300,6]` | Fails in XSlim/PPQ `ReduceMax` executor before output model generation. |
| Static PTQ traditional/default observer | `[1,84,8400]` | Bounded runs enter long `Runtime Calibration(BlockWise)` and do not finish. |
| Static PTQ traditional `minmax` | `[1,84,8400]` | Emits ONNX, but CPU oracle is bad: class scores are all zero. |
| Static PTQ traditional `percentile` | `[1,84,8400]` | Emits ONNX, but CPU oracle is bad: class scores are all zero. |
| Dynamic quantization `--dynq` | e2e/traditional | CPU-good and rt204-runnable diagnostic only; not static activation INT8. |

## ReduceMax Upstream Blocker

The e2e path fails at:

```text
/model.23/ReduceMax(Type: ReduceMax, Num of Input: 2, Num of Output: 1)
ValueError: too many values to unpack (expected 1)
```

The same failure reproduces on tiny ONNX models containing `ReduceMax` after
XSlim's internal opset conversion. This is not specific to YOLO26 postprocess
complexity.

Config-level attempts did not fix the e2e path:

- `ignore_op_types=["ReduceMax"]`;
- `ignore_op_names=["/model.23/ReduceMax"]`;
- `skip_onnxsim=true`;
- `opset=18`;
- `calibration_type=minmax`.

Truncating before the top-k tail avoids the immediate crash, but enters the
same bounded long calibration behavior and does not preserve the e2e
`[1,300,6]` model contract.

## Traditional Static Output

Changing `calibration_type` from `default` to `minmax` or `percentile` allowed
the traditional path to finish in the bounded run:

| Candidate | SHA256 | Graph summary | CPU oracle |
| --- | --- | --- | --- |
| `traditional_minmax_static` | `d497215700ffa65e296731cc4e9cab2d624a4378ce344dece6dcae54bb367f04` | 365 `QuantizeLinear`, 473 `DequantizeLinear`, 102 `Conv`, no `QLinearConv` | fail: all class scores zero |
| `traditional_percentile_static` | `0fb95ea9811d9a4d449c4e114b80ab77541a6d1c58cd18d6fe511dd22822b9c9` | 365 `QuantizeLinear`, 473 `DequantizeLinear`, 102 `Conv`, no `QLinearConv` | fail: all class scores zero |

The minmax model was also run as a diagnostic on rt204. It executed without the
previous `clip minmax` Q/DQ Conv compile error, but it remained semantically
unusable because CPU and rt204 both produced zero class scores.

## Deployment Status

There is no CPU-good and rt204-EP-good XSlim static INT8 candidate.

Do not treat XSlim dynamic quantization as an accelerated static INT8 path. Its
graphs contain ordinary `Conv` nodes with dequantized weights and no proven
`QLinearConv`/`QLinearMatMul` offload.

## Next Step

Send the XSlim ReduceMax minimal repro and traditional zero-score report to
SpacemiT/XSlim maintainers, and rerun static PTQ only after an upstream fix or a
vendor-recommended YOLO26 configuration is available.

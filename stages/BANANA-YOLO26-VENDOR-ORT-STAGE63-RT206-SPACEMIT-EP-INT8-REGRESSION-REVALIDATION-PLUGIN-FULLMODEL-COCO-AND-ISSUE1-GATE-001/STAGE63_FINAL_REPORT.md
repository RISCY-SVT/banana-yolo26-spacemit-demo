# Stage63 final report

## Classification

`stage63-rt206-plugin-fixed-int8-ep-regressions-persist`

This is vendor-runtime research evidence only. It does not promote 2.0.6,
replace the custom executor, or change a protected release.

## Official asset

The official Linux riscv64 asset is
`spacemit-ort.riscv64.2.0.6.tar.gz`, 15,002,263 bytes, SHA-256:

```text
bebcdfb7df6b49eefa3863afcd85a3da2aa83c3ae9252d7d856188c38a70b0e6
```

The archive passed traversal, link, device-node, ownership, and mode checks.
The public 2.0.5 and 2.0.6 source tags both resolve to
`61e7fc2319cd16aa5487fd1155dc15c5390c8a90`; binary assets are therefore the
authority for this comparison.

`libonnxruntime.so.1.24.2+spacemit.a1` is byte-identical to 2.0.5, retains
embedded commit `9bb02204b`, and has no GNU build ID. The provider changed from
SHA-256 `3927b51f...a41d` to `dcc95030...74ae`, grew by 258,304 bytes, and
advertises package version 2.0.6. It also has no GNU build ID. Package inventory
contains 61 identical paths, six changed paths, 24 additions, and two removals.

## Issue #1

| Item | 2.0.6 result | Classification |
|---|---|---|
| Q/DQ Conv without `kernel_shape` | assigned; exact | unchanged positive control |
| Q/DQ Conv with `kernel_shape=[3,3]` | clip-minmax error, then abort | unchanged |
| minimal QLinearConv | SIGILL, exit 132 | unchanged |
| minimal QLinearMatMul | SIGILL, exit 132 | unchanged |
| official plugin sample link/load | build, `ldd -r`, `dlopen`, ABI/init pass | fixed |
| independent plugin execution | exact, 1,011 dispatches | fixed |
| full YOLO26 Q/DQ INT8 EP | provider compilation abort | unchanged |

The QLinear fault packet records PC `0x3ff7963300` and instruction word
`0xe204082b` in the unchanged ORT core. The accepted objdump renders it as
`.insn`; no instruction meaning is inferred. The positive Q/DQ control executes
exactly on CPU0, CPU0-3, CPU4, CPU4-7, and CPU0-7. Broader crashing
QLinear affinity arms were not repeated after the CPU0 gate failed.

## Plugin result

2.0.6 exports the public plugin methods that were unresolved in 2.0.5. The
official sample and an independent plugin build, resolve, load, query ABI,
initialize, register, and execute. The independent uint8 XOR operator is exact
for 1,000 measured runs with a 90.238765 us mean.

The packaged official Track2 graph is not numerically valid against its CPU
control: 31,892 of 32,768 values differ, cosine similarity is 0.271277368069,
and maximum absolute difference is 6.549999713898. This is recorded as a
newly exposed sample correctness defect, not hidden by the ABI fix.

## Full YOLO26

The exact INT8 model SHA-256 is
`30a94e4738606673b5e0a73499cbc977167f046f8fa8637d6040ce744f429c0c`.
CPU FP32, CPU FP16, CPU INT8, EP FP32, and EP FP16 create sessions and produce
finite `1x300x6` outputs. EP INT8 aborts during compilation at the first
quantized Conv, under both `ORT_DISABLE_ALL` and `ORT_ENABLE_ALL`.

With `ORT_ENABLE_ALL`, transformed FP32 and FP16 EP graphs contain all 102 Conv
and four MatMul operations. EP profile-time shares are 99.3353853% and
99.1816852%; residual CPU events are primarily shape, gather, TopK, and output
housekeeping. Because the graph is transformed, source-node residual estimates
of 16.11% and 20.00% are approximations, not exact assignment fractions.
Executed INT8 placement is zero/unknown because compilation aborts.

The vendor provider overrides parts of the outer CPU0-3 affinity policy. A
bounded probe observed CPU INT8 entirely on allowed list 0-3, while EP FP32
created fixed CPU0/1/2/3 threads and a thread allowed on CPU4-7. This is
reported as actual vendor behavior, not as a CPU0-3-only EP run.

## Correctness and COCO

All runnable fixed fixtures (F0, bus, Zidane, and canonical) are finite and
structurally valid. Tiny operators use independent exact oracles. Cross-runtime
final detection tensors are task-level comparisons and are not required to be
byte-identical.

All four runnable Stage63 COCO surfaces completed 5,000/5,000:

| Surface | mAP50-95 | Prediction SHA-256 |
|---|---:|---|
| 2.0.6 CPU FP32 | 0.404730651123 | `e8c97ebf...96c1a` |
| 2.0.6 CPU INT8 | 0.374594101159 | `1dbc1183...d0b48` |
| 2.0.6 EP FP32 | 0.404745220420 | `cae114cf...80f7d` |
| 2.0.6 EP FP16 | 0.404686635006 | `0add992f...bfe30` |

CPU INT8 matches all 5,000 accepted 2.0.5 per-image output and detection
hashes. The changed EP differs from 2.0.4 at tensor level on all FP32 images
and 4,996 FP16 images while retaining effectively equal aggregate COCO
accuracy. EP INT8 is non-runnable and has no COCO row.

## Performance and stability

Matched five-block, 500-inference means:

| Surface | Mean us | Delta versus 2.0.5 |
|---|---:|---:|
| 2.0.6 CPU INT8 | 1,022,360.082 | +0.251% |
| 2.0.6 EP FP32 | 436,560.705 | +0.156% |
| 2.0.6 EP FP16 | 357,382.003 | +0.107% |

The matched B120 CPU INT8 control is 460,031.116 us; 2.0.6 CPU INT8 is
2.222x slower. FP16 EP is the fastest runnable vendor surface, but no INT8 EP
route exists to compare or promote.

All five runnable 2.0.6 routes passed 1,000-run stability with stable output
hashes and 1.6 GHz frequency. EP FP16 showed a bimodal tail
(mean 355,507.596 us, p95 407,284.499 us, CV 8.85%); this is retained rather
than collapsed into the five-block headline. A 10,000-run INT8 EP soak was not
run because that route cannot create a session.

## Closure

The issue update is a draft only; Stage63 made no GitHub issue mutation.
Protected main, tags, custom executor source, release archives, and
`/data/ncnn` are unchanged. One 432,902-byte provider-dump spill to eMMC was
fully hashed, moved to NVMe, removed, and prevented in the hardened launcher.

Decision: **2.0.6 deserves no runtime promotion. Human review may send the
sanitized issue draft and repro bundle to the vendor.**

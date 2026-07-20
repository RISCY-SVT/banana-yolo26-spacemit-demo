# Frozen R640 Reproduction

The frozen 0.9.2 control was rebuilt before any resolution-generalization
change was selected.

## Identity

| Item | Expected | Reproduced |
| --- | --- | --- |
| Source model SHA-256 | `30a94e4738606673b5e0a73499cbc977167f046f8fa8637d6040ce744f429c0c` | exact |
| Package manifest SHA-256 | `fab4a72cf524ce0a205ceca0384144f2eee7bc79dff3f4db8b7208614e8407be` | exact |
| Fixed output hash | `0xd43f5e018b415631` | exact |
| COCO prediction SHA-256 | `cda5c8c7a46d61d9c90f6292001eea190cb8f6617efe647a33dc6134dd57ccda` | exact |

Two clean package generations produced byte-identical trees. The rebuilt R640
graph/package also remained byte-identical after the Stage60 static-shape
generalization.

## Performance

The pre-change O2 control used warmup 10, 100 runs, and five repeats:

| Samples | Mean us | Median us | P95 us | P99 us | Max us |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 500 | 133224 | 133215 | 133699 | 133998 | 136218 |

The mean differs by -0.118% from the required 133.382 ms reference and passes
the 1% reproduction gate. Every sample reported affinity success, the frozen
output hash, and zero IME operations on CPU4-7. The O2 profile restored after
the run.

The final Stage60 500-sample R640 control measured 133012.150460 us mean,
133540.565550 us p95, and 133903.708940 us p99. It remains the same arithmetic
and package identity; this later row is the common Stage60 comparison surface.

## Correctness And Accuracy

- all 215 integer boundaries: exact scalar versus selected board route;
- F0-F7, bus/canonical, and Zidane: exact;
- FRM and vector CSR restoration: pass;
- deterministic repeated output: pass;
- CPU4-7 IME count: zero;
- COCO: 5000/5000, zero failures;
- mAP50-95: `0.3707408944391919`;
- mAP50: `0.5258465300872381`.

The accepted 640 baseline is therefore reproduced and is the control for all
Stage60 Q0 resolution comparisons.

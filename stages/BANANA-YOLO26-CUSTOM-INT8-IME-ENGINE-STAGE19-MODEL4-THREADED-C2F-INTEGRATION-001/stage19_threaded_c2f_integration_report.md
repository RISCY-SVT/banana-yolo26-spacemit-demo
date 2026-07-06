# Stage19 Threaded C2f Integration Report

Selected subset:

```text
candidate_K_model4_threaded_c2f_compact_oracle_scope
shape_class: compact_oracle_scope
```

Implementation:

```text
Stage18 threaded Conv sidecar was wired through the Stage15 model4 branch runner.
Stage16 model4 C2f runner exposes an explicit threaded branch0 mode.
Default scalar and IME hotpaths remain unchanged.
No graph-wide scheduler or default backend switch was added.
```

Board correctness:

```text
thread_count: 1, 2, 3, 4
activation sidecar: tested with 4 threads
branch1 mismatches: 0
concat mismatches: 0
model4_cv2 mismatches: 0
checksum: -143848
worker affinity: ok
```

Stable compact timing:

| candidate | threads | activation threaded | mean_total_us | stddev_total_us | CV % | mean_conv_us | branch0_conv_us | thread_overhead_us | total speedup vs A0 | branch0 conv speedup vs A0 | mismatches |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| A0_single_thread_c2f | 1 | 0 | 186.079392 | 0.917326 | 0.492976 | 122.624718 | 7.192042 | 0.000000 | 1.000000 | 1.000000 | 0 |
| A1_threaded_conv_1t | 1 | 0 | 210.511778 | 0.795247 | 0.377768 | 145.259406 | 28.966632 | 21.024598 | 0.883938 | 0.248287 | 0 |
| A2_threaded_conv_2t | 2 | 0 | 213.024168 | 0.747109 | 0.350716 | 147.414156 | 31.768988 | 23.272288 | 0.873513 | 0.226386 | 0 |
| A3_threaded_conv_3t | 3 | 0 | 215.200682 | 0.860686 | 0.399946 | 149.009836 | 33.091648 | 24.761346 | 0.864678 | 0.217337 | 0 |
| A4_threaded_conv_4t | 4 | 0 | 283.534780 | 1.182918 | 0.417204 | 212.156010 | 84.853328 | 76.068746 | 0.656284 | 0.084759 | 0 |
| A5_threaded_conv_threaded_activation_4t | 4 | 1 | 461.796518 | 172.024673 | 37.251185 | 204.429350 | 82.424028 | 258.251034 | 0.402947 | 0.087257 | 0 |

Conclusion:

```text
The integrated path is correct.
The compact fixture is too small for cluster0 threading to pay off.
A4 and A5 are not selected for compact C2f execution.
Representative/full-shape timing remains required before selecting a threaded model4 C2f mode.
```

# Threaded Activation Sidecar Report

Sidecar mode:

```text
A5_threaded_conv_threaded_activation_4t
thread pool: same persistent CPU0-3 worker pool as threaded Conv
activation math: unchanged A2 RVV f32 LUT boundary implementation
scope: compact Stage19 C2f oracle fixture only
```

Correctness:

```text
branch1 mismatches: 0
concat mismatches: 0
model4_cv2 mismatches: 0
checksum: -143848
affinity_ok: 1
CPU4-7 IME execution: none
```

Timing:

```text
A0 activation_requant_us: 27.298578
A4 activation_requant_us: 32.392534
A5 activation_requant_us: 216.748804
A5 total_us: 461.796518
A5 total speedup vs A0: 0.402947x
A5 CV: 37.251185%
```

Decision:

```text
threaded_activation_sidecar_status: fail_for_compact_fixture
```

The sidecar is exact but slower on compact tensors because worker/barrier overhead exceeds useful activation work. It should not be selected for compact C2f. A future representative/full-shape activation/fusion stage may still be useful because Stage18 representative A4 has activation share 44.971379%.

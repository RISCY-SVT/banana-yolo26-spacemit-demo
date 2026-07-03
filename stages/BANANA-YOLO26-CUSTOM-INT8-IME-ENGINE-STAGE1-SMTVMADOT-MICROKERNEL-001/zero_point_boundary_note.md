# Zero Point Boundary Note

The Stage 1 microkernel accepts signed int8 A and signed int8 B panels and accumulates into signed int32 C.

It deliberately does not implement:

- activation zero-point subtraction
- weight zero-point subtraction
- asymmetric correction
- requantization
- output zero-point addition
- clamp to int8/uint8
- graph tensor scale handling

For asymmetric Q/DQ tensors, later graph integration must handle:

```text
sum((a - za) * (w - zw))
= sum(a*w) - zw*sum(a) - za*sum(w) + K*za*zw
```

Stage 1 only proves the direct signed dot-product primitive:

```text
sum(int32(a_s8) * int32(b_s8))
```

This boundary is required because Stage 0 found asymmetric activation zero-points in real YOLO26 Q/DQ graph paths.

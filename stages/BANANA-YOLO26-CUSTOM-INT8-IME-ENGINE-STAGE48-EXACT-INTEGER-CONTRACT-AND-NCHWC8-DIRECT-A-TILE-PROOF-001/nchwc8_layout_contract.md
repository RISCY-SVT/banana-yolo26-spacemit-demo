# NCHWc8 direct-layout contract

Layout ID: `NCHWc8_SPATIAL_INNER_V1` (also described as NC8HW8).

```text
offset(n, cb, y, x, ci) = ((((n * C_blocks + cb) * H + y) * W + x) * 8 + ci)
cb = channel / 8
ci = channel % 8
```

For one channel block, four adjacent spatial locations are stored as:

```text
[x0 c0..c7][x1 c0..c7][x2 c0..c7][x3 c0..c7]
```

The Stage48 model5 proof uses `1x80x80x128`, 3x3, stride 2, pad 1,
and produces `1x40x40x128`. Input and output physical bytes are signed-storage
codes. Interior A groups copy four C8 chunks into vmadot order; borders use the
physical input-zero-point code. Only one bounded worker-local tile is used.
There is no full im2col tensor and no generic per-byte interior `pack_a` loop.

Input conversion is completed before warmup and excluded from internal Conv
timing. A future persistent-slice stage must measure layout residency before any
end-to-end claim.

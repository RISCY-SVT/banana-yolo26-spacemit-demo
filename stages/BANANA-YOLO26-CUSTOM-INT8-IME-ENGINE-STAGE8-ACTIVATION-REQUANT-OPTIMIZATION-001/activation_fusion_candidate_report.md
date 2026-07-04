# Activation Fusion Candidate Report

Candidate A4 was represented by `activation_mode=fused_lut_pack`.

Observed CPU0 result:

| mode | total us | activation us | mismatches |
|---|---:|---:|---:|
| int8_lut | 350092 | 192568 | 0 |
| fused_lut_pack | 347546 | 192589 | 0 |

Current implementation detail: A4 is the same fused requant + LUT + write-to-current-NHWC-layout path as A2. It does not fuse into the next Conv packA layout. A5 `fused_lut_pack_for_next_conv` was not attempted because it would touch Conv dataflow and should be isolated in a follow-up.

Selected mode remains `int8_lut`.

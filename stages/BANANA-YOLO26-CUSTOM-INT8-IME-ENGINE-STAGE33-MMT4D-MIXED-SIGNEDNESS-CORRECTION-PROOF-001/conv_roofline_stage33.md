# Conv Roofline Stage33

node: `/model.4/cv2/conv/Conv`
shape: `80x80x96 -> 80x80x128`
kernel: `1x1`
MAC_count: `78643200`

| mode | conv_us | effective_GMAC_s | rough_percent_of_2_TOPS |
|---|---:|---:|---:|
| baseline `smt.vmadot s8xs8` | 11852.7 | 6.635 | 0.332% |
| candidate `smt.vmadotus u8xs8` | 12862.2 | 6.114 | 0.306% |

Interpretation:

The candidate reduced explicit correction to zero but reduced effective node throughput because uint8 A packing plus fused copy/writeback overhead exceeded the removed correction pass.

This supports keeping the current MMT4D mainline and looking next at thread/copy overhead or output-quantize, not selecting mixed signedness for `/model.4/cv2`.

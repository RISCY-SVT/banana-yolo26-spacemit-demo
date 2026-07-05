# Conv/IME Roofline Stage17

Scope: selected representative/full-shape branch-entry only.

Measured Conv:

```text
node: /model.4/m.0/cv1/conv/Conv
shape: 1x80x80x32 -> 1x80x80x16
kernel: 3x3 stride1 pad1
MAC_count: 29491200
ime_single_thread_us: 20458.001284
effective_GMAC_s: 1.441548
effective_TOPS: 0.001442
percent_of_2TOPS: 0.072077
bottleneck_class: structural_low_K_or_packing
```

Interpretation: this is not full-model utilization. The selected Conv shape has low effective utilization in the current single-thread MMT4D path, but Stage17 threading shows the work can scale across cluster0 for this spatial partition.

`/model.4/cv1/conv/Conv` and `/model.4/m.0/cv2/conv/Conv` are recorded as metadata in this stage; their full-shape per-node timings were not measured and must not be compared to the measured branch-entry Conv.

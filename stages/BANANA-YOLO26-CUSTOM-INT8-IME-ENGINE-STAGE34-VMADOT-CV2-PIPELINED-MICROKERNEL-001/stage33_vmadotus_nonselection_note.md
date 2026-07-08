# Stage33 VMADOTUS Nonselection Note

Stage33 proved `smt.vmadotus u8 x s8` correctness for `/model.4/cv2/conv/Conv`, but the selected-cut timing regressed:

```text
baseline total_us: 40380.4
vmadotus total_us: 40934.1
baseline model4_cv2_conv_us: 11852.7
vmadotus model4_cv2_conv_us: 12862.2
baseline model4_cv2_compute_us: 8129.4
vmadotus model4_cv2_compute_us: 9699.05
baseline model4_cv2_correction_us: 1742.83
vmadotus model4_cv2_correction_us: 0
vmadotus model4_cv2_copy_us: 1127.18
```

Stage34 therefore keeps:

```text
s8 x s8 smt.vmadot + explicit correction
```

and does not select:

```text
u8 x s8 smt.vmadotus + reconstructed u8 activation
```

This remains selected-subset evidence only.

# In-Process Layout Conversion Report

The in-process scaffold currently needs two local layout adapters around the custom `/model.4` runner:

```text
ORT prefix output: NCHW uint8
custom /model.4 input: NHWC uint8
custom /model.4 output: NHWC uint8
ORT suffix input: NCHW uint8
```

Host exact scaffold timing:

```text
mean_layout_conversion_us: 1631.365500
mean_total_us: 301182.315667
layout_share: ~0.54%
```

Board selected-mode timing with SpacemiT ORT 2.0.1:

```text
mean_layout_conversion_us: 11022.462626
mean_total_us: 858404.224484
layout_share: ~1.28%
```

The adapter is not the Stage41 blocker. The blocker is ORT CPU reference contract mismatch on board.

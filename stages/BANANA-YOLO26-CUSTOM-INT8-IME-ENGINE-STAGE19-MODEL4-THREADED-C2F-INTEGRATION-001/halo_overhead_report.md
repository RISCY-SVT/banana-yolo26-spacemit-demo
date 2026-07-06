# Halo Overhead Report

Primary Stage18 partition:

```text
node: /model.4/m.0/cv1/conv/Conv
partition: spatial row split over output H
worker CPUs: CPU0-3
```

Stage19 compact C2f note:

```text
The compact oracle fixture has very small spatial extent.
Threaded workers may receive zero output rows.
Stage19 fixed zero-row worker handling so empty chunks no-op successfully.
```

Measured compact effect:

```text
A4 branch0_conv_us: 84.853328
A4 thread_overhead_us: 76.068746
A0 branch0_conv_us: 7.192042
```

Conclusion:

```text
For compact C2f, row-split overhead dominates useful work.
For representative/full-shape branch-entry, Stage18 replay still shows 3.455020x Conv speedup.
No new asymmetric halo optimization was selected in Stage19.
```

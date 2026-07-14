# Head decode contract

The selected head reads NCHWc8 blocks contiguously, performs one fused best-class scan, uses exact Q24 scores and the frozen strict-greater class tie rule, then applies the existing stable candidate ordering to produce the unchanged 1x300x6 output.

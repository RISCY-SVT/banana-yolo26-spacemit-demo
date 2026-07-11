# Model5 phase attribution

The Stage43 `model5_im2col_pack_us=0` value meant instrumentation was disabled. Stage44 enabled the existing tile-pack timer in diagnostic runs. The current path packs a reusable `4 x K` panel per output tile; it does not allocate a full 1.76-MiB im2col tensor.

R0 diagnostic pack mean was `4634.976 us`. R2a diagnostic pack mean was `4052.580 us`, a `1.143710x` pack speedup (`12.5652%` reduction). Model4 postactivation remained about `2.66 ms`, correction about `0.14 ms`, and model5 fixed requant plus SiLU about `5.13 ms`.

The per-tile clock instrumentation perturbs code layout/execution: separately launched diagnostic totals reverse the headline result. It is used only to identify the pack sub-bucket. Selection uses the instrumentation-off, in-process ABBA test: R0 `24636.0 us`, R2a `24157.4 us`, delta `-478.655 us` (`-1.94291%`).

Bias is already included in corrected int32 semantics; exact fixed requant and LUT remain a separate output pass. R2a does not implement R3 fusion.

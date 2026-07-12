
# Model4-model8 AOT slice

The 29-operation, 20-Conv resident-int8 slice is operational with one arena,
one worker pool, no internal transpose, and no ORT session. Board scalar/IME
outputs are stable and agree with the host custom integer route.

The fixed-host ORT no-tolerance gate fails. The first table mismatch is fixture
`F0` at `/model.6/Concat_output_0_QuantizeLinear_Output`; focused F2 forensics
localized an integer accumulator of -815 whose exact integer requant gives Conv
code 164 while host ORT's dequantized-float Conv gives 163. This is not U8S8 pair
saturation and not an IME/scalar disagreement. No tolerance was introduced.

Timing: internal `126826.026456 +/- 348.150158 us`; with one
entry and exit adapter `135761.369456 us`; B120 ORT intra4
`61239.054678 +/- 149.827643 us`. The custom
slice is `121.690831%` slower with adapters.

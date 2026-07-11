# First divergence under RT205

RT205 CPU with ORT_DISABLE_ALL is byte-exact to host ORT 1.27 at final `output0`
for every graph-valid full-model Stage43 fixture (F0, F5, F6, F7), the F8 blank
image, and the structured 0/1 F9 edge fixture. RT204 CPU is exact on the same
set. Stage43 F1-F4 are model4/block inputs, not full-model image tensors, and
were not relabeled.

RT205 SpacemiT EP never reaches an accepted full-model output. Its earliest
failure is the first quantized Conv (`/model.0/conv/Conv_token_21`) with the
historical clip-minmax compiler error and subsequent abort. A final-output
numerical divergence is therefore not defined for the EP path. The tiny 03
Q/DQ Conv without explicit kernel_shape is assigned to EP and byte-exact.

# Resize contract

Only the accepted model's nearest-neighbor resize surface is implemented.
Source coordinates are computed with the frozen integer mapping
`source = destination * input_extent / output_extent`. Resize is represented
inside the static package and is not a generic runtime mode dispatcher.

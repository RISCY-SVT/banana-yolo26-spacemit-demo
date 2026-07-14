# Depthwise V2 contract

DW2 folds the input-zero-point correction into prepare-time corrected bias, reuses C8 weights across adjacent X positions, separates vector interior from exact bounded borders, and uses one exact C8 E2c3 epilogue.

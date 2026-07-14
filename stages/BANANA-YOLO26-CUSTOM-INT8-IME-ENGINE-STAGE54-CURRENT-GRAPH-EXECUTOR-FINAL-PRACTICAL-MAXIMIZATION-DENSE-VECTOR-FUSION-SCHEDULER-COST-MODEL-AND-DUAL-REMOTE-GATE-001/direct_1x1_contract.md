# Direct 1x1 contract

P1 reads resident NCHWc8 C8 values through source/spatial/channel-block strides and does not materialize a full M12-by-K byte panel. It preserves signed storage, int32 accumulation, exact K1X_INT8_V1 Q62 RNE, and final NCHWc8 C8 stores.

The dispatcher is prepared from graph-known M/N/K and uses no string lookup or online autotuning in inference.

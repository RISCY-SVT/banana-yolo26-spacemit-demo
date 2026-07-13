# E2c2 sidecar contract

E2c2 is a bounded candidate under `K1X_INT8_V1`. It keeps accumulators and
M63 values in explicit RVV operations through `vsmul.e64`, zero-point add,
clamp, narrowing, and direct signed-storage output. It does not alter Q62
rounding or promote Q31.

Selection requires byte-exact board execution and either at least 8% model5
mean improvement or 5% complete full-model mean improvement, with no more
than 2% p99 regression. Otherwise the accepted E2c route remains selected.

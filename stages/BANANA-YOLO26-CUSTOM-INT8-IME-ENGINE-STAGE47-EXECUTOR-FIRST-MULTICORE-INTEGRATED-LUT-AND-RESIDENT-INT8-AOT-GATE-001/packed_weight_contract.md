
# Packed weight contract

Each Conv owns one immutable N16-packed signed-int8 weight object and one weight
sum table prepared before timing. All CPU0-3 workers share these objects. No
worker writes packed weights and no weight packing occurs during `run`.

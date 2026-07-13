# Persistent slice architecture

The executor prepares immutable packed weights and exact integer assets once, allocates one 2,252,800-byte arena, and uses one persistent CPU0-3 worker pool. Conv outputs remain NCHWc8. Eligible Concat segments are placed directly by producers; residual rescale work remains explicit. The selected scheduler counts only active workers for completion, and the explicit RVV LUT path handles activation/rescale tables.

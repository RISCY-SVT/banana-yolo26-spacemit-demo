# YOLO26 runtime model lineage

The primary surface remains the Stage42 manual static Q/DQ e2e model with hash
`30a94e...29c0c`. No model was re-exported, simplified, calibrated, or replaced.
CPU-bad exporter/XSlim candidates remain negative controls; a runtime cannot
repair their oracle semantics. QOperator and stripped-kernel models are diagnostic
branches and are not promoted to correctness authority.

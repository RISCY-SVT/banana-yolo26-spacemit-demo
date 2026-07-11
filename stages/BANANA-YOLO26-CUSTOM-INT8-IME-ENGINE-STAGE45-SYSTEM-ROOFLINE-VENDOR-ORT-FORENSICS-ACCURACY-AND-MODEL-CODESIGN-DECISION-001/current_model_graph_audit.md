# Current model graph audit

- Model SHA-256: `30a94e4738606673b5e0a73499cbc977167f046f8fa8637d6040ce744f429c0c`
- Contract: float32 `images` 1x3x640x640 -> float32 `output0` 1x300x6
- Nodes/initializers: `1069` / `1024`
- Conv/Q/DQ: `102` / `206` / `410`
- MatMul/Softmax/MaxPool/Concat/Split: 4 / 2 / 3 / 26 / 12
- Static Conv+MatMul/Gemm arithmetic: `2740153600` MAC, `5480307200` FLOPs at 2 FLOPs/MAC
- Graph-order activation liveness estimate: `19660848` bytes
- Peak all noninitializer estimate: `29463456` bytes
- Sum of materialized float outputs: `453896968` bytes

MAC/FLOP totals exclude non-MAC elementwise work. Liveness is a static graph-order
estimate, not an allocator or RSS measurement. The graph is a manual QDQ surface,
not assumed equivalent to a fused marketing export without proof.

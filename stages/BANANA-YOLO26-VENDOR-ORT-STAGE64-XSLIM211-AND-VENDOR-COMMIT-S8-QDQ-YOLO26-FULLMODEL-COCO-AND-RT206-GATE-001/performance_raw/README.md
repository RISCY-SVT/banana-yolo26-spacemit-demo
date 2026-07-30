# Raw performance samples

Per-inference samples, `/usr/bin/time` output, frequency snapshots, thermal
snapshots, and runner logs remain under the Stage64 task-owned NVMe root.
Compact distributions and artifact hashes are committed in
`performance_matrix.tsv`, `steady_state_timing.tsv`, `long_soak.tsv`, and
`thermal_frequency.tsv`.

The fixed-input placement/stability captures explicitly include ORT profiling.
The full-COCO CPU/EP timing streams do not enable profiling and provide the
primary matched steady inference comparison over identical images. The public
wrapper now requires `--enable-profiling` rather than enabling it implicitly.

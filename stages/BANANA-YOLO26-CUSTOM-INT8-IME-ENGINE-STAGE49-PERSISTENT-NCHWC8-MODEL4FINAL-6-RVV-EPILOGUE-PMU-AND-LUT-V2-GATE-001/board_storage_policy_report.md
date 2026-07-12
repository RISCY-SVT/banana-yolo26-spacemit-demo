# Board storage policy report

All Stage 49 binaries, packages, oracles, profiles, PMU data, logs, and temporary/cache paths are under the stage-owned NVMe `/data` root. The eMMC exception count is zero. Existing B120 ORT under `/home/svt` was read in place and not copied or modified. No broad cleanup was performed.

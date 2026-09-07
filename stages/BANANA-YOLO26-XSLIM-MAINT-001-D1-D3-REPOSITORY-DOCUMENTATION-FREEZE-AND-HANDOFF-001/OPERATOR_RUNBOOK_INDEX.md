# Source-Bound Operator Runbook

This is procedural knowledge from inspected source and accepted records, not
a fresh board validation. Do not install old instructions wholesale.

| Procedure | Canonical source | Status and constraint |
|---|---|---|
| Path/toolchain defaults | /data/k1x-env-overview.md; /data/build_scripts/01-env.sh | inspected; flags -march=rv64gcv_zvfh -mabi=lp64d; overlay/base sysroots read-only here |
| ncnn benchmark build | /data/build_scripts/build-ncnn-bench.sh | historical-command-verified by source inspection; NOT executed; source writes require new authorization |
| Vendor matched timing | vendor_ort_validation/stage65e_board_performance.sh | historical-command-verified; CPU0-3, intra=4/inter=1, fixed F0, separate tail; NOT executed |
| Runtime binding | accepted Stage65E runtime_asset_identity.tsv | container archive /data/vendor-runtimes/downloads/spacemit-ort.riscv64.2.0.6.tar.gz; core/EP under /data/vendor-runtimes/spacemit-ort/2.0.6/spacemit-ort.riscv64.2.0.6/lib; board namespace is separate |
| Board storage | k1x_board_storage_policy skill | historical reference only; board /data must be re-proved NVMe/RAM, never confused with container /data |
| Dual-remote auth | k1x_dual_remote_auth skill | Git SSH success is not REST release permission; environment token selection can override stored CLI identity; never log credentials or infer publication authority |
| Packet export | /data/lab/scripts/export-result-packet.sh | scanner first; check destination absent externally because helper replaces destination; no historical overwrite |

Known negative cases: tiny Conv requires explicit kernel_shape; parser/structural
acceptance is not EP placement proof; six separated bbox/confidence outputs
feed the exact float tail. Historical direct E2E had 100/100 score collapse.
Use direct thermal sysfs globbing; zero-byte thermal files are not evidence.
The old awk scalar named index was nonportable. Partial output roots do not
prove a reboot; current-stage boot/process evidence is required.

B2 remains universal rollback. C2 has its existing TIER-1 higher-AP waiver,
historical universal FAIL and a required application-specific threshold choice.
No new waiver, runtime default, board action, publication or co-design is
authorized by this handoff. Current numerical dependencies remain frozen.

# Deploying the K1X INT8 Executor 0.9.3

Keep the complete versioned release on board NVMe:

```text
/data/releases/banana-yolo26-k1x-int8-executor/0.9.3-stage60m-maintenance-runtime
```

Do not place packages, logs, or build trees on eMMC. Verify `SHA256SUMS`, then run
the bundled healthcheck before integration. The exact commands are in
[HANDOFF_EN.md](HANDOFF_EN.md) and [QUICKSTART_RU.md](QUICKSTART_RU.md).

Public profiles are selected with
`--profile compatibility|low-latency|low-latency-dedicated` or the C API
wake-policy field. No stage environment file is sourced. The dedicated profile
must be launched through the reversible wrapper:

```bash
$RELEASE/scripts/o2-system-profile.sh run -- COMMAND [ARG...]
```

O2 preserves the original boot, NVMe runtime, sysctls, THP, frequency, and
scheduler class. After an untrappable wrapper `SIGKILL`, run `restore-stale`.

The release is self-contained apart from documented system libraries. Its CLI
and healthcheck use a relative `$ORIGIN/../lib` RUNPATH. External consumers may
use the installed CMake target or pkg-config file.

To uninstall, restore O2, stop users of the library, and remove only the
versioned release directory. Nothing is installed under `/usr`.

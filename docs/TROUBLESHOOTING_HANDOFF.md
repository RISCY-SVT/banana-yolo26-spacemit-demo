# Handoff Troubleshooting

## Package manifest mismatch

Symptom: `package error: trusted package manifest SHA-256 mismatch`.

Action: use the package shipped in the same release and pass exactly:

```text
fab4a72cf524ce0a205ceca0384144f2eee7bc79dff3f4db8b7208614e8407be
```

Do not replace the expected value with a hash read from an arbitrary package;
that defeats stale-package protection.

## Asset missing, corrupt, extra, or symlinked

The loader intentionally rejects these states. Run `sha256sum -c SHA256SUMS` from
the release root and redeploy the complete versioned tree. Do not repair individual
binary assets by hand.

## Wrong profile or model

The release accepts only `K1X_INT8_V1_YOLO26N_640_FULL_GRAPH_001` and model SHA
`30a94e...429c0c`. Generate no substitute package on the board. Use a separate
authorized project for another graph.

## `run before prepare`, `prepare twice`, or `executor busy`

Follow the lifecycle in `INTEGRATION_GUIDE.md`. Prepare once. Serialize calls on
one handle or create independent handles. Inspect `y26_executor_last_error()` and
translate status with `y26_status_string()`.

## Invalid CPU topology

The release requires worker CPUs 0-3 and controller CPU4. Confirm `0-7` are online
and do not request CPU4-7 as IME workers.

## O2 snapshot already exists

```bash
scripts/o2-system-profile.sh status
scripts/o2-system-profile.sh restore-stale
scripts/o2-system-profile.sh dry-run
```

Do not delete the snapshot before restoring; it contains the rollback values.

## O2 reports an IRQ as not movable

This is expected for managed MSI/NVMe queues on the accepted kernel. The wrapper
does not claim they moved. Do not force broad affinity writes.

## Output hash differs

Confirm input bytes, input mode, package manifest, release binaries, and model
profile. `0xd43f5e018b415631` is valid only for the frozen bus fixture, not arbitrary
images. Use `--verify-determinism` for arbitrary repeated input and
`--verify-known-fixture` only with the provided fixture.

## `SIGILL`

Verify the board is K1X, the shipped binary is used, CPU0-3 run IME, and no global
ISA/compiler override was introduced. Preserve the fault PC/opcode and stop; do
not substitute raw opcodes or broaden `-march`.

## Latency is higher than documented

Check profile choice, CPU affinity, governor/frequency, thermal state, concurrent
load, and O2 status. Keep logs in `/data`; per-run writes can perturb timing.

## Uninstall

Restore O2, stop processes using the library, then remove only the versioned
release directory. No `/usr` files are owned by this release.

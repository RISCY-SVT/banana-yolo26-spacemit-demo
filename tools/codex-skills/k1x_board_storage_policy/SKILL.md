---
name: k1x-board-storage-policy
description: Validate and use the Banana-Pi K1X board NVMe /data workspace for builds, deployments, benchmarks, profiles, and raw artifacts without silently writing bulky data to eMMC.
metadata:
  short-description: Keep K1X board artifacts on NVMe
---

# K1X Board Storage Policy

## Preflight

Before the first board write, record `hostname`, `findmnt -T /data`,
`df -hT /data /`, `lsblk -o NAME,TYPE,FSTYPE,SIZE,MOUNTPOINTS`, and writable
directory checks. Require `/data` to be writable, backed by a non-`mmcblk`
filesystem, and to have sufficient free space. Do not silently fall back to
eMMC.

## Stage-Owned Root

Use `/data/k1x-stage-runs/$STAGE_ID` with task-owned subdirectories for binaries,
models, inputs, outputs, logs, profiles, temporary files, and caches. Set
`TMPDIR` and `XDG_CACHE_HOME` below that root where supported.

## eMMC Exceptions

Do not place bulky or persistent task artifacts under `/home`, `/tmp`,
`/var/tmp`, or `/root`. Record any technically unavoidable small exception with
its path, byte count, reason, and disposition.

## Cleanup Safety

Never clean broad board directories. Remove only an exact stage-owned root,
only after evidence is hashed and copied, and only when the task or operator
explicitly requests cleanup.

## Required Evidence

Preserve the storage preflight, stage-root manifest, free-space state, and eMMC
exception table with the task evidence.

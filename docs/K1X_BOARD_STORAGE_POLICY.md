# K1X Board Storage Policy

Board-side stage artifacts belong on the NVMe-mounted `/data` filesystem, not on the eMMC root filesystem.

## Preflight

Before the first board write, capture `hostname`, `findmnt -T /data`, `df -hT /data /`, and `lsblk -o NAME,TYPE,FSTYPE,SIZE,MOUNTPOINTS`. Require `/data` to be writable, backed by a non-`mmcblk` device, and to have adequate free space. Stop instead of silently falling back to eMMC.

## Stage-Owned Root

Use:

```bash
BOARD_STAGE_ROOT="/data/k1x-stage-runs/$STAGE_ID"
mkdir -p "$BOARD_STAGE_ROOT"/{bin,models,inputs,outputs,logs,profiles,perf,objdump,tmp,cache}
export TMPDIR="$BOARD_STAGE_ROOT/tmp"
export XDG_CACHE_HOME="$BOARD_STAGE_ROOT/cache"
```

Deploy new runners, models, tensor dumps, profiles, perf data, logs, temporary files, and caches only under that root. Shared installed libraries may be read in place.

## eMMC Exceptions

Do not place bulky or persistent task data under `/home`, `/tmp`, `/var/tmp`, or `/root`. For a small unavoidable exception, record the path, byte count, reason, why `/data` could not be used, and cleanup/disposition.

## Cleanup

Never clean broad board directories. Remove only the exact stage-owned root, after artifacts are hashed and copied, and only when explicitly requested.

## Evidence

Each board stage records a storage preflight, stage-root manifest, and eMMC exception table alongside its raw command log.

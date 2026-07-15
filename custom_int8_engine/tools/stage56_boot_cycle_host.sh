#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
    printf 'usage: %s <board-stage-root> <cycle>\n' "$0" >&2
    exit 2
fi

stage_root=$1
cycle=$2
case "$cycle" in
    ''|*[!0-9]*) printf 'invalid cycle: %s\n' "$cycle" >&2; exit 2 ;;
esac

board=${BANANA_SSH_TARGET:-svt@banana}
ledger="$stage_root/boot/reboot_ledger_raw.tsv"
utc_before=$(date -u +%Y-%m-%dT%H:%M:%SZ)
boot_before=$(ssh "$board" cat /proc/sys/kernel/random/boot_id)

if ! ssh "$board" test -e "$ledger"; then
    printf 'cycle\tutc_before\tboot_id_before\tutc_after\tboot_id_after\tuname\tcmdline_sha256\tstatus\n' |
        ssh "$board" "cat >'$ledger'"
fi

ssh "$board" 'sync; sudo -n reboot' || true

boot_after=
for _ in $(seq 1 240); do
    candidate=$(ssh -o ConnectTimeout=2 -o BatchMode=yes "$board" \
        'cat /proc/sys/kernel/random/boot_id' 2>/dev/null || true)
    if [[ -n $candidate && $candidate != "$boot_before" ]]; then
        boot_after=$candidate
        break
    fi
    sleep 1
done
if [[ -z $boot_after ]]; then
    printf 'board did not return with a new boot_id within 240 seconds\n' >&2
    exit 1
fi

utc_after=$(date -u +%Y-%m-%dT%H:%M:%SZ)
kernel=$(ssh "$board" uname -r)
cmdline_sha256=$(ssh "$board" sha256sum /proc/cmdline | awk '{print $1}')
printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\tpass\n' \
    "$cycle" "$utc_before" "$boot_before" "$utc_after" "$boot_after" \
    "$kernel" "$cmdline_sha256" | ssh "$board" "cat >>'$ledger'"

label="boot_b0_cycle$cycle"
ssh "$board" "set -euo pipefail
export Y26_STAGE56_BINARY='$stage_root/bin/yolo26_stage55_release'
'$stage_root/bin/stage56_benchmark_arm.sh' '$stage_root' '$label' low-latency 100 5
grep '^raw' '$stage_root/profiles/$label.log' | wc -l
cat /proc/sys/kernel/random/boot_id"

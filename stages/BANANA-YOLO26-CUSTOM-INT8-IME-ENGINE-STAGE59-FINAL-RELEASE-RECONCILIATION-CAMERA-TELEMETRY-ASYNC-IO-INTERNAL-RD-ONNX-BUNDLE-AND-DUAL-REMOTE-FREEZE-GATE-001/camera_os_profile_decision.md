# Camera CPU and IRQ Profile Decision

## Decision

Select the narrow reversible camera profile for the public performance
launcher. It pins the latest-frame capture thread and the single reviewed
`xhci-hcd:usb2` IRQ to CPU5. It does not apply executor O2, change the boot
profile, move unrelated IRQs, or permit IME outside CPU0-3.

## Evidence

On matched 640x480 MJPG full-GUI 180-second runs, decoded throughput remained
effectively fixed at 15.000552 versus 14.996945 FPS. Processed/displayed rate
improved from 6.620014 to 6.826387 FPS (`+3.117410%`). Consumer-loop mean,
p95, and nearest-rank p99 improved by 3.031970%, 3.862755%, and 5.402076%.
The gate of at least 1% processed-FPS gain is met.

The selected 30-minute public performance run processed 12,478 frames at
6.818437 FPS with a 146.540016 ms consumer-loop mean and 154.882199 ms p99.
CPU0-4 remained at 1.6 GHz, temperature stayed between 48 and 62 C, and the
original IRQ affinity was restored at exit.

## Safety

`camera-system-profile.sh` discovers exactly one numeric IRQ from the reviewed
action name instead of hardcoding IRQ 89. It validates CPU5 and effective
affinity, serializes use with `flock`, records the original affinity atomically,
and restores on normal exit, child failure, INT, TERM, and HUP. A separate
`restore-stale` action handles uncatchable process termination. Success,
failure, and real camera process-group signal tests all restored the original
`0-7` requested mask with effective CPU0 and left no state file.

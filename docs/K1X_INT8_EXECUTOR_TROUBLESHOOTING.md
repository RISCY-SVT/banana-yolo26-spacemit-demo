# K1X INT8 Executor Troubleshooting

## Package Rejected

Confirm the package path, `asset_hashes.tsv` SHA-256, profile ID, schema,
endianness, file sizes, and model lineage. Regenerate the package rather than
editing assets by hand.

## Loader Failure

Run `ldd` on the CLI. For image mode, ensure the release OpenCV libraries are
under the documented NVMe deployment root or in `LD_LIBRARY_PATH`. Do not copy
new runtime libraries to eMMC.

## SIGILL

Stop immediately and verify CPU affinity. IME code is approved only on CPU0-3.
Inspect the binary instruction inventory and selected `-march`; do not add a
raw-opcode workaround.

## Slow Or Unstable Timing

Check the performance governor, core frequencies, temperature, background
load, worker affinity, controller CPU, scheduler mode, and package/binary
hashes. Compare per-inference samples using the same statistical unit. Disable
diagnostic boundary capture and operation profiling for headline timing.

## Different Detections

Verify padding 114, OpenCV linear resize, RGB order, normalization by 255,
confidence threshold 0.001, no extra NMS, COCO category mapping, package hash,
and output tie policy. Compare integer boundaries before comparing rendered
boxes.

## Removal

Run the bundle's `uninstall.sh` with the exact NVMe deployment root. It must not
remove shared datasets, logs, or unrelated stage roots.

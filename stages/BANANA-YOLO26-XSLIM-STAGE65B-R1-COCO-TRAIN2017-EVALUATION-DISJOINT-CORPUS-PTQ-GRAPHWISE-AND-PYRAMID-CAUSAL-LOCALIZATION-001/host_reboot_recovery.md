# Host reboot recovery

Status: recovery complete.

The execution host restarted while the seeded PTQ matrix was running. The new
boot identity is `3f8dc916-9922-4677-83c9-5d7f50da7bb5`; its observed boot
time was approximately 2026-08-10 06:18 UTC.

The resume log proves that B1 and B2 second generations completed before the
restart and passed both deployable-ONNX byte equality and normalized Graphwise
report equality. B3 second generation started at 2026-08-10 04:25:45 UTC and
was interrupted during `Runtime Calibration (BlockWise)`. It has no generation
summary, no emitted ONNX model, and no output files.

The interrupted B3 tree contains three regular files, eight directories, no
symlinks, and 4,085 regular-file bytes. Its sorted file-content manifest hash
is `114b39f93aa8771923b3deaf510ed908f3458071c497abe3c93e3e0105860475`.
It is retained as non-decision fault evidence under
`faults/host-reboot-2026-08-10T0618Z-boot-3f8dc916/B3-run2-incomplete` and was
not resumed in place.

No stale Stage65B-R1, XSlim, ONNX Runtime, COCO evaluator, or PTQ process was
present after restart. `/data` remounted read-write on XFS with approximately
465 GB free. Memory and swap were healthy and no swap was in use.

The host does not retain a readable persistent journal for the preceding boot,
and access to the current kernel ring buffer is denied. Therefore the reboot's
root cause cannot be proven from this container and is classified `unknown`;
the report does not infer an OOM, filesystem fault, kernel panic, or power
failure.

Post-restart immutable checks passed for the Banana protected and research
refs on both remotes, the custom-executor tree, XSlim refs/tags on both
remotes, and the accepted `/data/ncnn` HEAD/tree/dirty diff. Completed B1-B6
first generations and B1/B2 second generations were rehashed through the
fail-closed generation checker before work resumed.

A new empty B3 second-generation directory was then used for the bounded
restart. It completed in 4,169.765145 seconds and emitted deployable model
SHA-256
`b781da491ec13c7d7da7f528dd7120afe4caa43b831237b8fca85c54b6a81046`,
identical to B3 run1; the normalized Graphwise reports were also identical.
B4-B6 second generations, every host semantic and boundary gate, B1-B6 full
COCO, and all six required full-COCO hybrid arms subsequently completed.

The only post-recovery tooling failure occurred after the complete matrix had
reported `full-matrix-pass`: Python's default 128 KiB CSV field limit rejected
a valid boundary-histogram TSV field during report derivation. The report
reader was bounded at 16 MiB, compiled, and rerun successfully without
repeating any PTQ or evaluation. Final process inspection found no surviving
Stage65B-R1, XSlim, ONNX Runtime, COCO, or PTQ process.

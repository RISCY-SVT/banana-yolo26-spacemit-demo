# Workspace preflight

Status: pass; host-only Stage65B-R1 matrix complete.

The Banana protected main, continuing vendor-research branch, custom-executor
tree, XSlim main and long-lived branch, XSlim release/evidence tag objects and
peeled commits, released wheel, and accepted Stage65B result packet all matched
the launch identities before dataset acquisition or repository mutation.

The Banana research worktree began clean at
34154fa9f8311c8df9934cf586629afd4b6f6a75. The XSlim repositories are read-only
for this stage. The K1X board was not contacted or executed.

The /data/ncnn repository retained accepted HEAD
a245a70c641a1f20f357c65d103e5f9e50fe84a1, tree
20b96dadbd1fc0a53159cb35749719e967b55906, and its same three pre-existing
dirty paths.

The full official COCO train2017 and train/val annotation archives fit within
the authorized download budget. Selective extraction left more than 464 GB
free on /data after acquisition.

The released XSlim implementation uses unseeded Python, NumPy, Torch, and
OpenCV random surfaces. Six first attempts were stopped before model output and
retained as non-decision evidence. Decision runs used a stage-local launcher
that seeds those libraries and enables deterministic Torch algorithms while
executing the immutable released package. B1-B6 all completed two clean,
byte-identical deployable-model generations.

The execution-host reboot and exact recovery boundary are recorded in
`host_reboot_recovery.md`. No board connection or command occurred before or
after recovery.

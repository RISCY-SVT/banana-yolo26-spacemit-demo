# Current-Graph Final Freeze Record

Stage59 closes release maintenance for the unchanged YOLO26n-640 graph on
branch `yolo26-custom-int8-engine`. The exact executor source checkpoint built
into the 0.9.2 binaries is
`5a08e0e4f4c9157cdcdcab976169006be431efd9`; the canonical containing
publication SHA and local/GitHub/GitLab parity are recorded in the post-push
result packet because a tracked file cannot contain the hash of its own commit.

## Frozen Identity

| Identity | Value |
|---|---|
| Integer contract | `K1X_INT8_V1` |
| Full-graph profile | `K1X_INT8_V1_YOLO26N_640_FULL_GRAPH_001` |
| Source model SHA-256 | `30a94e4738606673b5e0a73499cbc977167f046f8fa8637d6040ce744f429c0c` |
| Package manifest SHA-256 | `fab4a72cf524ce0a205ceca0384144f2eee7bc79dff3f4db8b7208614e8407be` |
| Prediction SHA-256 | `cda5c8c7a46d61d9c90f6292001eea190cb8f6617efe647a33dc6134dd57ccda` |
| Known output hash | `0xd43f5e018b415631` |
| Release / SONAME / ABI | `0.9.2` / `1` / `1` |

The final same-session O2 comparison measured 134100.921 us mean and
135724.150 us p95 over 1,000 samples. The separate 13,500-run soak measured
133381.666593 us mean, 135151.521 us p99.9, and 135853.000 us maximum. Full
COCO remained 5000/5000 and byte-identical at mAP50-95
0.3707408944391919.

## Selected Maintenance Policy

- Runtime delivery contains the SDK and prepared package, but no source ONNX.
- Internal-R&D delivery additionally contains the exact source ONNX and
  internal-only provenance/license records; external redistribution is not
  cleared.
- Public camera presets are `quality-wide` and measured `performance`.
- The fast camera launcher uses bounded asynchronous recording and a reversible
  CPU5/xHCI IRQ profile. O2 remains an explicitly named diagnostic for camera
  use and the selected dedicated pure-model profile.
- Original boot, supported board image, and NVMe `/data` remain selected. No
  persistent boot, sysctl, eMMC, or kernel change is part of the release.

## Maintenance Boundary

Allowed work is limited to correctness/security fixes, build or dependency
regressions, board-kernel compatibility, and documentation/release maintenance.
New performance research, Q31, model/layout/resolution changes, training,
student selection, co-design, a vendor runtime lane, or CPU4-7 IME requires an
explicit unfreeze and separate branch/project authorization.

This is an optimized engineering handoff, not production certification and not
a 20 FPS claim.

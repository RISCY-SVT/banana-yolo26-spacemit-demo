# Stage58 Post-Publication Addendum

This is an append-only Stage59 correction. It does not modify Stage58 raw evidence.

## Publication identity

Stage58 contained eight commits, not seven. Its final local, GitHub, and GitLab
publication point was:

```text
5465b68cf41f5547d6455dc229b196c77e590743
```

The Stage58 `commit_inventory.tsv` and `final_remote_parity.tsv` were generated
before the final evidence commit and therefore stopped at
`bf965e83e650839a95c32027f1a6443d35d44caa`. Stage59 Gate 0 independently
verified both remotes at the final Stage58 SHA before any Stage59 edit.

## Camera terminology

The Stage58 camera fields require these narrower interpretations:

- `60 FPS` was requested and reported by the V4L2/OpenCV configuration; it was
  not a measured sensor rate.
- `9.980414 FPS` was the OpenCV decoded-frame return rate in that run.
- `40.869097%` counted application queue-slot replacements. It did not count
  all possible camera, USB, driver, or remote-device losses.
- `218.715619 ms` measured from an OpenCV `read()` return to the display call.
  It was not sensor-to-screen latency.
- The prior 640x480 comparison mixed headless and GUI surfaces and therefore
  did not establish a matched 12.9% camera-preset gain.

Stage59 metric schema v2 replaces the ambiguous public names.

## Pipeline table

Stage58 `final_pipeline_performance.tsv` left three rows without values. The
reconciled table in this Stage59 directory records the exact available source
for each row. Values inherited from Stage57 are labeled inherited and are not
represented as Stage58 remeasurements.

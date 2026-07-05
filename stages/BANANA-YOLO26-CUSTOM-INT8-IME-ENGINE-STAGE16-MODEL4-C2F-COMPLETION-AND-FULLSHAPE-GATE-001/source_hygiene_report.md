# Source Hygiene Report

stage_id: `BANANA-YOLO26-CUSTOM-INT8-IME-ENGINE-STAGE16-MODEL4-C2F-COMPLETION-AND-FULLSHAPE-GATE-001`
status: `pass`

Checks:

- `git diff --check`: pass
- `find custom_int8_engine stages -type l -print`: pass, no symlinks reported
- changed-file secret-like scan excluding `commands.txt`: pass, `scanned_files=40`, `status=clean-no-matches`
- changed path ASCII/control-character scan: pass, `scanned_paths=49`, `status=clean-ascii-no-control-no-backslash`
- large generated ONNX/model/tensor dumps: not staged
- `/data/ncnn` mutation: none
- `/data/banana-yolo11-spacemit-demo` mutation: none

Notes:

The broad scan over all historical `stages/**/commands.txt` finds self-matches from prior secret-scan command strings. The changed-file filtered scan excludes command logs and reports no secret-like matches.

# Source Hygiene Report

## Scope

The scan covered all files changed from Banana launch commit `d3afe144...` and
XSlim launch commit `3e275c64...`, plus both current working trees.

## Results

- Banana: 47 changed tracked files; largest file 88,735 bytes.
- XSlim: 26 changed tracked files; largest file 30,519 bytes.
- No changed file exceeds 1 MiB.
- No changed file is a symlink or multiply linked hardlink.
- No ONNX, NPZ, NPY, image, archive, object, library, model or weight payload
  is tracked by this stage.
- No private-key marker, GitHub/GitLab token prefix, AWS key, live
  Authorization header, raw credential path or Codex configuration path was
  found.
- No NUL-bearing binary content was found in the changed set.
- `git diff --check` and `git diff --cached --check` pass in both repositories.
- No shell file changed, so `bash -n`/ShellCheck have no stage delta surface.
- Both worktrees were clean at the scan point.

## Verification

- XSlim full pytest: 207 passed, 4 inherited FP16-converter warnings, 65
  subtests.
- Focused reconstruction pytest: 16 passed.
- Banana candidate-tool pytest: 12 passed.
- Ruff passes for both changed Python surfaces.
- Strict mypy passes for the three new XSlim modules; the six changed legacy
  modules have the same inherited strict-error count as the frozen base.
- `compileall`, fresh wheel/sdist install, `pip check`, CLI help and package
  smokes pass.

Raw models, predictions, reconstruction arrays, bootstrap NPZ data, package
artifacts and test logs remain in the stage raw root and are excluded from Git
and the result packet.

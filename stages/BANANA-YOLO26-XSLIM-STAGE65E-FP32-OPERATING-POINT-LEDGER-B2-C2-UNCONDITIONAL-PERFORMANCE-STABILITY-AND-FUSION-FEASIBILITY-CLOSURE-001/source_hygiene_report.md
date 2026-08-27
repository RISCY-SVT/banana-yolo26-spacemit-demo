# Source hygiene report

Status: `pass`.

Scanned files: `96`. Forbidden model/data/runtime payloads: `0`. Files over 16 MiB: `0`. Symlinks: `0`. Hard-linked files: `0`. Secret-pattern hits: `0`. `git diff --check`: `pass`. Compile/syntax/structured-data verification matrix: `pass`; Ruff and shellcheck are explicitly recorded as unavailable rather than installed into the immutable environment.

Raw ONNX, predictions, images, samples, provider artifacts and large timing logs remain under the Stage raw root. No credential or authorization material is part of the tracked evidence.

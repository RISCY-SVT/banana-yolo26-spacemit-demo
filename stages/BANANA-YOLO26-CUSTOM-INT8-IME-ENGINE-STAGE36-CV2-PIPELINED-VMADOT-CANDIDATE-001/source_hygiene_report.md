# Source Hygiene Report

## Scope

Changed tracked and untracked stage files under:

- `custom_int8_engine`
- `stages`

## Checks

| check | status | notes |
| --- | --- | --- |
| `git diff --check` | pass | exit code 0 |
| symlink scan | pass | `find custom_int8_engine stages -type l -print` produced no entries |
| changed-file secret/path scan | pass | retry loop exit code 0, findings 0 |
| `/data/ncnn` mutation | pass | Stage36 did not modify `/data/ncnn` |
| YOLO11 production repo mutation | pass | Stage36 did not touch `/data/banana-yolo11-spacemit-demo` |
| forbidden generated binaries in git | pass | build products remained under `.deps` and board/log dirs |

The first secret/path scan used `xargs` and returned an `xargs` bookkeeping code with no stdout/stderr findings. It was rerun with an explicit per-file loop and completed cleanly. The retry result is the accepted hygiene result.

Raw evidence:

- `git_diff_check.stdout`
- `symlink_scan.stdout`
- `changed_file_secret_path_scan_retry.stdout`
- `changed_files_for_scan.txt`

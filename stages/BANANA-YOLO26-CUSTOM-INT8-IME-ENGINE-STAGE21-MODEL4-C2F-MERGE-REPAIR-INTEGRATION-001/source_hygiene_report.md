# Source Hygiene Report

stage_id: `BANANA-YOLO26-CUSTOM-INT8-IME-ENGINE-STAGE21-MODEL4-C2F-MERGE-REPAIR-INTEGRATION-001`

## Checks

```text
git_diff_check: pass
symlink_scan: pass
changed_only_secret_scan: pass
cross_track_ncnn_mutation: not performed
large_artifacts_staged: no
```

## Notes

The broad repository secret-like scan found historical self-matches in earlier stage `commands.txt` files where the scan command itself was recorded. The changed-only scan excluding `commands.txt` produced no findings.

Stage21 did not stage or export `.deps` tensor dumps, board binaries, credentials, SSH keys, `.env` files, `/data/.codex`, `/home/svt/.codex`, or `/control` content.

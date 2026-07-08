# Source Hygiene Report

Stage ID: `BANANA-YOLO26-CUSTOM-INT8-IME-ENGINE-STAGE30-VMADOT123-DIRECT-CONV-PROOF-AND-KERNEL-GATE-001`

Checks:

| check | status | log |
| --- | --- | --- |
| `git diff --check` | pass | `/data/ncnn-logs/ai-team/2026-07-08_09-43-20/BANANA-YOLO26-CUSTOM-INT8-IME-ENGINE-STAGE30-VMADOT123-DIRECT-CONV-PROOF-AND-KERNEL-GATE-001/run_logs/final_git_diff_check_stage30.log` |
| `git diff --cached --check` | pass | `/data/ncnn-logs/ai-team/2026-07-08_09-43-20/BANANA-YOLO26-CUSTOM-INT8-IME-ENGINE-STAGE30-VMADOT123-DIRECT-CONV-PROOF-AND-KERNEL-GATE-001/run_logs/final_git_diff_cached_check_stage30.log` |
| symlink scan | pass, no symlinks printed | `/data/ncnn-logs/ai-team/2026-07-08_09-43-20/BANANA-YOLO26-CUSTOM-INT8-IME-ENGINE-STAGE30-VMADOT123-DIRECT-CONV-PROOF-AND-KERNEL-GATE-001/run_logs/final_symlink_scan_stage30.log` |
| secret-like scan | pass with documented self-match | `/data/ncnn-logs/ai-team/2026-07-08_09-43-20/BANANA-YOLO26-CUSTOM-INT8-IME-ENGINE-STAGE30-VMADOT123-DIRECT-CONV-PROOF-AND-KERNEL-GATE-001/run_logs/final_secret_scan_stage30.log` |
| ASCII/control scan | pass | `/data/ncnn-logs/ai-team/2026-07-08_09-43-20/BANANA-YOLO26-CUSTOM-INT8-IME-ENGINE-STAGE30-VMADOT123-DIRECT-CONV-PROOF-AND-KERNEL-GATE-001/run_logs/final_ascii_control_scan_stage30.log` |

Secret scan note:

The only finding was the scan command pattern recorded inside Stage30 `commands.txt` itself. This is a command-log self-match, not a secret. No private keys, credentials, OAuth tokens, or Authorization headers were found in changed source/stage artifacts.

Large/generated artifact policy:

- Build outputs remain under `.deps/` and are not staged.
- Raw board logs remain under `/data/ncnn-logs/ai-team/2026-07-08_09-43-20/BANANA-YOLO26-CUSTOM-INT8-IME-ENGINE-STAGE30-VMADOT123-DIRECT-CONV-PROOF-AND-KERNEL-GATE-001`.
- Repo stage artifacts are small text reports.

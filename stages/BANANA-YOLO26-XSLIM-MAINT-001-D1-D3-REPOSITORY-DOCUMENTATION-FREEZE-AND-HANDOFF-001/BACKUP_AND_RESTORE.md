# Backup and Restore

Restricted local backup: /home/svt/.local/state/k1x-maint001-backup/maint001-019f26cb
Relevant start refs are in banana-before.bundle and xslim-before.bundle;
git bundle verify passed for both. Final bundles/receipts are added at closure.
Bundles preserve committed selected histories, not ignored files, virtualenvs,
models, datasets, raw result roots or unrelated branch histories. Both target
worktrees were initially tracked/untracked clean; there were no target dirty
changes to rescue. The accepted ncnn three-file diff is preserved separately.
Approved non-secret helpers are separately byte-hashed in snapshot_coverage.tsv.
Credentials, account exports, raw conversations and entire homes are excluded.

For verification use git bundle verify and git bundle list-heads. To inspect
recovery, clone a chosen bundle with --no-checkout into a NEW disposable private
directory, then verify the expected commit/tree before checkout. Never restore
over an active worktree. For helpers compare SHA-256 before and after copying
to a disposable inspection folder; do not source or execute board scripts.

No prune/gc/cleanup or deletion of evidence was performed. No second approved
backup location was supplied; off-host redundancy and full restore execution
are incomplete, not evidence of data loss. /data/ai-handoff/k1x-transition was
not created because managed mirror filters could not be proven. Private local
handoff copies are available without making an unsafe sync assumption.

# Backup and Handoff Coverage

Four existing selected-ref bundles (Banana/XSlim before/after MAINT-001) were
restored with `git clone --no-checkout` into new restricted disposable local
repositories. Exact selected refs/HEAD/tree, full `git fsck`, no alternates,
and tracked README readback passed. Hooks, filters and helper scripts were not
executed. Four allowed helper snapshots plus the accepted ncnn dirty diff were
copied into a fresh private inspection directory and hash-checked.

The two after-bundle file hashes/sizes match original packet receipts. The two
before-bundle creation receipts did not record file SHA/size. Current bytes are
attested now and exact selected Git contents were independently restored; a
historical before-byte-hash comparison is not fabricated.

Versioned final docs-only selected-ref bundles are preserved independently;
original bundles are never replaced. Exact private receipts are referenced by
the final result packet without exporting bundle/helper contents.

Local selected-ref restore: pass. This is not a backup of ignored datasets,
models, venvs, all raw experiments, unrelated branches or all chat histories.
Second backup destination: not-provided-not-attempted. Full off-host backup:
incomplete. No uploads, cleanup, prune, gc or evidence deletion were performed.

Five-source assimilation remains independent: codex-container is a partial
source-bound procedural export; claude-chat, gpt-old, gpt-new and gpt-current
are missing. Audit files are not chat transcripts. No new external material was
supplied or reconstructed from quotations. Missing owner audit attachments,
inherited mypy/REUSE debt and unverified external links remain explicit gaps,
not a blocker on already verified D1-D3 source publication.

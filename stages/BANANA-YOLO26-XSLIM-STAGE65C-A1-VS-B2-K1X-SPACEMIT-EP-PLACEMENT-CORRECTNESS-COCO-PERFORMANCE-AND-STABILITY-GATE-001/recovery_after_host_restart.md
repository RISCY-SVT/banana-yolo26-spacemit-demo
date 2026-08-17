# Recovery after host restart

- The Stage workspace, accepted input identities, Git refs, raw board outputs,
  and completed H500 predictions were revalidated before resuming.
- No board inference, PTQ, model-generation, performance, soak, or publication
  process remained active after recovery.
- The board retained boot ID `0a0691d1-7502-44c3-903b-444dba83c1d9` throughout
  the Stage; the external host restart did not restart the board.
- The completed B2/A1 CPU and EP H500 prediction files passed their expected
  count, finite-output, and SHA-256 checks.
- One host bootstrap attempt that ran two memory-heavy pair evaluations at the
  same time was stopped after bounded throughput assessment. Its partial output
  directories are explicitly marked `attempt1-concurrent-contention` and are
  excluded from accepted evidence.
- Both required remaining bootstrap pairs were resumed sequentially from new
  empty output directories. No partial replicate payload is accepted.

The recovery changed no model, runtime, source branch, protected project, or
board system setting.

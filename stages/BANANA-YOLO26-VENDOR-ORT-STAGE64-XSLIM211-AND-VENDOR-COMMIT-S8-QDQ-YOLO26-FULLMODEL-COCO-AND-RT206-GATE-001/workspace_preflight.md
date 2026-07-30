# Stage64 workspace preflight

Captured at `2026-07-30T07:08:43Z`.

The new research worktree was created from the accepted Stage63 commit
`94b86a6bf011cc83fefaf2a960191e97a8daf728` at:

```text
/data/worktrees/banana-yolo26-xslim211-s8-qdq-validation
```

Preflight verified exact local, GitHub, and GitLab identities for protected
main and the accepted Stage63 branch. Both protected annotated tags peeled to
their expected commits. The accepted ORT 2.0.6 archive and Stage63 result
packet were found by exact hash before reuse.

The base repository had no Stage64 branch or worktree before creation. The
worktree was clean. No accepted custom-executor source, package, archive, tag,
or release root was selected as a write target.

All Stage64 generated models, environments, runtime files, measurements, and
predictions use the task-local NVMe root under `/data/k1x-stage-runs/`. Git is
limited to new validation tools, compact reports, sanitized tiny controls, and
the public issue draft/bundle.

The board was reachable by SSH. It reported Bianbu 2.2.1, Linux 6.6.63,
SpacemiT X60 CPUs 0-7, the performance governor, 1.6 GHz on all CPUs, and an
NVMe-backed `/data` mount. No stale Stage64 process or stale task profile was
present.

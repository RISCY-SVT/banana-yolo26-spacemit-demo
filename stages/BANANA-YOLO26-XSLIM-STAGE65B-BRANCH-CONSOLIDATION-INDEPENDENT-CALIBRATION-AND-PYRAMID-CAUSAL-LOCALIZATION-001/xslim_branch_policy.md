# XSlim branch policy

The policy-only commit is `2bc1be073c84ffd8b4e22e372b8f33de4218f9f8`
on `riscy/k1x-yolo26`, based on clean release commit
`12647b4a79fe5ec9a3973515a17cece4cb83daf4`.

It adds only `RISCY_BRANCH_POLICY.md` and establishes:

- `main` is the exact upstream mirror;
- `riscy/k1x-yolo26` is the sole downstream source branch;
- no branch is created per stage;
- downstream changes are sequential and reviewable;
- releases use immutable annotated version tags;
- evidence lives in result packets/shared logs plus occasional immutable evidence tags;
- force pushes and tag rewrites are prohibited;
- upstream/vendor branches are read-only comparison inputs.

No package or runtime source file changed.

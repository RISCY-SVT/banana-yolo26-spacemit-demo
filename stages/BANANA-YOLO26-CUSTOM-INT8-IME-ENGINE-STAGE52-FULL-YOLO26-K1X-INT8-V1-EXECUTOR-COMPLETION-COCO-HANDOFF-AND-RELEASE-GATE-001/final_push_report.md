# Final push report

The final push is intentionally a post-commit operation. Embedding the final
commit SHA or the observed remote SHA in this same tracked tree would change
that SHA. The authorized procedure is:

1. fetch all remotes;
2. prove the tracking branch is an ancestor of local `HEAD`;
3. perform one normal fast-forward push to
   `github/yolo26-custom-int8-engine`;
4. compare `git rev-parse HEAD` with `git ls-remote`.

Exact commands, exit codes, local SHA, and remote SHA are preserved in the raw
command ledger and exported result packet. Force push, rebase, merge, PR, tag,
and GitHub Release operations are not used.

# Final push report

The final push is intentionally a post-commit operation: embedding the final commit SHA or its
post-push observation in the same tracked tree would change that SHA. The authorized procedure
is a fetch, ancestor check, normal fast-forward push, and `ls-remote` parity check. Exact command
output and the immutable final SHA are recorded in the exported result packet and final console
response. No force push or pull request is permitted.

# Codex skill storage policy

Created the local reusable `k1x_board_storage_policy` skill and self-test. Added
short references from the local task-template, environment-sanity, and deployment
skills. The policy requires NVMe verification, a stage-owned `/data` root,
TMP/cache redirection, no silent eMMC fallback, documented exceptions, and safe
stage-owned cleanup. The writable local skill tree is not a Git repository and is
outside the project commit. Before/after hashes and a sanitized diff are recorded.
Codex may require a future process restart to discover the new skill automatically.

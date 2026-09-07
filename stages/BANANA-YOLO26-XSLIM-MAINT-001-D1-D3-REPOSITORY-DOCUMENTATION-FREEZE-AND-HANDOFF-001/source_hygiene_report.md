# Source Hygiene

Compact evidence JSON/TSV parse checks pass; only nonempty text files under 2 MB, no symlink/hardlink, model/runtime payload or detected token/private-key content. Source paths are provenance references, not copied credentials. The authoritative bridge scanner is additionally run before commit/export. Runtime source and dependency identity are unchanged. No wildcard staging, branch/tag mutation, prune or board command.

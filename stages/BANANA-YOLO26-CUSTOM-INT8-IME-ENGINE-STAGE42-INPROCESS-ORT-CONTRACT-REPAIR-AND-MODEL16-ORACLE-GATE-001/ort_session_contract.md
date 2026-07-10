# ORT Session Contract

## Diagnostic contract

The mandatory host/board comparison uses:

```text
optimization: ORT_DISABLE_ALL
execution mode: sequential
intra-op threads: 1
inter-op threads: 1
memory pattern: disabled
CPU arena: disabled
intra/inter thread spinning: disabled through session config entries
provider: default CPU only; no vendor EP appended
```

The vendor log prints an internal thread-pool structure with `allow_spinning:1`, while the same log records both explicit session config entries as `0`. Stage42 reports the requested options as accepted but does not infer an unobservable internal implementation detail from this inconsistency.

## Operational contract

The Stage41 accepted export/output contract is:

```text
optimization: ORT_ENABLE_ALL
execution mode: sequential
intra-op threads: 1
inter-op threads: 1
memory pattern: enabled
CPU arena: enabled
thread spinning: enabled
```

Host ORT 1.27.0 under this contract is the fixed correctness authority. Board ORT 1.20.2+spacemit under the same settings is an integration and timing scaffold only.

No BASIC/EXTENDED matrix was needed after DISABLE and ALL decisively bounded the mismatch. See `ort_session_matrix.tsv`.

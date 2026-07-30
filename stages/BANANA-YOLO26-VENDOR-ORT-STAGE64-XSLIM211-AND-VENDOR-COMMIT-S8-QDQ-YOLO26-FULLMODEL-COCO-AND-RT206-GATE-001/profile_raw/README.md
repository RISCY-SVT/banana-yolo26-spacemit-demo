# Raw ORT profiles

Raw ORT JSON profiles remain under the Stage64 board evidence root on NVMe.
The committed `provider_assignment.tsv` and provider-attribution reports are
derived from those immutable profiles. Raw profiles are excluded from Git and
the result packet because they are large and may contain task-local paths.

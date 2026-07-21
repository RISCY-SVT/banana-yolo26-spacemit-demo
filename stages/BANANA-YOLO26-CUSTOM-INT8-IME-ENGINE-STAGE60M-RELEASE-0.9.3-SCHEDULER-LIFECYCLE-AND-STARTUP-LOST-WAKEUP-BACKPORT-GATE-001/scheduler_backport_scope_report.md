# Stage60M Scheduler Backport Scope

## Authority and source

The maintenance baseline is `175c1d939cc93fba0e730dba3f1281704e8f25b9`. The two repairs are mapped from Stage60 commit `5000ac5838225ff53ff296cdd066120410f40448`; that commit is evidence only and is not cherry-picked.

The selected patch changes exactly:

- `custom_int8_engine/kernels/conv_threaded.cpp`: readiness publication uses the same mutex as the creator's condition-variable predicate.
- `custom_int8_engine/kernels/stage49_persistent_slice.cpp`: active-window transitions are serialized, park/wake acknowledgements are counted, and unchanged job generations are rejected.

## Scope proof before application

The selected hunks do not alter Conv, MatMul, depthwise, requantization, RNE, saturation, LUT, tensor layout, arena planning, package loading, graph identity, profile identity, operator dispatch, public headers, exported ABI, camera policy, or model assets. They alter only synchronization and lifecycle state in the existing worker implementations.

The complete Stage60 implementation diff touches 39 paths. Every path other than the two kernel files is rejected in `scheduler_backport_rejected_stage60_hunks.tsv`. In particular, Stage60 static-resolution support, dynamic dimensions, package/profile handling, fixtures, board sweep tools, and camera-resolution behavior are not part of this maintenance release.

## Selection decision

The five mapped scheduler hunks in `scheduler_backport_source_map.tsv` are selected as one indivisible liveness repair. No other Stage60 hunk is selected. Validation must prove ABI1/SOVERSION1 identity, arithmetic disassembly equivalence, exact output and COCO identity, and bounded performance non-regression before release.

## Additional-file justification

Files outside the two implementation units are limited to the following maintenance roles:

- `custom_int8_engine/tests/CMakeLists.txt` and the two `test_stage60m_*` sources add watchdog-bounded startup and active-window regression coverage. They are not linked into release artifacts.
- Top-level and engine `CMakeLists.txt`, `config/release.env`, `config/k1x-int8-executor-safe.conf`, and `stage58_capability_api.c` change only the release string from 0.9.2 to 0.9.3, keep ABI/SOVERSION at 1, and remove official-release RPATH/RUNPATH as required by the binary audit.
- The primary handoff documents and `docs/RELEASE_NOTES_0.9.3.md` update only versioned installation paths and describe the scheduler maintenance release. No execution policy or camera behavior changes.
- `scripts/k1x-int8-executor/{benchmark,smoke-test}.sh` and `scripts/y26_executor_common.sh` provide an explicit release-root-relative `LD_LIBRARY_PATH` on the board. This is required because the clean delivery intentionally has no RPATH/RUNPATH; it does not change the invoked binary, profile, package, or camera options.
- Files under this Stage60M report directory are append-only source maps, raw-result summaries, release hashes, and closure reports.

No other tracked path is changed. The final source-hygiene report verifies this list against `git diff --name-status 175c1d9..HEAD`.

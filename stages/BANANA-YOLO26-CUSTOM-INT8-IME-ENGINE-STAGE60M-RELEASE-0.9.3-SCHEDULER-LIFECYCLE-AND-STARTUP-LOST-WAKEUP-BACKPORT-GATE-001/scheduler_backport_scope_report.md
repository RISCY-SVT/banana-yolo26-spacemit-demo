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

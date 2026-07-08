# Stage 31 Hypotheses

stage_id: BANANA-YOLO26-CUSTOM-INT8-IME-ENGINE-STAGE31-VMADOT123-DIRECT-CONV-INTEGRATION-001

H1: Stage30 `smt.vmadot1/2/3` semantics remain parser/assembler/disassembly/board/oracle valid on CPU0-3 in the current branch.

H2: A bounded real-node direct/sliding 3x3 sidecar for `/model.4/m.0/cv1/conv/Conv` can be built without changing the accepted Stage28 runner path or default backend policy.

H3: The direct/sliding sidecar must match the current accepted MMT4D/scalar corrected int32 output exactly before any timing result is considered.

H4: The direct/sliding candidate is useful only if it beats same-thread MMT4D by at least 1.20x or current best threaded MMT4D by at least 1.15x after panel-build, correction, writeback, and threading overhead are included.

H5: If direct/sliding correctness passes but panel-build or duplicate-row overhead dominates, Stage31 must close as a negative proof and must not integrate the candidate.

H6: `vmadotn` is proof-only in Stage31. Even if parser/board/oracle proof appears, it is not authorized for engine integration in this stage.

H7: Unsigned and mixed signedness dot-product variants are report-only in Stage31 unless they expose a hard dtype-contract blocker in the current selected path.

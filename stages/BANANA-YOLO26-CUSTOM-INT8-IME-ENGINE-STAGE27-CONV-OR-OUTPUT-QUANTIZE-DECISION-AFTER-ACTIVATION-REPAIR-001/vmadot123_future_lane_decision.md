# vmadot123 Future Lane Decision

stage_id: BANANA-YOLO26-CUSTOM-INT8-IME-ENGINE-STAGE27-CONV-OR-OUTPUT-QUANTIZE-DECISION-AFTER-ACTIVATION-REPAIR-001

## Current Evidence

Stage27 does not implement `vmadot1`, `vmadot2`, `vmadot3`, or `vmadotn`.

The current selected `/model.4` cut Conv path is still low-utilization under the existing plain `smt.vmadot` MMT4D implementation:

```text
/model.4/m.0/cv1/conv/Conv threaded_GMAC_s: 3.779804
/model.4/m.0/cv2/conv/Conv threaded_GMAC_s: 4.545240
/model.4/cv2/conv/Conv threaded_GMAC_s: 6.251993
```

The dominant single Conv is `/model.4/cv2/conv/Conv`, a 1x1 Conv. The two branch 3x3 Conv nodes are collectively material but not the only bottleneck.

## Decision

Stage27 selects a tile/prepack/correction future stage first, not an immediate `vmadot123` proof lane, because:

```text
1. The selected cut has both 3x3 and 1x1 dominant Conv work.
2. The 1x1 post-Concat Conv is the largest single Conv bucket.
3. Existing cluster0 threading is already useful and exact.
4. A narrow MMT4D/tile/prepack/correction stage can test lower-risk improvements across all current Conv nodes.
```

## Future Proof Lane Trigger

Open:

```text
BANANA-YOLO26-CUSTOM-INT8-IME-ENGINE-VMADOT123-SEMANTICS-AND-CONV-APPLICABILITY-001
```

only if a future tile/prepack stage confirms that the current MMT4D path remains structurally limited, especially for a real dominant 3x3 node.

Required future evidence:

```text
spec/source recovery
assembler/parser acceptance
objdump/disassembly proof
board CPU0-3 execution
CPU4/5 negative policy if safe
exact scalar oracle
comparison vs current MMT4D on one real Conv node
no graph integration until semantics and speedup are proven
```

`vmadotn` remains not authorized.

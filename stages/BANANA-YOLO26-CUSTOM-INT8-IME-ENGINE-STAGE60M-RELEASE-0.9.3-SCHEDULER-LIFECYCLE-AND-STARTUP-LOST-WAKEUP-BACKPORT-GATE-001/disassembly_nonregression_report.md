# Disassembly Non-Regression

The frozen 0.9.2 baseline and 0.9.3 candidate were rebuilt with the same
compiler, sysroot, optimization flags, source package, and object order.
Object-level SHA-256 comparison proves byte identity for every release
translation unit containing arithmetic or package semantics:

| Translation unit | Result |
|---|---|
| `stage48_nchwc8_model5.cpp` | byte-identical |
| `stage51_q62_epilogue.cpp` | byte-identical |
| `vmadot_4x4x8_ime.cpp` | byte-identical |
| `int8_v1.cpp` | byte-identical |
| `package_loader.cpp` | byte-identical |
| `stage52_full_executor.cpp` | byte-identical |

`stage49_persistent_slice.cpp` changes only the documented lifecycle mutex,
park/wake acknowledgement, and stale-generation control flow.
`conv_threaded.cpp` changes only readiness mutex acquisition and notification;
it is part of the research compatibility library, not the frozen release
library. `stage52_c_api.cpp` differs only in immutable 0.9.3 build metadata.

The complete no-address RISC-V disassembly, release-object hashes, and dynamic
relocation inventories are retained under the raw evidence root. No Conv,
MatMul, depthwise, Q62/RNE, saturation, LUT, layout, arena, package, profile, or
dispatch opcode body changed.

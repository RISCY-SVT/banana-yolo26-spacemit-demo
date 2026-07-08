# Integer Dot Signedness Family Audit

Current selected real Conv contract:

- Activation storage: signed int8 storage derived from QDQ code by subtracting 128 where required.
- Weight storage: signed int8.
- Current accepted IME path: `smt.vmadot` signed int8 x signed int8 -> int32.
- Zero-point correction remains explicit in the existing Conv path.

Parser/disassembly-visible family members in the current toolchain:

| Family | Parser/disassembly status | Stage31 use |
| --- | --- | --- |
| `smt.vmadot` | accepted | Existing signed path |
| `smt.vmadotu` | accepted | Report-only |
| `smt.vmadotsu` | accepted | Report-only |
| `smt.vmadotus` | accepted | Report-only |
| `smt.vmadot1u` | accepted | Report-only |
| `smt.vmadot1su` | accepted | Report-only |
| `smt.vmadot1us` | accepted | Report-only |
| `smt.vmadot2u` | accepted | Report-only |
| `smt.vmadot2su` | accepted | Report-only |
| `smt.vmadot2us` | accepted | Report-only |
| `smt.vmadot3u` | accepted | Report-only |
| `smt.vmadot3su` | accepted | Report-only |
| `smt.vmadot3us` | accepted | Report-only |

Audit conclusion:

The unsigned/mixed variants may be relevant to future correction-reduction work, especially where activation or weight storage can remain in an unsigned/mixed domain. Stage31 did not prove their independent semantics and did not implement them.

Required future proof:

- Independent scalar oracle per variant.
- Board CPU0-3 execution.
- Exact dtype/zero-point contract for a real node.
- Comparison against current signed path including correction overhead.

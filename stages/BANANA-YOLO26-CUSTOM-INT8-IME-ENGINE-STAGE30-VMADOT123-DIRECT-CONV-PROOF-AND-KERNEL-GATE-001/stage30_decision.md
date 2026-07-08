# Stage30 Decision

classification: `stage30-vmadot123-semantics-proven-but-no-speed-win`

Decision:

- `smt.vmadot1`, `smt.vmadot2`, and `smt.vmadot3` are now semantics-authorized at the micro-instruction level.
- They are not integrated into the custom engine runner in Stage30.
- No real direct Conv speed win was measured in Stage30.

Why no direct Conv integration in Stage30:

- The proven instruction behavior is shifted-tile behavior over an expanded A panel.
- Turning that into a useful direct/sliding 3x3 Conv kernel requires a careful expanded-A-panel packer/scheduler, duplicate-row policy, and benchmark comparison against current threaded MMT4D.
- Doing that inside the proof stage would become a broader kernel rewrite, which the prompt forbids.

Next recommended step:

`BANANA-YOLO26-CUSTOM-INT8-IME-ENGINE-STAGE31-VMADOT123-DIRECT-CONV-INTEGRATION-001`

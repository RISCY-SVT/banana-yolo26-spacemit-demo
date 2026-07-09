# Stage36 Recovered Facts

Stage35:

- Stage34 SIGILL root cause was `rdcycle` in benchmark timing, not `smt.vmadot`.
- Named and raw `smt.vmadot` payloads execute on CPU0; key raw cases pass on CPU1/2/3.
- A5 6-accumulator raw case reached `0.861 ns/vmadot` in the mandatory protocol and `0.625 ns/vmadot` in the supplemental high-iteration diagnostic.
- `smt.vmadot` is fully pipelined in the tested raw independent-accumulator loops; latency is approximately 6 cycles; `>=6` independent accumulators hide latency.
- Stage35 did not integrate `/model.4/cv2/conv/Conv`.

Stage33:

- `smt.vmadotus` `u8xs8` was byte-correct but regressed selected-cut total and remains non-selected.

Stage31/32:

- `smt.vmadot1/2/3` semantics are proven, but the current direct/sliding `3x3` path is rejected due expensive panel/layout.
- `smt.vmadotn` remains not authorized.

Current selected output SHA:

```text
70adcf083e85ea95d5fb42e57ea26f51255944aaa442cde4d919f5ae551b3433
```

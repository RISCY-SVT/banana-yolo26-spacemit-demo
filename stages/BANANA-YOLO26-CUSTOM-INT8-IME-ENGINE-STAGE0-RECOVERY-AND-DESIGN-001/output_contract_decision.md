# Output Contract Decision

## Final Demo Promise

```text
e2e [1,300,6], no NMS
```

The final user-visible YOLO26 demo should return e2e-style detections with no
host NMS dependency in the primary path.

## First Engine Implementation Contract

```text
traditional-first / trunk-first [1,84,8400]
```

Reasoning:

- The e2e model does end at `[1,300,6]`, but its tail contains `ReduceMax`,
  `TopK`, `GatherElements`, `Tile`, `Expand`, multiple `Cast`, and final dynamic
  selection logic.
- The traditional model has the same quantized Conv/MatMul body, but avoids the
  e2e TopK-heavy head.
- Stage 1 is a microkernel stage, not a graph executor stage.

The future full demo can still return e2e-style detections by either:

- implementing the e2e tail after trunk/kernel correctness is proven; or
- using the traditional `[1,84,8400]` output plus a separately approved decode
  and selection layer that matches the e2e contract.

No production FPS or production-readiness claim is made.

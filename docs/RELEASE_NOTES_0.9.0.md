# Release Notes 0.9.0

## Scope

First clean engineering-handoff release of the exact full YOLO26n-640
`K1X_INT8_V1` executor for Banana-Pi BPI-F3 / SpacemiT K1X.

## Release Changes

- Split the minimal installed release library from historical research sources.
- Set project version 0.9.0 and shared-library SOVERSION 1.
- Hide all symbols except the public C ABI.
- Install relocatable CMake and pkg-config metadata plus C consumer examples.
- Freeze selected operator routes into the release; stage environment variables
  are no longer part of the public execution contract.
- Add explicit condition-variable and frame-gated-spin wake policies.
- Add `y26_executor_options_init()` and `y26_status_string()` while preserving
  ABI1 legacy option size.
- Harden state, bounds, overlap, topology, package identity, and same-handle
  concurrency errors.
- Separate determinism, known-fixture, and expected-manifest CLI verification.
- Add exact dual-C4 E2c5 and attention MatMul C8 epilogues.
- Add explicit RVV compact RGB copy.
- Retain measured no-win Softmax-cache and head-bucket candidates as research
  evidence, not release defaults.
- Add reversible, lock-protected O2 lifecycle with stale-state recovery and a
  reviewed IRQ policy.

The final installed low-latency-dedicated O2 surface measured 133305.232 us mean
and 133825.050 us p95 over 500 samples. The separate 13,500-run soak measured
135040.533 us mean and 140242.000 us maximum. Full COCO remained byte-identical.

## Exact Identities

Model, package, prediction, and known-fixture identities are listed in
`HANDOFF_EN.md` and the release manifest.

## Compatibility

The C ABI remains ABI1. Existing options structures at the documented legacy
size remain accepted and default to condition-variable wake. New callers should
initialize the current structure with `y26_executor_options_init()`.

## Classification

`optimized-engineering-handoff-ready`, `reference-ready`, and
`not-production-certified`. This release does not claim 20 FPS.

# Fallback Removal Plan

Fallback removal is staged:

```text
1. Keep ORT CPU fallback for correctness scaffold only.
2. Add one custom block at a time with same-input cut oracle.
3. Replace fallback island only after byte-exact block output and full scaffold output pass.
4. Keep a debug/oracle build that can re-enable ORT cuts for regression testing.
5. Remove ORT from final fast path only after enough custom coverage exists.
```

No fallback removal happened in Stage41.

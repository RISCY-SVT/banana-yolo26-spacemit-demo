# Pure-Model Regression Root Cause

## Finding

The Stage57 to published-Stage58 regression was a release-build contract
regression, not a camera cost and not a changed arithmetic route.

The first Stage58 commit, `e05a4240ab9424ca243a01bd67a140a4bc3dfba4`,
already reproduces the slower 241,652-byte `.text`. All eight Stage58 commits
produce the same `.text` SHA-256 when built through the affected top-level
flow. The measured bisection shows a 6.00% to 6.68% penalty against the
same-session Stage57 control.

The top-level Stage58 cross build omitted:

```text
-mtune=spacemit-x60
-funroll-loops
```

## Controlled Confirmation

A neutral ABI1 `dlopen` benchmark prepared both libraries outside the timing
loop and interleaved 1,000 samples per arm in one O2 window:

| Arm | Mean us | Median us | p95 us | p99 us |
|---|---:|---:|---:|---:|
| Stage57 | 133279.667 | 133094.000 | 134670.100 | 135236.030 |
| Stage58 source rebuilt with accepted flags | 133834.854 | 133601.000 | 135434.950 | 135954.250 |

The rebuilt Stage58 delta is +0.416558%, within the required 1% equivalence
gate. Both arms produced only `0xd43f5e018b415631`, had zero affinity
failures, and executed zero IME operations on CPU4-7.

## Final 0.9.2 Confirmation

After the maintenance source was frozen, the same harness compared the final
0.9.2 library with the Stage57 control using another 1,000 samples per arm.
Total means were 133356.369 us and 134100.921 us respectively, a +0.558%
Stage59 delta. Internal executor means were 130699.600 us and 131422.434 us, a
+0.553% delta. The final result remains inside the prompt's 1% same-session
equivalence gate and below its preferred absolute mean/p95 ceilings. This
closes the published 0.9.1 regression without changing executor arithmetic.

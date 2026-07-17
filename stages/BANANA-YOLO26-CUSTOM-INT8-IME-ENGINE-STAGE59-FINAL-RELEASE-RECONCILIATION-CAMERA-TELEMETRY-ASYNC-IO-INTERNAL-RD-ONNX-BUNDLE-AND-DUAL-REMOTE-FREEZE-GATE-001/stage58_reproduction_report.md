# Stage58 Reproduction

The published 0.9.1 release was rebuilt and exercised before Stage59 changes.
Its ABI1 build-info flags are `0xf`: IME, RVV, the frozen operator profile, and
RGB API are all present. The release healthcheck returned the accepted
`0xd43f5e018b415631` board output hash and the package manifest remained
`fab4a72cf524ce0a205ceca0384144f2eee7bc79dff3f4db8b7208614e8407be`.

The published library reproduced its slower O2 surface at 141116.050 us over
the bounded 20-run scout. A clean Stage58 source build with the accepted X60
flags restored the same arithmetic at 133301.150 us. This confirms that the
regression belonged to release compilation, not model data or arithmetic.
The neutral 1000-per-arm comparison and all eight-commit bisection are reported
separately.

The historical GUI demo was also reproduced at 1280x720 MJPG with 60 requested
and backend-reported FPS. It processed 5.691041 FPS, while OpenCV returned
decoded frames at 9.985989 FPS and the application replaced 43.256314% of its
single-slot queue. These are historical schema semantics, not raw sensor FPS,
driver drops, or sensor-to-display latency.

The 0.9.1 archive verified all `SHA256SUMS` from a clean extraction. Its
embedded source build ID is `bf965e83e650839a95c32027f1a6443d35d44caa`;
the later Stage58 publication/evidence commit is
`5465b68cf41f5547d6455dc229b196c77e590743`. Both identities are preserved
rather than conflated.

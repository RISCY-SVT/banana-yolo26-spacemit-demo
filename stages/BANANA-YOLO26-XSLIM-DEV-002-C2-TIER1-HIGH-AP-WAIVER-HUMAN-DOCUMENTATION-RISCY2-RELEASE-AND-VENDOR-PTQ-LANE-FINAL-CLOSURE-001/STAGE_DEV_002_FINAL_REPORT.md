# DEV-002 Final Report

Classification: `xslim-dev-002-riscy2-human-documentation-release-and-vendor-ptq-closure-complete`

Publication classification: `github-gitlab-release-published-pypi-not-authorized-not-attempted`

Stage: `BANANA-YOLO26-XSLIM-DEV-002-C2-TIER1-HIGH-AP-WAIVER-HUMAN-DOCUMENTATION-RISCY2-RELEASE-AND-VENDOR-PTQ-LANE-FINAL-CLOSURE-001`

## Human disposition

The append-only waiver `BANANA-YOLO26-C2-HIGH-AP-PROFILE-TIER1-WAIVER-001` records C2 as a separate frozen higher-AP profile. The historical universal gate remains failed. B2 remains universal control/rollback. C2 requires an application-specific score threshold because Stage65E proves fewer FP but more FN at score 0.25, IoU 0.50, maxDets 100.

## Upstream and documentation

Upstream `main` remains exact accepted base `9a33f2f...`, tree `05d2c842...`; zero new main commits require integration. Thirteen branch refs, 47 PR records and 11 public releases were inventoried. No semantic upstream change was merged.

Twenty human documents were created and three entry/provenance documents rewritten, including complete English and Russian K1X/YOLO26 paths. Clean-source validation covered 48 Markdown files, 128 links and 164 snippets: 106 parsed/compiled, 58 intentional fragments, zero failures.

## Quality and neutrality

Final XSlim pytest: 212 cases plus 65 subtests, zero failures/errors/skips and zero uncaptured warnings. Ruff, compileall, both shell scripts, 11-module isolated strict mypy, 28 Banana tooling tests, CLI/config smoke, twine, wheel contents and SPDX validation pass.

Whole-tree strict mypy remains honestly reported at 2877 errors in 94 inherited files. REUSE per-file headers remain incomplete, while bad/missing/deprecated license-file counts are zero and aggregate Apache/provenance/notices/SBOM pass. No quantization-semantic source changed, so accepted no-override neutrality evidence remains applicable; no forbidden PTQ regeneration was performed.

## Release

XSlim advanced on the existing branch from `46d5d36...` to `80204be22906962c82879112014f255828f69c64`, tree `535f0b4d...`, version `2.1.2+riscy.2`. Annotated tag object `30ab31f2...` peels to that commit on local/GitHub/GitLab.

Two clean builds produced raw byte-identical wheel, sdist, source/docs archives, SPDX SBOM, manifest and checksum files. Clean wheel and sdist environments passed dependency install, import, exact version, all CLI helps, config smoke, `pip check` and uninstall. Eight downloaded release assets and release notes are byte-identical across GitHub and GitLab. GitHub Actions remain disabled; no target version exists on PyPI.

## Closure

Vendor PTQ, provider-numerics and current fusion lanes are closed. Full BRECQ/QDrop remain deferred, not active. The next recorded direction is separately authorized model/executor co-design with symmetric-S8 `K1X_INT8_V2` only as a hypothesis; this Stage did not execute it.

Protected Banana main, custom executor and `/data/ncnn` are unchanged. No board command, model generation, camera work, model publication or eMMC write occurred.

Raw evidence: `/data/k1x-stage-runs/BANANA-YOLO26-XSLIM-DEV-002-C2-TIER1-HIGH-AP-WAIVER-HUMAN-DOCUMENTATION-RISCY2-RELEASE-AND-VENDOR-PTQ-LANE-FINAL-CLOSURE-001`.

Shared log: `/data/ncnn-logs/ai-team/2026-08-27/2026-08-27_10-39-26__contcodex__BANANA-YOLO26-XSLIM-DEV-002-C2-TIER1-HIGH-AP-WAIVER-HUMAN-DOCUMENTATION-RISCY2-RELEASE-AND-VENDOR-PTQ-LANE-FINAL-CLOSURE-001__dev002-release-closure`.

The exact final Banana metadata-attestation commit is recorded in the result packet post-push attestation because a tracked file cannot contain its own commit hash.

Timestamp: `2026-08-27T12:11:48Z`.

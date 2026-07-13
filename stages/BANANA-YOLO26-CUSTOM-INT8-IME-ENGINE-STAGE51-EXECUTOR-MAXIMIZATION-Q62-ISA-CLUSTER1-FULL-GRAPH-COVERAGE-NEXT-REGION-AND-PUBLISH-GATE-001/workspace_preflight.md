# Workspace preflight

- Expected and observed start HEAD: `ea993fb4255f12592380b975bd3cc6dbc73bea57`.
- Branch: `yolo26-custom-int8-engine`.
- Initial worktree: clean.
- `git diff --check` and `git diff --cached --check`: pass.
- Repository history was neither reset, rebased, squashed, nor rewritten.

Bounded raw command output:

```text
[dev] glibc: ldd (Ubuntu GLIBC 2.39-0ubuntu8.7) 2.39
[dev] node : /usr/bin/node  v24.15.0 | npm: 12.0.0
[dev] codex: /home/svt/.npm-global/bin/codex
[dev] claude: /home/svt/.npm-global/bin/claude
[dev] riscv default: /opt/riscv -> /opt/SpacemiT
## AGENTS files
../ncnn/AGENTS.md
../banana-yolo11-spacemit-demo/AGENTS.md

## status
## yolo26-custom-int8-engine

## head
ea993fb4255f12592380b975bd3cc6dbc73bea57

## branch
yolo26-custom-int8-engine

## graph
* ea993fb (HEAD -> yolo26-custom-int8-engine) Close the persistent NCHWc8 model4-to-model8 slice on K1X
* eb8e281 Prove the persistent NCHWc8 integer slice on K1X
* b163c83 Define K1X INT8 semantics and test direct NCHWc8 delivery
* 3c1eabb Measure and gate the K1X resident INT8 AOT executor
* da213f6 Revalidate YOLO26 INT8 with SpacemiT ORT 2.0.5
* 860ee58 Recalibrate YOLO26 K1X system strategy and model co-design path
* bdefd89 Repair and evaluate YOLO26 model5 stride2 dataflow
* f363c84 Checkpoint exact YOLO26 Stage43 model5 evidence
* 7a9b679 Repair YOLO26 in-process ORT contract and add model16 oracle
* 6559e2a Add YOLO26 full-model skeleton gate
* 57ad1bf Repair YOLO26 model4 branch3x3 im2col pack path
* 11675cc Gate YOLO26 model4 post-3x3 bottleneck repair
* 97d9e8c Gate YOLO26 model4 post-cv2 bottleneck after pipelined vmadot
* a945d60 Select YOLO26 cv2 pipelined vmadot sidecar
* a8b7607 Repair YOLO26 vmadot SIGILL diagnostic emission
* 71a89f4 Probe YOLO26 cv2 software-pipelined vmadot microkernel
* 6c64bdd Gate YOLO26 MMT4D mixed signedness correction
* 1304c76 Decide YOLO26 MMT4D mainline after sliding layout proof
* 00aa667 Gate YOLO26 vmadot123 direct Conv sidecar
* 921c1d7 Prove vmadot123 semantics for YOLO26 direct Conv lane
* 8c3d647 Add YOLO26 vendor ORT rt204 COCO baseline
* b54c876 (gitlab-rd/yolo26-custom-int8-engine, github/yolo26-custom-int8-engine) Repair YOLO26 model4 Conv correction writeback
* 502f7ab Select YOLO26 model4 Conv tile prepack lane
* 6a32c90 Repair YOLO26 model4 activation requant after Conv threading
* b382bd7 Select YOLO26 model4 Conv threading lane
* e3bbacb Repair YOLO26 model4 merge dataflow after ONNX cut proof
* fce411e Close model4 ONNX cut runner API and repair output quantization
* 8350c57 Close model4 C2f ONNX cut oracle gate
* d802598 Integrate model4 C2f merge repair into YOLO26 custom INT8 runner
* 6ea3f07 Repair model4 fullshape merge dataflow for YOLO26 custom INT8 engine
* 53ac15a Integrate threaded model4 C2f sidecar for YOLO26 custom INT8 engine
* 6c4c825 Integrate cluster0 threaded Conv sidecar for YOLO26 custom INT8 engine
* 92e7d87 Measure YOLO26 custom INT8 Conv IME roofline and cluster0 threading feasibility
* 3ca9499 Complete model4 C2f gate for YOLO26 custom INT8 engine
* 14d0e74 Extend YOLO26 custom INT8 coverage into model4 branch entry
* 5cc0905 Expand YOLO26 custom INT8 C2f coverage after merge repair
* 9219f89 Repair YOLO26 C2f merge dataflow
* cae5301 Complete YOLO26 C2f residual concat block
* 4cf60ff Expand YOLO26 custom INT8 branch block subset
* 56a612b Expand YOLO26 custom INT8 backbone subset after RVV activation gate

## remotes
github	git@github.com:RISCY-SVT/banana-yolo26-spacemit-demo.git (fetch)
github	git@github.com:RISCY-SVT/banana-yolo26-spacemit-demo.git (push)
gitlab-rd	git@gitlab.itglobal.com:riscy/sw/banana-yolo26-spacemit-demo.git (fetch)
gitlab-rd	git@gitlab.itglobal.com:riscy/sw/banana-yolo26-spacemit-demo.git (push)
template-yolo11-gitlab	git@gitlab.itglobal.com:riscy/sw/banana-yolo11-spacemit-demo.git (fetch)
template-yolo11-gitlab	git@gitlab.itglobal.com:riscy/sw/banana-yolo11-spacemit-demo.git (push)

## diff check

## cached diff check

## last commits fuller
commit ea993fb4255f12592380b975bd3cc6dbc73bea57
Author:     Sergio <svt@zmail.ru>
AuthorDate: Mon Jul 13 09:32:04 2026 +0200
Commit:     Sergio <svt@zmail.ru>
CommitDate: Mon Jul 13 09:32:04 2026 +0200

    Close the persistent NCHWc8 model4-to-model8 slice on K1X

commit eb8e28194853d8f70c8a8a8d008253396327aac1
Author:     Sergio <svt@zmail.ru>
AuthorDate: Sun Jul 12 22:01:14 2026 +0200
Commit:     Sergio <svt@zmail.ru>
CommitDate: Sun Jul 12 22:01:14 2026 +0200

    Prove the persistent NCHWc8 integer slice on K1X

commit b163c83f7dc1677c8b31b9a2cc75e227d5992b0d
Author:     Sergio <svt@zmail.ru>
AuthorDate: Sun Jul 12 17:34:05 2026 +0200
Commit:     Sergio <svt@zmail.ru>
CommitDate: Sun Jul 12 17:34:05 2026 +0200

    Define K1X INT8 semantics and test direct NCHWc8 delivery
```

# Remote preflight

The configured GitHub remote is `git@github.com:RISCY-SVT/banana-yolo26-spacemit-demo.git` under
the local name `github`; no `origin` remote is configured. Direct `git ls-remote` was used as the
authority. The remote Stage51 branch was an ancestor of local HEAD, so only a normal
fast-forward push was permitted.

```text
[dev] glibc: ldd (Ubuntu GLIBC 2.39-0ubuntu8.7) 2.39
[dev] node : /usr/bin/node  v24.15.0 | npm: 12.0.0
[dev] codex: /home/svt/.npm-global/bin/codex
[dev] claude: /home/svt/.npm-global/bin/claude
[dev] riscv default: /opt/riscv -> /opt/SpacemiT

## target branch
b54c8767e691dc57cbd035a13d2d2d348d2f5366	refs/heads/yolo26-custom-int8-engine

## remote HEAD
ref: refs/heads/yolo26-rd-bootstrap	HEAD
9c307f8a2d2fed5f39375ebacb0dbc92b59a0510	HEAD

## fetched refs
b54c8767e691dc57cbd035a13d2d2d348d2f5366 refs/remotes/github/yolo26-custom-int8-engine
```

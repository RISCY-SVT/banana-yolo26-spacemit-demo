# Template Source

This repository was created as an isolated YOLO26 R&D workspace from the frozen
YOLO11 production release.

## Source Repository

- GitHub: `git@github.com:RISCY-SVT/banana-yolo11-spacemit-demo.git`
- GitLab: `git@gitlab.itglobal.com:riscy/sw/banana-yolo11-spacemit-demo.git`
- Template tag: `production-2026-07-02`
- Template commit: `9c0933be58ee122389d1a43f45f81e80655d6904`

## Isolation Policy

- The production YOLO11 repository must remain read-only for YOLO26 R&D work.
- The inherited remote in this repository is renamed to
  `template-yolo11-gitlab`.
- No `origin` remote is configured until a dedicated YOLO26 remote is explicitly
  authorized.
- Runtime/model policy changes must stay in this YOLO26 R&D repository only.

## Initial R&D Scope

- SpacemiT ONNX Runtime 2.0.4 release and provider-option inventory.
- K1/K3/X60/X100/A100 architecture-selection forensics.
- YOLO26n export and decode contract validation.
- CPU oracle comparison before SpaceMIT EP acceleration.
- INT8 feasibility only after the float/CPU path is semantically correct.

# BANANA-YOLO26-CUSTOM-INT8-IME-ENGINE-STAGE41-POST-MODEL4-BLOCK-PROFILING-AND-FIRST-EXPANSION-GATE-001

## Mission

Continue from Stage40 full-model skeleton correctness. Split the suffix region after `/model.4` into block-level ONNX CPU cuts, rank the next high-value blocks, and integrate or gate exactly one next custom block if its oracle contract is clean.

Expected start head should be the Stage40 final commit.

## Hard Boundaries

Do not implement a full YOLO26 engine, graph-wide scheduler, camera/full-image path, COCO/mAP, production/default backend, XSlim, new ISA lanes, `vmadotn`, `vfmadot`, CPU4-7 IME, or all-core OpenMP dispatch.

## Required Work

1. Replay Stage40 skeleton:
   - all-ORT fallback final `output0` exact vs full ORT CPU;
   - custom `/model.4` insertion exact vs full ORT CPU final output.
2. Build block-level suffix cuts starting at:
   - `/model.5/conv/Conv_output_0_QuantizeLinear_Output`;
   - `/model.6/...` block boundaries;
   - later suffix regions if cut extraction is cheap.
3. Rank blocks by ORT CPU cut time, tensor handoff size, Conv/QDQ density, and availability of existing custom runner patterns.
4. Select exactly one target:
   - likely `/model.5/conv/Conv` if it is locally clean and material;
   - otherwise the highest-value clean block from measured suffix cuts.
5. If integrating a custom block, keep it explicit/local and prove:
   - same-input boundary exactness;
   - final skeleton `output0` exactness;
   - no default backend switch;
   - no full-model FPS claim.

## Required Reports

Create:

```text
STAGE41_FINAL_REPORT.md
STAGE41_SUMMARY_RU.md
stage40_replay_report.md
suffix_block_cut_inventory.md
suffix_block_profile.tsv
target_block_selection.md
selected_block_contract.md
selected_block_correctness_report.md
full_skeleton_output_comparison_report.md
stage42_prompt.md
```

## Acceptance

Use one of:

```text
stage41-next-block-selected-and-correct
stage41-suffix-profile-complete-expansion-deferred
stage41-blocked-suffix-cut-contract
stage41-blocked-custom-block-correctness
```

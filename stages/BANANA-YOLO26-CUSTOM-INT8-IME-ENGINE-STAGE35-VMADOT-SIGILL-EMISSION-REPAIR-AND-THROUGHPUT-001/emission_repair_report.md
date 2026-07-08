# Emission Repair Report

stage_id: `BANANA-YOLO26-CUSTOM-INT8-IME-ENGINE-STAGE35-VMADOT-SIGILL-EMISSION-REPAIR-AND-THROUGHPUT-001`

## Repair

The Stage34 throughput diagnostic used direct `rdcycle` timing. On this board/session, that instruction trapped before the Stage35 payload could prove or disprove `smt.vmadot` execution.

Stage35 changed the diagnostic timing path to:

```text
std::chrono::steady_clock
```

and preserved payload emission:

```text
named smt.vmadot v28,v0,v1
raw .word 0xe2103e2b
standalone named/raw helper-shaped payloads
independent raw accumulator groups
```

## Evidence

Before repair:

```text
faulting_insn32_hex: 0xc0002773 / 0xc0002873
classification: SIGILL-at-rdcycle
```

After repair:

```text
rdcycle_in_objdump: absent
smt.vmadot_in_objdump: present
raw_0xe2103e2b_in_objdump: present
CPU0_exact_single_wrapper_shape_named: pass
CPU0_exact_single_wrapper_shape_raw_same_as_helper: pass
CPU0_standalone_named: pass
CPU0_standalone_raw: pass
CPU1_2_3_key_smoke: pass
```

## Final Emission Route Decision

For Stage35 throughput measurement, raw/proven `.word` routes were used for independent accumulator patterns because they directly preserve the helper word and avoid any ambiguity from future mnemonic parsing changes.

Named `smt.vmadot` is not rejected by Stage35:

```text
named_encoding_status: pass for tested shapes
raw_same_as_helper_status: pass
recommended_benchmark_substrate: raw/proven helper-word emission for microbench, named helper path remains accepted for existing engine helper
```

## Non-Claims

This report does not claim full YOLO26 inference, model FPS, camera/full-image speed, COCO/mAP, production readiness, default backend readiness, vmadotn support, or vfmadot support.

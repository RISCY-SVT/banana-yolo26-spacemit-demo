# VMADOT Encoding Comparison

stage_id: `BANANA-YOLO26-CUSTOM-INT8-IME-ENGINE-STAGE35-VMADOT-SIGILL-EMISSION-REPAIR-AND-THROUGHPUT-001`

## Objects Compared

```text
working_helper_objdump: artifacts/objdump_working_helper.txt
stage35_before_chrono_objdump: artifacts/objdump_stage35_bench.txt
stage35_after_repair_objdump: artifacts/objdump_stage35_bench_after_a5.txt
```

## Working Helper

The accepted helper path contains:

```text
mnemonic: smt.vmadot v28,v0,v1
instruction_word: 0xe2103e2b
vd: v28
vs1: v0
vs2: v1
vd_even: yes
destination_pair: v28/v29
input_overlap: no
```

The surrounding sequence uses:

```text
accumulator_view: vsetvli t0, zero, e32, m2
input_view:       vsetvli t0, zero, e8, m1
load_A:           vle8.v v0
load_B:           vle8.v v1
store_C:          vse32.v v28
```

## Stage35 Named / Raw Comparison

The repaired Stage35 tool emits the same core word for the exact helper-shaped cases:

```text
case1_named: smt.vmadot v28,v0,v1 -> 0xe2103e2b
case2_raw:   .word 0xe2103e2b
case4_standalone_named: smt.vmadot v28,v0,v1 -> 0xe2103e2b
case5_standalone_raw:   .word 0xe2103e2b
```

No `rdcycle` instruction remains in the repaired Stage35 binary:

```text
rdcycle_after_repair: absent
```

## Register Shapes

CPU0 post-repair board status:

```text
v28/v29 single accumulator: pass
v24/v25 single accumulator: pass
v20/v21 single accumulator: pass
v28/v29 + v30/v31 two accumulators: pass
v20/v21 + v22/v23 + v24/v25 + v26/v27 four accumulators: pass
v16/v17 + v18/v19 + v20/v21 + v22/v23 + v24/v25 + v26/v27 six accumulators: pass
```

## Root Cause Classification

```text
named_asm_encoding_diff: no
raw_encoding_diff: no
vtype_or_AVL_diff: no evidence after repair
inline_asm_context_or_clobber: no evidence after repair
measurement_path_SIGILL: yes
```

The Stage34 trap was caused by benchmark measurement emission, not by a different `smt.vmadot` instruction word.

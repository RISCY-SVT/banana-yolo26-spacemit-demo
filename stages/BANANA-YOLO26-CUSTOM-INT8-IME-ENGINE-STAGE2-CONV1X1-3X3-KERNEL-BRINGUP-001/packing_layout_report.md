# Packing Layout Report

## MMT4D Tile Contract

| tile | shape | layout | tail behavior |
| --- | --- | --- | --- |
| A | `4x8 int8` | row-major, K-contiguous, `A[m*8+k]` | zero-pad invalid M/K |
| B | `4x8 int8` | output-channel-major transposed NK, `B[n*8+k]` | zero-pad invalid N/K |
| C | `4x4 int32` | row-major, `C[m*4+n]` | store valid M/N only |

This matches the Stage 1 `smt.vmadot 4x4x8 s8xs8->s32` contract.

## Tests

- `test_pack_a_layout`: validates legacy Stage 0 4x8 pack and new MMT4D tail zero-padding.
- `test_pack_b_layout`: validates legacy Stage 0 transposed NK pack and new MMT4D tail zero-padding.
- Host CTest: 14/14 pass.
- Board Conv fixtures: tails for `Cin` and `Cout` pass with mismatches 0.

## Current Cost Caveat

Stage 2 packing is deliberately simple and done inside the Conv wrappers. Board microbench shows that raw MMT4D tile execution is faster than scalar, but packing-included Conv wrappers are not yet faster. Stage 3 should add B prepacking, A block reuse, and larger output-channel/spatial blocks.

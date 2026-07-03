# smt.vmadot 4x4x8 Semantics

Stage 1 implements exactly one model-independent microkernel contract:

```cpp
void y26_vmadot_4x4x8_scalar_s8s8s32(
    const int8_t* a_4x8_row_major,
    const int8_t* b_4x8_transposed_nk,
    int32_t* c_4x4_row_major,
    bool accumulate);

bool y26_vmadot_4x4x8_ime_available_buildtime();

int y26_vmadot_4x4x8_ime_s8s8s32(
    const int8_t* a_4x8_row_major,
    const int8_t* b_4x8_transposed_nk,
    int32_t* c_4x4_row_major,
    bool accumulate);
```

## Layout

- A: `4x8 int8` row-major, index `A[m * 8 + k]`.
- B: `4x8 int8` transposed output-major, index `B[n * 8 + k]`.
- C: `4x4 int32` row-major, index `C[m * 4 + n]`.

## Operation

```text
if accumulate == false:
  C[m,n] = 0

C[m,n] += sum_{k=0..7} int32(A[m,k]) * int32(B[n,k])
```

Signedness is strictly signed x signed. The microkernel does not apply activation zero-point, weight zero-point, output zero-point, scaling, rounding, or clamp.

## Status codes

```text
0 = success
1 = not built with IME
2 = runtime affinity/safety check failed
3 = SIGILL caught
4 = invalid argument
```

## Safety boundary

The public IME entry point checks build support, null pointers, current CPU cluster0 affinity, and catches SIGILL. The direct asm hot loop in `bench_vmadot_microkernel.cpp` is benchmark-only and is guarded by a one-shot SIGILL smoke before timing. It is not an engine API.

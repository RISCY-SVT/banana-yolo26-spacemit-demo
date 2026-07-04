# Activation LUT Candidate Report

## Candidate A0

Mode: `scalar_float_reference`

| metric | us |
|---|---:|
| selected-subset IME total | 620735 |
| activation total | 465901 |
| Act0 | 311321 |
| Act1 | 154580 |
| mismatches | 0 |

## Candidate A1

Mode: `fixed_requant_only`

| metric | us |
|---|---:|
| selected-subset IME total | 516970 |
| activation total | 361666 |
| Act0 | 241527 |
| Act1 | 120140 |
| mismatches | 0 |

This is a diagnostic path. It was not selected because A2 is faster and directly removes SiLU float work.

## Candidate A2

Mode: `int8_lut`

| metric | us |
|---|---:|
| selected-subset IME total | 350092 |
| activation total | 192568 |
| Act0 | 128644 |
| Act1 | 63924.9 |
| mismatches | 0 |
| speedup vs A0 IME total | 1.77307x |
| speedup vs scalar total | 3.58374x |

A2 is the selected Stage 8 mode.

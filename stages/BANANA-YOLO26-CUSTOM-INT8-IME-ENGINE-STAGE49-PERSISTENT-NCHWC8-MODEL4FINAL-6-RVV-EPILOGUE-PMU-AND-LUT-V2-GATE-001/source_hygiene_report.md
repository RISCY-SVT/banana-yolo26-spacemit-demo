# Source hygiene report

The intended Git payload is limited to Stage49 engine/library code, focused tools/tests, reports, and the Stage50 prompt. Build trees, model/package binaries, tensor dumps, board logs, PMU source/build trees, datasets, credentials, and `/data/.codex` are excluded.

Final checks:

- symlink scan: no matches (`0275_symlink-scan`)
- files larger than 10 MiB: no matches (`0276_large-file-scan`)
- Stage49 payload-only secret scan: no matches (`0281_payload-secret-scan`)
- private-path scan: no sensitive path matches (`0278_private-path-scan`)
- all 61 required report files exist and are non-empty (`0272_required-report-audit`)

The first indexed whitespace check (`0284_final-cached-diff-check`) exposed trailing spaces in captured `lsblk` text and empty trailing TSV fields. The generator now strips text-line padding and represents trailing empty TSV fields explicitly as `NA`; the failed check remains in raw evidence and is superseded by the final indexed check.

The earlier broad secret scan matched ordinary `token` variable names in three unrelated, already tracked oracle extraction scripts. Those files are not in the Stage49 payload; the payload-only scan is authoritative.

`/data/ncnn` was not modified. Its final HEAD remains `a245a70c641a1f20f357c65d103e5f9e50fe84a1`, and its three pre-existing dirty file hashes remain:

- `convolution_1x1_int8_xsmtvdot.S`: `b50bd3355fea15adc142d7ae70e8916ef78781563d1aeccdf500d014f91c5229`
- `convolution_1x1_int8_xsmtvdot.cpp`: `f36b8d1a40ac905ba744edf9c575b64b0ff109a15cf8418af98ca115514acdb6`
- `convolution_1x1_int8_xsmtvdot.h`: `fe136f6470d52d3ad1cf580a8a4d53393c669b1191a19ae3600ff6df118ce359`

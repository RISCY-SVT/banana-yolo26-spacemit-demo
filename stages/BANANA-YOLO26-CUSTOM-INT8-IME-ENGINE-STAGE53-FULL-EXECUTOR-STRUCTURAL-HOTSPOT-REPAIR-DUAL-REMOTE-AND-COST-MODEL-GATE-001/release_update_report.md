# Release update

The Stage52 functional-reference bundle remains unchanged. The Stage53 optimized research bundle is `/data/releases/banana-yolo26-k1x-int8-executor/stage53-optimized-research`.

The updated bundle is not a production release. It preserves API/CLI compatibility, records the condition-variable compatibility mode and epoch-spin research mode, and contains no COCO dataset or raw private logs.

The release build installs the selected statically linked executor CLI while
also shipping the static and shared public libraries. This prevents the shared
install pass from replacing the selected CLI with the slower dynamically linked
benchmark route. The deployed static CLI measured 239025 us mean and 241582 us
p95 under the 10/100/5 epoch-spin protocol, with output hash
`0xd43f5e018b415631` and zero CPU4-7 IME execution.

```text
source commit:
  534881509feb2a34e5ce7aa33b6ceaf9580d2224

package manifest SHA-256:
  fab4a72cf524ce0a205ceca0384144f2eee7bc79dff3f4db8b7208614e8407be

release tree manifest SHA-256:
  b394f4b13fb0775c46e30e96fa550687961ff20ade4ac9c9275d2f49992b4991

release_sha256.txt SHA-256:
  ddafb7c2be2d746686f4061e851efcc20ccc36c6a9bef4e5838eddac2faa2f08

executor CLI SHA-256:
  729fa9ed7728fed32d9a60e0e07f5a098fc849946c270f133a6713a9052fe021

static library SHA-256:
  d6b6a8faaf1e97fcd0b4eddd64fe442763bc5f82942f2fc3f4e18c8dc228680a

shared library SHA-256:
  0ee6caf7bbd67c5be145a800a7add53f1bd6b8ec6e6e159024642ed3b25db87f
```

Two independent bundle generations produced identical manifests and payload
hash inventories. The deployed bundle passed its complete checksum inventory,
the C ABI smoke test, and the CLI smoke test.

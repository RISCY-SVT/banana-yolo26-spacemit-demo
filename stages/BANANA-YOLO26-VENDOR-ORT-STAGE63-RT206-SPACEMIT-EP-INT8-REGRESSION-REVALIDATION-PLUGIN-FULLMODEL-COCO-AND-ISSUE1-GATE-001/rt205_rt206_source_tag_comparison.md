# Public source tag comparison

GitHub resolves public tags `2.0.5` and `2.0.6` to the same commit:

```text
61e7fc2319cd16aa5487fd1155dc15c5390c8a90
```

Both public tag trees contain `VERSION_NUMBER` value
`1.24.2+spacemit.a1`. Therefore the public source tags cannot explain the
binary delta. Stage63 treats release-asset files, hashes, symbols, embedded
strings, and board behavior as primary evidence and makes no source-level
provenance claim for the changed closed provider binary.

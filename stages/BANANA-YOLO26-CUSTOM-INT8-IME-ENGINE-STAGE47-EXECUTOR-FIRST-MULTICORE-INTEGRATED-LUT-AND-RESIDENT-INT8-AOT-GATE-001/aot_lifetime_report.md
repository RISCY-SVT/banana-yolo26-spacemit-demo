
# AOT lifetime report

`aot_tensor_arena_manifest.tsv` records each tensor's producer/last consumer,
64-byte-aligned offset, and bytes. The linear-scan allocator reuses dead buffers;
the slice requires 1,638,400 arena bytes instead of allocating every logical
tensor independently. Packed weights live outside the arena for executor life.

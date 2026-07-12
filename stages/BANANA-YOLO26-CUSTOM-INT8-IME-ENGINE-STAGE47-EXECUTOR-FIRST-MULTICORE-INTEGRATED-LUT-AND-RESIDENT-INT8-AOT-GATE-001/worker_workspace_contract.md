
# Worker workspace contract

One persistent pool owns up to four CPU0-3 workers. Each worker has only its A
panel, accumulator tile, output tile, and synchronization state. Static spatial
partitioning won the bounded comparison; output-channel partitioning was slower.
CPU4-7 never execute IME instructions.

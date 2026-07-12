
# NCHWc8 direct-load candidate report

The byte-order proof covers border (`m_begin=0`), interior (`4`), and row-edge
(`40`) tiles. All C8-u64, `vlse64`, and `vlseg2e64` panels are byte-identical.
Disassembly confirms the intended `vlse64.v`, `vlseg2e64.v`, and existing
`smt.vmadot` instructions.

The selected strategy is `rvv_vlseg2e64_c8x4`. Its same-mode scout mean was
`6443.637798 us`; the final full candidate matrix measured
`6516.213018 us`. Interior delivery uses C8-group vector loads and
bounded worker-local A tiles. The generic Stage47 per-byte interior `pack_a`
loop and full im2col materialization are absent. Borders use exact semantic
zero-point C8 chunks.

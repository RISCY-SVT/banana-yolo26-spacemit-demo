
# Integrated kernel report

All nine deterministic full-shape graph representatives were exact on board for
M4, M8, and M12, including complete M12 tails. On model5, the valid sequential
M12 results scale from `95943.333086 us` on one worker to
`26388.005044 us` on four workers
(`90.896728%` parallel efficiency).

The selected full-shape rates vary from `0.240191`
to `13.351728 GMAC/s`; therefore the Stage45 CPU0 prepacked-compute M12
rate is not used as a graph-wide rate. Commands 0217-0222 are preserved but
rejected because six benchmarks ran concurrently.

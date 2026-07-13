# LUT-v2 report

M4/M8/M12, 1-4 workers, and spatial/output-channel partitions were measured for representative model7/model8 classes. M12xN16 with exact tail, P3 where spatial gather is needed, E1, four CPU0-3 workers, and spatial partition remains selected. Spatial partition materially beat output-channel partition in every tested representative. Stable 10/100/5 rows cover 1x1, 3x3 stride1, 3x3 stride2, and small-N classes used in the slice.

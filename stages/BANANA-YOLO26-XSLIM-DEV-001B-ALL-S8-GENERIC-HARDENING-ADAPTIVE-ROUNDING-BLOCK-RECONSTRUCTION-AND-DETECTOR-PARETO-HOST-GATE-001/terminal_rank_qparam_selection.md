# Terminal rank qparam selection

Selection used no COCO labels. For each P3/P4/P5 confidence domain, the lexicographic decision first maximized exact common-tail TopK overlap with the FP32 teacher, then minimized teacher-top-2K pair inversions, threshold crossings, reconstruction error, clipping and rail occupancy.

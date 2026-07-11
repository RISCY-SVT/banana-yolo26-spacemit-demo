# Student 416/512 disposition

Both untrained hypotheses remain in the next decision packet. The 416 candidate
is latency-primary; the 512 candidate is the accuracy fallback and may become
primary only after measured runtime and trained accuracy justify it. Stage46
does not train, select, or claim either candidate. Resolution selection must use
trained COCO accuracy, measured K1X latency, cache/tail behavior, head scale
count, and the now-negative RT205/plugin evidence rather than quadratic FLOP
scaling alone.

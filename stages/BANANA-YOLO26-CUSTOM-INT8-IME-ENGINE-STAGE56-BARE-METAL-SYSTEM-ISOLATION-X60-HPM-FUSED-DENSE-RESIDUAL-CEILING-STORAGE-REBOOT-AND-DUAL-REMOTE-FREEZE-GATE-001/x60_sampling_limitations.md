# X60 sampling limitations

Selection uses same-process counting. Hardware overflow sampling was not claimed because the current PMU/perf surface did not provide a verified X60 overflow map for every event. Tracepoint correlation, not cycles sampling, was used for OS-noise diagnosis.

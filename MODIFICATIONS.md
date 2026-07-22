# Modifications

This repository implements a K1X-specific exact INT8 executor and camera/release
tooling around a YOLO26 model export. Its Git history preserves the progression
through the frozen R640 executor, scheduler maintenance 0.9.3, Stage60's Q0
resolution sweep, and Stage61's exact attention N-tail/R768 work.

Stage62 merges the accepted maintenance and research histories without squash or
rebase. It adds no numerical kernel, qparam, model, or performance research. Its
source changes are limited to integration policy, an overflow-safe overlap
helper, tests, build/release metadata, license/source notices, packaging, and
human documentation.

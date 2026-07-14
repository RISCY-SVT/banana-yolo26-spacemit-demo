# Stage51 estimator errata

Stage51 representative MAC coverage was not optimized wall-time coverage. The
158.973694 ms optimistic, 204.380817 ms central, and 269.869364 ms conservative
envelopes are superseded as full-model predictors by the measured Stage52 and
Stage53 complete-executor surfaces.

The error arose because newly completed correctness-first operators and
materialized graph boundaries were mapped as if representative dense-kernel
coverage also represented their optimized wall time. Stage53 replaces that
method with one measured row per selected operation/range plus measured
schedule and unaccounted wall.

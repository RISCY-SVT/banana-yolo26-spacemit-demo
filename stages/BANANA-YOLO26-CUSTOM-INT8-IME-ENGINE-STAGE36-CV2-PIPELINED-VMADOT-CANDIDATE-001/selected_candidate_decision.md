# Selected Candidate Decision

Decision: select `A1_branch1_add_lut_cv2_pipelined4`.

Why:

- Correctness is exact: mismatches=0, max_abs_diff=0, output SHA unchanged.
- FRM sweep passes.
- Board CPU0 single-thread correctness passes.
- Board CPU0-3 stable benchmark passes.
- `model4_cv2_compute_us` speedup is 2.085580x.
- selected-cut total speedup is 1.124979x.
- A2 6-accumulator path is correct but slightly slower in total and in `model4_cv2_compute_us`.

Rejected or deferred:

- `A2_branch1_add_lut_cv2_pipelined6`: correct, not selected because A1 is faster in the same session.
- `smt.vmadotus`: explicitly not selected in Stage36; Stage33 already proved it correct but regressed for the current signed-storage pipeline.
- `smt.vmadot1/2/3`: not integrated; direct/sliding panel path remains rejected by Stage31/32 evidence.
- `vmadotn`: not authorized.

Next bottleneck:

After A1, selected-cut shares are:

- conv: 56.9509%
- output_quantize: 19.8476%
- activation/requant: 8.79814%
- merge: 6.48432%

Conv remains the largest bucket, but `/model.4/cv2/conv/Conv` compute was reduced substantially. Stage37 should rebuild per-Conv attribution and choose exactly one local lane.

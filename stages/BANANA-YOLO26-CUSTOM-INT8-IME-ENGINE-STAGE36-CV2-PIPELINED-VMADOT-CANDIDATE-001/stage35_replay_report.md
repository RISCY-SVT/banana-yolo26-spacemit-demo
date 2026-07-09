# Stage35 Replay Report

Purpose: confirm that Stage35 `smt.vmadot` SIGILL emission repair remains valid before attaching a real `/model.4/cv2/conv/Conv` candidate.

Board protocol:

- CPU0 full replay for A1/A4/A5
- CPU1/2/3 smoke for A5
- no CPU4-7 IME execution

| case | cpu | status | mismatches | trap | ns_per_vmadot | note |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| A1_raw_single_acc_dependent_chain | 0 | 0 | 0 | 0 | 6.00000 | dependent chain |
| A4_raw_independent_4_accumulators | 0 | 0 | 0 | 0 | 1.56250 | 4 accumulator groups |
| A5_raw_independent_6_accumulators | 0 | 0 | 0 | 0 | 1.01333 | 6 accumulator groups |
| A5_raw_independent_6_accumulators | 1 | 0 | 0 | 0 | 5.55000 | CPU smoke |
| A5_raw_independent_6_accumulators | 2 | 0 | 0 | 0 | 6.95000 | CPU smoke |
| A5_raw_independent_6_accumulators | 3 | 0 | 0 | 0 | 5.55000 | CPU smoke |

Conclusion: Stage35 repair remains board-executable on CPU0 and the key A5 raw helper-shaped case remains executable on CPU1/2/3. This replay is microbench evidence only, not selected-cut timing.

Raw evidence:

- `/data/ncnn-logs/ai-team/2026-07-09_05-29-08/BANANA-YOLO26-CUSTOM-INT8-IME-ENGINE-STAGE36-CV2-PIPELINED-VMADOT-CANDIDATE-001/run_logs/stage35_replay_board.log`

# Next Target Strategy

## Corrected board evidence

The custom scaffold mean is `826008.582826 us`. Its largest measured regions are:

- suffix after model4: `554052.102166 us` (`67.08%`).
- prefix through model4 input: `229662.287042 us` (`27.80%`).
- custom model4 plus adapters: `42280.804170 us` (`5.12%`).

Stage41 cumulative-session subtraction is retained only as a diagnostic. Negative deltas and high CV prevent treating it as isolated block timing.

## Objective A: reusable C2f proof

Model16 is oracle-ready and has a model4-like 9-Conv C2f structure. It is a valid reuse-proof target with bounded implementation risk. It is not adjacent to the current custom island, so optimizing it alone would not form an end-to-end custom path.

## Objective B: contiguous acceleration island

The selected next objective is the adjacent `model.5 -> model.6 -> model.7 -> model.8` region. It contains 20 Conv nodes, directly extends model4, and targets the dominant measured suffix. Stage43 must first generate fixed-host quantized boundaries and board-isolated profiles; it must not inherit the invalid cumulative-subtraction ranking.

Known subsequent blockers are model9 SPPF/MaxPool and model10 attention/MatMul/Softmax. They are not authorized for implementation in the first contiguous-island stage.

## Objective C: prefix

The prefix is the second-largest region and deserves a later dedicated contract/profile stage. It includes different shapes and first-layer behavior, so it should not be mixed with the contiguous suffix task.

selected_next_objective: contiguous model5-8 island oracle/profile and first-block gate

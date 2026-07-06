# Cluster1 Non-IME Pipeline Note

IME execution remains cluster0-only on CPU0-3.

Stage20 did not run IME on CPU4-7 and did not add cluster1 scheduling.

Cluster1 may be considered later only for non-IME work, with explicit scheduling and correctness proof. It is not part of Stage20.

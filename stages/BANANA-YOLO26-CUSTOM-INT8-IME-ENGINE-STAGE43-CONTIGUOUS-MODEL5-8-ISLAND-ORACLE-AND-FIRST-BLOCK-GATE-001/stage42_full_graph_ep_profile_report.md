# Stage 42 EP Profile Reconciliation

The recovered Stage 42 JSON has SHA-256 `12e9cdef9e270d0c03316772fb817ba2e81b9176bdd518c7ad93362d181baae5`.

It is not a full-graph profile. It contains 10 events and six node events from the model0 diagnostic cut. All six node events record `CPUExecutionProvider`. The artifact therefore proves CPU assignment for those six cut nodes only; it cannot prove provider assignment for every full-model node.

Stage 43 isolated model5-8 profiles expose provider fields for all captured node events: model5 16, model6 128, model7 16, and model8 128 events, all assigned to `CPUExecutionProvider`. Registered-provider inventory is not used as assignment proof.

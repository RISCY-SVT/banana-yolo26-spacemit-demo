# Board ORT model5 forensics

ORT profiling proves model5 executes `QLinearConv`, inserted transposes, Q/DQ,
Sigmoid, and Mul on `CPUExecutionProvider`. Static library scanning found 192
accepted-mask vmadot instruction words, but the resolvable nearby symbols are
SpacemiT SQ4BitGemm IME packing paths, not proof of model5 QLinearConv execution.
The board has no `perf` command and the runtime is stripped enough that no inner
call graph was recovered. Classification: `code-present-but-execution-unknown`.

The profile's QLinearConv events average roughly `5.047 ms`; full isolated model5
at four threads is `11.701 ms`, showing that transposes and activation/QDQ are a
material part of the winning runtime path.

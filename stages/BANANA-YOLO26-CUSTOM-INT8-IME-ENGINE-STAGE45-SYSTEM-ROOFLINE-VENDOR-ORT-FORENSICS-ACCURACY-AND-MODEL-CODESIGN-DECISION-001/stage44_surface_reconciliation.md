# Stage44 surface reconciliation

Stage44's official classification remains `stage44-model5-exact-no-net-win`.
The `24157.4 us` custom value starts at the model4 preactivation code and includes
approximately `2660 us` of model4 final activation/requant. The `11701.121842 us`
ORT arm starts at model4 postactivation. Their direct ratio therefore overstates
the model5-only custom deficit.

The subtraction-only custom model5 residual is `21497.400 us`, still
`83.721%` slower than the four-thread ORT block, but this is a
diagnostic estimate rather than a benchmark. Stage44's instrumentation-off paired
scaffold is the authoritative system result: adding the exact model5 island made
the scaffold `4198.784898 us` (`0.821898%`) slower.

This report also supersedes the Stage44 repository placeholder with final HEAD
`bdefd89cc4247cb9e0ddac6fd06b561b05d29c87` without rewriting historical raw evidence.

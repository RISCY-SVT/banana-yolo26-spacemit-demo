# D1-D3 Verification

D1: the immutable published example fails Python API binding (unsupported
fp_weight; missing rounders). The current guide is byte-matched to
samples/reconstruction_minimal.py. Its real AdaptiveWeightRounder,
ReconstructionConfig and teacher/student callbacks execute on tiny CPU tensors
with a local seed and eight-iteration bound. Shape/dtype/finite-output and
documented result fields are asserted. This is a synthetic API exercise, not
full BRECQ or detector validation. Old tag contents are untouched.

D2: published >=3.9 contradicted ONNX 1.21.0 metadata (>=3.10); the accepted
NumPy 2.5.2 closure further requires >=3.12. Maintenance metadata conservatively
declares >=3.12.3,<3.13. Python 3.12.3/Linux x86_64 is the actually executed
reference. Other patch/platform combinations are not separately certified.
No numeric dependency was downgraded or broadened. Fresh package installs use
separate venvs preseeded offline with the accepted closure; this is NOT a fresh
internet dependency-resolution claim. Pip cross-target rejections are not
executed lower-version Python installations.

D3: clean committed-source audit covers 49 documents, 164 fenced blocks and
164 links. Fourteen Python blocks compile; exactly one fenced example is
executed-synthetic/API-bound by the checker. Zero detector workflows are
executed. 146 local links pass; 18 external links are not checked. Text/YAML
fragments retain explicit states and reasons. Block IDs bind content hashes.
Historical Stage64 recipes are not exact frozen B2/C2 reproduction.

Install config checks bind two shipped configs after the explicit adapter
data_list_path=None; model generation is never invoked. Original placeholder
paths remain in docs. Earlier harness mistakes (wrong API module, an expected
missing user dataset, canonical specifier ordering) remain in raw receipts and
are superseded by successful corrected checks, not erased.

The full inherited XSlim unit suite passes 218 cases plus 65 subtests; the
final D1-D3 focused rerun passes 26 plus 2 subtests. Banana focused tooling:
28 pass. Small synthetic tests from the inherited suite are not a YOLO PTQ
campaign. Ruff, compileall, shipped shell syntax/ShellCheck and strict mypy on
the three changed executable Python files pass. No runtime Python file changes.

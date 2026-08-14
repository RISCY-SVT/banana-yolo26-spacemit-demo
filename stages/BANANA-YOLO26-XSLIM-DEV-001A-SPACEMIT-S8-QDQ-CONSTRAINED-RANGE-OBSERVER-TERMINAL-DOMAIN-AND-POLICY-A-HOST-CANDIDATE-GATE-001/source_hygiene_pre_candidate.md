# Source Hygiene Before Candidate Generation

- XSlim source changes are confined to the generic constrained-range selector/observer, target-profile validator, tests, documentation, notices, and development version metadata.
- No private YOLO tensor name, model, dataset path, credential, or release artifact is embedded in XSlim source.
- The published `v2.1.2-riscy.1` tag and wheel remain immutable.
- The development wheel and normalized sdist install successfully in fresh environments; `pip check`, CLI smokes, Ruff, compileall, focused mypy, 174 pytest tests, and 65 subtests pass.
- No-override B2 deployable, inference, and tail bytes are exactly identical to the frozen artifacts.
- Raw activations, generated models, logs, and predictions remain outside Git under the stage raw root.

# License and Provenance Report

Release status: `pass-with-disclosed-upstream-per-file-metadata-debt`.

The wheel, sdist, source and documentation archives contain the Apache-2.0 license, upstream identity, modification record, downstream notice and third-party notices/licenses. The SPDX 2.3 SBOM validates and covers 222 source files. `twine check` and `check-wheel-contents` pass.

REUSE 5.1.1 reports zero bad, deprecated, missing-file, unused-license or read-error findings, but the inherited tree is not REUSE 3.3 compliant: only 29/241 files carry detected copyright metadata and 0/241 carry per-file license metadata. This is not hidden or globally waived. Adding headers to 241 brownfield files solely for this closure would create high provenance churn; the debt is accepted as inherited while aggregate legal artifacts and the release SBOM remain mandatory.

No model, weight, prediction, image, dataset, SpaceMIT runtime, vendor binary, credential or private lab path is released. One literal `/data/` string is an allowlisted negative-test sentinel asserting that a sanitized example does not contain a lab path; it is not a path disclosure.

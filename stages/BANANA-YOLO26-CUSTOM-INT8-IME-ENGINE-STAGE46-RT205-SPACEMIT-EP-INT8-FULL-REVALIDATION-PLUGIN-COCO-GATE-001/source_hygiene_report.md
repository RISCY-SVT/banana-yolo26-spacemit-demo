# Source hygiene

- Project start tree was clean at the required head.
- Vendor archives/libraries, COCO data, model binaries, outputs, subgraphs,
  profiles, build trees, and raw board logs remain outside Git.
- Repository changes are limited to diagnostic source, the tiny plugin proof,
  reproducible storage-skill source/installer, Stage45 addenda, Stage46 reports,
  and the Stage47 prompt.
- `/data/ncnn` was not mutated.
- The only preliminary secret-pattern hit is the storage skill self-test's own
  deny-list regex; it contains no credential value and is an intentional self-match.
- Symlink, large-file, secret/private-path, `git diff --check`, and staged-diff
  checks are required immediately before the local commit and are recorded in
  the shared command ledger/result packet.

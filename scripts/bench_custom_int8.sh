#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

cat <<USAGE
custom_int8_engine Stage 0 benchmark placeholder

No benchmark is implemented in Stage 0.
Future protocol:
  --pin cluster0
  --threads 4
  --warmup 10
  --runs 100
  --repeats 5
  record model/image/output hashes

repo: ${ROOT_DIR}
USAGE

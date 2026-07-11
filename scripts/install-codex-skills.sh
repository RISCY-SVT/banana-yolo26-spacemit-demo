#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "$0")/.." && pwd)"
source_dir="$repo_root/tools/codex-skills/k1x_board_storage_policy"
destination_root="${CODEX_SKILLS_ROOT:-${CODEX_HOME:-${HOME}/.codex}/skills}"
destination_dir="$destination_root/k1x_board_storage_policy"

(cd "$repo_root" && sha256sum --check tools/codex-skills/SHA256SUMS)
"$source_dir/selftest.sh" "$source_dir/SKILL.md"
install -d "$destination_dir"
install -m 0644 "$source_dir/SKILL.md" "$destination_dir/SKILL.md"
install -m 0755 "$source_dir/selftest.sh" "$destination_dir/selftest.sh"
cmp "$source_dir/SKILL.md" "$destination_dir/SKILL.md"
cmp "$source_dir/selftest.sh" "$destination_dir/selftest.sh"
"$destination_dir/selftest.sh" "$destination_dir/SKILL.md"
printf 'installed_skill=%s\n' "$destination_dir"

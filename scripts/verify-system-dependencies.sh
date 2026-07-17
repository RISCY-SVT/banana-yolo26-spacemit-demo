#!/usr/bin/env bash
set -euo pipefail

if [[ ${1:-} == --help || ${1:-} == -h ]]; then
  echo "usage: $0 [RELEASE_ROOT]"
  echo "Checks the supported-board system sonames required by an extracted release."
  exit 0
fi
root=${1:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
manifest=$root/required-system-sonames.tsv
[[ -f $manifest ]] || { echo "missing dependency manifest: $manifest" >&2; exit 2; }

missing=0
while IFS=$'\t' read -r soname role; do
  [[ $soname == soname ]] && continue
  [[ -n $soname ]] || continue
  if ldconfig -p 2>/dev/null | grep -F " $soname " >/dev/null; then
    printf 'found\t%s\t%s\n' "$soname" "$role"
    continue
  fi
  found=
  for directory in /lib /usr/lib /lib64 /usr/lib64; do
    [[ -d $directory ]] || continue
    found=$(find "$directory" -type f -name "$soname" -print -quit 2>/dev/null || true)
    [[ -z $found ]] || break
  done
  if [[ -n $found ]]; then
    printf 'found\t%s\t%s\t%s\n' "$soname" "$role" "$found"
  else
    printf 'missing\t%s\t%s\n' "$soname" "$role" >&2
    missing=$((missing + 1))
  fi
done <"$manifest"

((missing == 0)) || exit 1

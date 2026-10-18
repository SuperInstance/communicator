#!/usr/bin/env bash
# Scan every local repo's FULL HISTORY for credential-shaped strings.
# Publishing a repo whose history contains a key leaks it permanently -- the
# object stays reachable by SHA after a "fix". So this walks every blob in
# every commit, not just the working tree.
set -uo pipefail

PATTERNS='apikey_[A-Za-z0-9_-]{16,}|cfut_[A-Za-z0-9_-]{16,}|sk-[A-Za-z0-9_-]{20,}|gh[pousr]_[A-Za-z0-9]{20,}|AKIA[0-9A-Z]{16}|xox[baprs]-[A-Za-z0-9-]{10,}|-----BEGIN [A-Z ]*PRIVATE KEY-----|(api[_-]?key|secret|token|password|passwd)["'"'"']?\s*[:=]\s*["'"'"']?[A-Za-z0-9_/+-]{24,}'

for dir in /home/proart/projects/*/; do
  name=$(basename "$dir")
  [ -d "$dir/.git" ] || { printf '%-16s (not a git repo)\n' "$name"; continue; }
  cd "$dir" || continue

  branch=$(git rev-parse --abbrev-ref HEAD 2>/dev/null)
  ncommits=$(git rev-list --count --all 2>/dev/null || echo 0)

  # Every blob reachable from any ref, concatenated.
  hits=$(git rev-list --objects --all 2>/dev/null | awk '{print $1}' | sort -u | \
    while read -r o; do
      [ "$(git cat-file -t "$o" 2>/dev/null)" = "blob" ] || continue
      git cat-file -p "$o" 2>/dev/null
    done | grep -aoEi "$PATTERNS" | sort -u)

  n=$(printf '%s' "$hits" | grep -c . || true)
  if [ "$n" -eq 0 ]; then
    printf '%-16s %-8s %2s commits  CLEAN\n' "$name" "$branch" "$ncommits"
  else
    printf '%-16s %-8s %2s commits  *** %d HIT(S) ***\n' "$name" "$branch" "$ncommits" "$n"
    printf '%s\n' "$hits" | head -6 | sed 's/^/                     /'
  fi
done
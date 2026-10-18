#!/usr/bin/env bash
# Run zcode against whichever model the account actually has registered.
#
# Availability is not stable. Right now only GLM-5.3 resolves on this account:
# both GLM-5.3-flash and GLM-5.2 fail with
#   "Provider Registry does not contain Model"
# but that has been true before and is expected to change, so the candidates
# are probed in preference order at dispatch time rather than hardcoded.
#
# A model is "available" only if it answers a trivial prompt correctly. Probing
# by error string alone is not enough -- an unavailable model can fail at a
# different layer than an unavailable provider.
#
# Usage: tools/zcode-run.sh "<prompt>" [more args...]
#   prints the chosen model to stderr; zcode's own output goes to stdout.

set -uo pipefail

ZCODE_BIN="${ZCODE_BIN:-/home/proart/.nvm/versions/node/v24.21.0/bin/zcode}"
PROVIDER="${ZCODE_PROVIDER:-account:zai-individual-coding-plan}"

# Best first. Cheap variants are faster but this is the one that reliably works.
CANDIDATES=(
  "GLM-5.3"
  "GLM-5.3-flash"
  "GLM-5.2"
  "glm-5.3"
  "glm-5.3-flash"
  "glm-5.2"
)

PROBE_TIMEOUT="${ZCODE_PROBE_TIMEOUT:-75}"

log() { printf '[zcode-run] %s\n' "$*" >&2; }

available() {
  local model="$1" out
  out="$(cd /tmp && timeout "$PROBE_TIMEOUT" "$ZCODE_BIN" -p "Say OK" \
         --model "$PROVIDER/$model" 2>&1)"
  # Only a literal OK counts. Anything else -- including an empty response from
  # a hung process -- means unavailable.
  [ "$(printf '%s' "$out" | tr -d '\r' | grep -c '^OK$')" -ge 1 ]
}

if [ "$#" -lt 1 ]; then
  log "usage: $0 \"<prompt>\""
  exit 2
fi

chosen=""
for m in "${CANDIDATES[@]}"; do
  if available "$m"; then
    chosen="$m"
    break
  fi
  log "unavailable: $m"
done

if [ -z "$chosen" ]; then
  log "NO AVAILABLE MODEL. All of: ${CANDIDATES[*]}"
  log "Try: zcode login --oauth   (needs a browser)"
  exit 1
fi

log "using $PROVIDER/$chosen"
cd "${ZCODE_CWD:-/home/proart/projects/communicator}"
exec "$ZCODE_BIN" -p "$1" --model "$PROVIDER/$chosen" "${@:2}"
#!/usr/bin/env bash
# deploy-hoso.sh — build the static site and ship it to holisticsofa.ai/nlm-fad/
#
# The demos are a static resident of the HolisticSofa gateway (deployment: "static",
# spa: false, registered once in /Users/holisticsofa/projects.json as "nlm-fad").
# The gateway serves <its home>/nlm-fad/dist/, so this script builds _site/ with the
# /nlm-fad/ base path and rsyncs it there. Re-run after any demo change; no gateway
# reload is needed for content updates.
#
#   scripts/deploy-hoso.sh
set -euo pipefail
cd "$(dirname "$0")/.."

REMOTE_USER="holisticsofa"
REMOTE_HOST="lovelace"
REMOTE_PATH="/Users/holisticsofa/nlm-fad/dist"

echo "== build (base /nlm-fad/)"
SITE_BASE=/nlm-fad/ scripts/build-site.sh
[ -f _site/index.html ] || { echo "build produced no _site/index.html" >&2; exit 1; }

echo "== ship"
ssh "${REMOTE_USER}@${REMOTE_HOST}" "mkdir -p '${REMOTE_PATH}'"
rsync -az --delete _site/ "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_PATH}/"

echo "== verify"
for p in "" fisheye-2d/ diffusion-mag/ data-flow/ face-magnify/; do
  code=$(curl -s -o /dev/null -w '%{http_code}' "https://holisticsofa.ai/nlm-fad/${p}")
  echo "  /nlm-fad/${p}  ${code}"
done
echo "done → https://holisticsofa.ai/nlm-fad/"

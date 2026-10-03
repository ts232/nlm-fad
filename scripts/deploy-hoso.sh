#!/usr/bin/env bash
# deploy-hoso.sh — build the static site and ship it to holisticsofa.ai/nlm-fad/
#
# The demos are a static resident of the HolisticSofa gateway, registered in its project
# registry as "nlm-fad". The gateway serves <its home>/nlm-fad/dist/, so this script builds
# _site/ with the /nlm-fad/ base path and rsyncs it there. Re-run after any demo change; no
# gateway reload is needed for content updates.
#
# Where it ships comes from scripts/deploy-hoso.env (gitignored; copy deploy-hoso.env.example)
# or from the environment: DEPLOY_USER, DEPLOY_HOST, DEPLOY_PATH.
#
#   scripts/deploy-hoso.sh
set -euo pipefail
cd "$(dirname "$0")/.."

[ -f scripts/deploy-hoso.env ] && . scripts/deploy-hoso.env
: "${DEPLOY_USER:?set DEPLOY_USER (see scripts/deploy-hoso.env.example)}"
: "${DEPLOY_HOST:?set DEPLOY_HOST (see scripts/deploy-hoso.env.example)}"
: "${DEPLOY_PATH:?set DEPLOY_PATH (see scripts/deploy-hoso.env.example)}"
REMOTE_USER="$DEPLOY_USER"
REMOTE_HOST="$DEPLOY_HOST"
REMOTE_PATH="$DEPLOY_PATH"

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

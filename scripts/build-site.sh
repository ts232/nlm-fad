#!/usr/bin/env bash
# build-site.sh — build all four demos under a common base path and assemble
# the static site in _site/ (landing page + one subdirectory per demo).
#
#   scripts/build-site.sh              # base /nlm-fad/  (GitHub Pages project site)
#   SITE_BASE=/ scripts/build-site.sh  # any other base path
set -euo pipefail
cd "$(dirname "$0")/.."

BASE="${SITE_BASE:-/nlm-fad/}"
OUT="_site"
DEMOS=(fisheye-2d diffusion-mag data-flow face-magnify)

rm -rf "$OUT"
mkdir -p "$OUT"
cp pages/index.html "$OUT/index.html"
touch "$OUT/.nojekyll"

for d in "${DEMOS[@]}"; do
  echo "== $d"
  pushd "demos/$d" >/dev/null
  if [ ! -d node_modules ]; then npm ci --no-audit --no-fund; fi
  npx vite build --base="${BASE}${d}/" --outDir "../../$OUT/$d" --emptyOutDir
  popd >/dev/null
done

echo "built → $OUT/  (base path: $BASE)"

#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"

echo "[1/3] Sync web_sim_envs to public/ ..."
mkdir -p "$ROOT/public/web_sim_envs"
rsync -a --delete "$ROOT/src/assets/web_sim_envs/" "$ROOT/public/web_sim_envs/"

echo "[2/3] Render viz_main_compare.pdf to public/results/viz_main_compare_full.png ..."
mkdir -p "$ROOT/public/results"
magick -density 300 "$ROOT/src/assets/paper_material/viz_main_compare.pdf[0]" \
  -background white -alpha remove -alpha off \
  "$ROOT/public/results/viz_main_compare_full.png"

echo "[3/3] Done."
echo "Generated:"
echo "  - public/web_sim_envs/"
echo "  - public/results/viz_main_compare_full.png"

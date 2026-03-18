#!/usr/bin/env bash

# echo "PATH="$(brew --prefix ffmpeg-full)/bin:$PATH" bash scripts/extract-efficiency-frames.sh"

PATH="$(brew --prefix ffmpeg-full)/bin:$PATH"
mkdir -p /tmp/vectorworld_scrub

for f in \
  src/assets/paper_material/ScenDream_denoise.mp4 \
  src/assets/paper_material/ScenDream_denoise2.mp4 \
  src/assets/paper_material/ScenDream_denoise3.mp4 \
  src/assets/paper_material/VectorWorld_flow_denoise.mp4 \
  src/assets/paper_material/VectorWorld_flow_denoise2.mp4 \
  src/assets/paper_material/VectorWorld_flow_denoise3.mp4 \
  src/assets/paper_material/VectorWorld_meanflow_denoise.mp4 \
  src/assets/paper_material/VectorWorld_meanflow_denoise2.mp4 \
  src/assets/paper_material/VectorWorld_meanflow_denoise3.mp4
do
  ffmpeg -y -i "$f" \
    -c:v libx264 -preset slow -crf 18 \
    -g 12 -keyint_min 12 -sc_threshold 0 \
    -movflags +faststart -pix_fmt yuv420p -an \
    "/tmp/vectorworld_scrub/$(basename "$f")"
done

cp /tmp/vectorworld_scrub/*.mp4 src/assets/paper_material/
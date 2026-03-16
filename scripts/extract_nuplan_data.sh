#!/usr/bin/env bash

CODE_DIR="$PROJECT_ROOT/tools/preprocess"

cd "$CODE_DIR"
# ignore all the tf warnings
python generate_nuplan_dataset.py
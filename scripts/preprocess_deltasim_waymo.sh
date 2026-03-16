#!/bin/bash
set -e

source scripts/define_env_variables.sh

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

export CTRL_SIM_DISABLE_CUDA=1
export CUDA_VISIBLE_DEVICES=""

export CTRL_SIM_PREPROCESS_SPLITS=train
export CTRL_SIM_SCAN_EXISTING=1
export CTRL_SIM_PREPROCESS_WORKERS=64
export CTRL_SIM_PROGRESS_INTERVAL=200

export CTRL_SIM_NUM_SHARDS=8
export CTRL_SIM_SHARD_ID=7

python3 tools/preprocess/preprocess_deltasim.py \
  dataset_name=waymo \
  model_name=deltasim \
  deltasim.dataset.preprocess=false \
  deltasim.dataset.preprocess_dir=metadata/datasets/waymo/deltasim
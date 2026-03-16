#!/bin/bash

source scripts/define_env_variables.sh

# split: train / val / test

python3 tools/preprocess/preprocess_waymo.py \
      dataset_name=waymo \
      preprocess_waymo.mode=train \
      preprocess_waymo.num_workers=1 \
      ae.dataset.save_to_cos=False \
      ae.dataset.preprocess_dir=metadata/datasets/waymo/sd_ae_motion_preprocess 2>&1 | tee metadata/outputs/log/data/predata_train_waymo.log
#!/bin/bash

# split: train / val / test

source scripts/define_env_variables.sh

python3 tools/preprocess/preprocess_nuplan.py \
      dataset_name=nuplan \
      preprocess_nuplan.mode=train \
      preprocess_nuplan.num_workers=1 \
      ae.dataset.preprocess_dir=metadata/datasets/nuplan/sd_ae_motion_preprocess 2>&1 | tee metadata/outputs/log/data/predata_train_waymo.log
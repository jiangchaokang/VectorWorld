#!/bin/bash

# Batch generation sample + uniform metrics assessment, 50k for gentlemen (in)DDPM‑LDM(for example)
python eval.py \
  dataset_name=waymo \
  model_name=ldm \
  ldm.model.ldm_type=diffusion \
  ldm.eval.mode=initial_scene \
  ldm.eval.num_samples=50000 \
  ldm.eval.batch_size=64 \
  ldm.eval.cache_samples=True \
  ldm.eval.run_name=sd_ldm_diff_waymo_motion_large \
  ldm.eval.save_dir=/workspace/jca3data/vectorworld/outputs/checkpoints/sd_ldm_diff_waymo_motion_large/initial_scene_samples \
  ldm.model.autoencoder_path=outputs/checkpoints/sd_ae_waymo_motion_large/last.ckpt \
  ldm.eval.ckpt_path=outputs/checkpoints/sd_ldm_diff_waymo_motion_large/last.ckpt \
  ldm.train.guidance_scale=4.0


# Calculate Lane + Agent + Motion Indicators
python eval.py \
  dataset_name=waymo \
  model_name=ldm \
  ldm.model.ldm_type=diffusion \
  ldm.eval.mode=metrics \
  ldm.eval.run_name=sd_ldm_diff_waymo_motion_large \
  ldm.eval.metrics.samples_path=/workspace/jca3data/vectorworld/outputs/checkpoints/sd_ldm_diff_waymo_motion_large/initial_scene_samples/sd_ldm_diff_waymo_motion_large/initial_scene_samples \
  ldm.eval.metrics.metrics_save_path=outputs/checkpoints/sd_ldm_diff_waymo_motion_large/metrics \
  ldm.eval.metrics.eval_set=data/metadata/waymo_eval_set.pkl \
  ldm.eval.metrics.gt_test_dir=/workspace/jca3data/datasets/waymo/sd_ae_motion_preprocess/val \


# Metrics contains:
# - Lane: Urban Planning (Connectivity / Density / Reach / Convention) + roote Length + endpoint recognition
# -Agent:JSD(Recent proximity, horizontal deviation, angle deviation, width, speed) + collision rate
# -Motion (motion code if enabled): path length / span / offset distributionJSDWait.
#!/bin/bash
# Train DeltaSim behavior model
# Usage: bash scripts/train_deltasim.sh [waymo]

DATASET=${1:-waymo}

python3 tools/train.py \
    dataset_name=${DATASET} \
    model_name=deltasim \
    deltasim.train.devices=1 \
    deltasim.train.max_steps=100000 \
    deltasim.train.lr=6e-5 \
    deltasim.train.init_ckpt_path=metadata/checkpoints/waymo/delta_sim/epoch41_step79884.ckpt \
    deltasim.train.run_name=ctrlsim_hybrid_head \
    deltasim.model.dkal.enabled=true \
    deltasim.model.residual_refine.enabled=true \
    deltasim.model.residual_refine.loss_coef=0.1 \
    deltasim.model.residual_refine.w_xy=1.0 \
    deltasim.model.residual_refine.w_yaw=0.3 \
    deltasim.dataset.preprocess=true \
    deltasim.dataset.rtg.horizon_steps=50 \
    deltasim.dataset.rtg.discount=0.97 \
    deltasim.dataset.rtg.discretization=350 \
    deltasim.model.map_encoder.use_point_pos_emb=true \
    deltasim.model.attn_gate.enabled=true \
    deltasim.model.attn_gate.init_bias=-2.0 \
    deltasim.model.rtg_conditioning.enabled=true \
    deltasim.model.rtg_conditioning.anneal_steps=30000 \
    deltasim.model.rtg_conditioning.alpha_start=1.0 \
    deltasim.model.rtg_conditioning.alpha_end=0.0 \
    deltasim.model.rtg_conditioning.stopgrad_pred=true \
    deltasim.model.rtg_conditioning.use_tilt_in_eval=true \
    deltasim.model.phys_prior.enabled=true \
    deltasim.model.risk_se2_loss.enabled=true \
    deltasim.model.risk_se2_loss.coef=0.01 \
    deltasim.model.temporal_smooth_loss.enabled=true \
    deltasim.model.temporal_smooth_loss.coef=0.02
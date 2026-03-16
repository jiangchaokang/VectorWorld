#!/bin/bash
# ====================================================
# Train VectorWorld EGR-DiT (Latent Generative Model)
# ====================================================
# Usage: bash scripts/train_egr_dit.sh [dataset_name] [devices] [max_steps] [autoencoder_path] [init_ckpt_path] [ldm_type]
# Example: bash scripts/train_egr_dit.sh waymo 1 165000 /path/to/ae.ckpt /path/to/init.ckpt flow

# ==================== Configurable Parameters ====================
DATASET_NAME=${1:-waymo}              # Dataset: waymo | nuplan
DEVICES=${2:-1}                       # Number of GPU devices
MAX_STEPS=${3:-165000}                # Maximum training steps
AUTOENCODER_PATH=${4:-"metadata/checkpoints/waymo/vae/last.ckpt"}             # Path to autoencoder checkpoint
INIT_CKPT_PATH=${5:-"metadata/checkpoints/waymo/ldm/diffusion/last.ckpt"}     # Path to initial checkpoint (optional)
LDM_TYPE=${6:-diffusion}              # LDM type: flow | diffusion | meanflow

# ==================== Build Additional Parameters ====================
EXTRA_PARAMS=""
if [ "${LDM_TYPE}" != "meanflow" ]; then
    EXTRA_PARAMS="ldm.model.use_rel_bias=true \
    ldm.model.use_gcf=true \
    ldm.model.qk_norm=true \
    ldm.model.attn_logit_clip=30.0 \
    ldm.model.lane_rel_dim=64 \
    ldm.model.agent_rel_dim=32 \
    ldm.model.edge_dim=32 \
    ldm.model.use_cross_rel_bias=true \
    ldm.model.use_rel_gate=true \
    ldm.model.gcf_var_scale=0.15"
fi

python3 tools/train.py \
    dataset_name=${DATASET_NAME} \
    model_name=ldm \
    ldm.train.devices=${DEVICES} \
    ldm.train.max_steps=${MAX_STEPS} \
    ldm.train.check_val_every_n_epoch=2 \
    ldm.model.autoencoder_path=${AUTOENCODER_PATH} \
    ldm.train.init_ckpt_path=${INIT_CKPT_PATH} \
    ldm.model.ldm_type=${LDM_TYPE} \
    ${EXTRA_PARAMS}
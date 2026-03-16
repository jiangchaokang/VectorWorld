#!/bin/bash
# Train VectorWorld VAE (scene autoencoder)
# Usage: bash scripts/train_vae.sh [waymo|nuplan]

DATASET=${1:-waymo}

python3 tools/train.py \
    dataset_name=${DATASET} \
    model_name=vae \
    ae.dataset.preprocess=True \
    ae.train.devices=1 \
    ae.train.max_steps=85000 \
    ae.train.init_ckpt_path=metadata/checkpoints/waymo/vae/last.ckpt
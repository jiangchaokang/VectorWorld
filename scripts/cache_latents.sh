#!/usr/bin/env bash
set -e

source scripts/define_env_variables.sh
CKPT_PATH="/workspace/jca3sgh/e2e/vectorworld/outputs/checkpoints/sd_ae_waymo_motion_large/last.ckpt"
RUN_NAME="sd_ae_waymo_motion_large"
SPLIT_NAME="val"  # split: train / val / test
NUM_SHARDS=1
BATCH_SIZE=64      # Batch size per process
LATENT_DIR="metadata/datasets/waymo/sd_ae_large_motion_latents"
LOG_DIR="metadata/outputs/log"
# ==========================================================

mkdir -p "${LOG_DIR}"

echo "Checkpoint: ${CKPT_PATH}"
echo "Split:      ${SPLIT_NAME}"
echo "Latent dir: ${LATENT_DIR}"
echo "Batch size per GPU: ${BATCH_SIZE}"
echo

for SHARD_ID in $(seq 0 $((NUM_SHARDS-1))); do
  export CUDA_VISIBLE_DEVICES=${SHARD_ID}

  echo "Starting process shard ${SHARD_ID}/${NUM_SHARDS} on GPU ${CUDA_VISIBLE_DEVICES}..."

  python3 tools/generate.py \
    dataset_name=waymo \
    model_name=vae \
    ae.eval.mode=null \
    ae.eval.ckpt_path="${CKPT_PATH}" \
    ae.eval.run_name="${RUN_NAME}" \
    ae.eval.split_name="${SPLIT_NAME}" \
    ae.eval.save_dir="outputs/checkpoints" \
    ae.eval.batch_size=${BATCH_SIZE} \
    ae.eval.accelerator="gpu" \
    ae.eval.devices=1 \
    ae.datamodule.val_batch_size=${BATCH_SIZE} \
    ae.eval.cache_latents.enable_caching=True \
    ae.eval.cache_latents.split_name="${SPLIT_NAME}" \
    ae.eval.cache_latents.latent_dir="${LATENT_DIR}" \
    ae.eval.cache_latents.num_shards=${NUM_SHARDS} \
    ae.eval.cache_latents.shard_id=${SHARD_ID} \
    ae.eval.cache_latents.save_to_cos=False \
    ae.eval.cache_latents.save_local_copy=True \
    > "${LOG_DIR}/cache_latents_shard_${SHARD_ID}.log" 2>&1 &

done

wait
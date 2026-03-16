#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# scripts/infer_egr_dit.sh — LDM inference / generation
#
# Env-var overrides (applied globally to all experiments):
#   DATASET       — dataset name
#   LDM_TYPE      — model type: diffusion | flow | meanflow
#   CKPT_PATH     — checkpoint path override
#   RUN_NAME      — output run name override
#   MODE          — eval mode (default: initial_scene)
#   NUM_SAMPLES   — number of samples   (default: 16)
#   BATCH_SIZE    — batch size           (default: 16)
#   VISUALIZE     — enable visualization (default: True)
# ─────────────────────────────────────────────────────────────────────────────

source scripts/define_env_variables.sh

# Rel-aware model args — automatically excluded when ldm_type=meanflow
REL_ARGS=(
  ldm.model.use_rel_bias=true    ldm.model.use_gcf=true
  ldm.model.qk_norm=true         ldm.model.attn_logit_clip=30.0
  ldm.model.lane_rel_dim=64      ldm.model.agent_rel_dim=32
  ldm.model.edge_dim=32          ldm.model.use_cross_rel_bias=true
  ldm.model.use_rel_gate=true    ldm.model.gcf_var_scale=0.15
  ldm.model.num_l2l_blocks=3
)

# run_generate <dataset> <ldm_type> <ckpt> <vae> <run_name> <mode> [extra_args...]
run_generate() {
  local dataset=${DATASET:-$1}    ldm_type=${LDM_TYPE:-$2}
  local ckpt=${CKPT_PATH:-$3}     vae=$4
  local run_name=${RUN_NAME:-$5}  mode=${MODE:-$6}
  shift 6

  local guidance
  local -a type_args=() rel_args=()

  if [[ "$ldm_type" == "meanflow" ]]; then
    guidance=1.0                      # guidance_scale must be 1.0 for meanflow
    type_args=(ldm.eval.meanflow_num_steps=3)
    # rel_args stays empty — rel-aware args not supported for meanflow
  else
    guidance=4.0
    rel_args=("${REL_ARGS[@]}")
    [[ "$ldm_type" == "flow" ]] && type_args=(ldm.model.flow_num_steps=24)
  fi

  python3 tools/generate.py \
    dataset_name="$dataset"    model_name=ldm \
    ldm.model.ldm_type="$ldm_type" \
    ldm.train.guidance_scale="$guidance" \
    "${type_args[@]}" \
    "${rel_args[@]}" \
    ldm.model.autoencoder_path="$vae" \
    ldm.eval.ckpt_path="$ckpt" \
    ldm.eval.run_name="$run_name" \
    ldm.eval.mode="$mode" \
    ldm.eval.num_samples="${NUM_SAMPLES:-16}" \
    ldm.eval.batch_size="${BATCH_SIZE:-16}" \
    ldm.eval.visualize="${VISUALIZE:-True}" \
    "$@"
}

# # ── Experiment 1: nuplan · diffusion ─────────────────────────────────────────
# run_generate \
#   nuplan diffusion \
#   metadata/checkpoints/nuplan/ldm/diffusion/last.ckpt \
#   metadata/checkpoints/nuplan/vae/last.ckpt \
#   ldm_diff_nuplan_relaware \
#   initial_scene

# # ── Experiment 2: waymo · flow · initial_scene ───────────────────────────────
# run_generate \
#   waymo flow \
#   metadata/checkpoints/waymo/ldm/flow/last.ckpt \
#   metadata/checkpoints/waymo/vae/last.ckpt \
#   ldm_flow_waymo_relaware \
#   initial_scene

# # ── Experiment 3: waymo · meanflow  (guidance=1.0 enforced, rel-aware disabled)
# run_generate \
#   waymo meanflow \
#   metadata/checkpoints/waymo/ldm/meanflow/last.ckpt \
#   metadata/checkpoints/waymo/vae/last.ckpt \
#   ldm_meanflow_waymo_relaware \
#   initial_scene

# # ── Experiment 4: waymo · flow · simulation ──────────────────────────────────
run_generate \
  waymo flow \
  metadata/checkpoints/waymo/ldm/flow/last.ckpt \
  metadata/checkpoints/waymo/vae/last.ckpt \
  diffusion_motion_waymo_sim_50 \
  simulation_environments \
  ldm.eval.sim_envs.route_length=50 \
  ldm.eval.sim_envs.overhead_factor=8 \
  ldm.eval.sim_envs.num_inpainting_candidates=10 \
  ldm.eval.sim_envs.nocturne_compatible_only=false \
  ldm.eval.sim_envs.viz_dir=outputs/viz/simulation_environments/flow \
  ldm.eval.sim_envs.web_export.enabled=false
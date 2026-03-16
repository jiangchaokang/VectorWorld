#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  bash scripts/run_simulation_parallel.sh --help

Examples:
  # 1) Online: direct simulation, no postprocess
  bash scripts/run_simulation_parallel.sh \
    sim=online \
    sim.mode=vectorworld_online \
    ldm.model.ldm_type=meanflow \
    ldm.eval.run_name=vectorworld_meanflow_deltasim_online

  # 2) Offline: run postprocess first, then simulation
  bash scripts/run_simulation_parallel.sh \
    sim=base \
    sim.mode=vectorworld \
    ldm.model.ldm_type=flow \
    ldm.eval.run_name=vectorworld_flow_deltasim_offline \
    postprocess_sim_envs.run_name=diffusion_motion_waymo_sim_500

  # 3) Offline + diffusion + custom postprocess paths
  bash scripts/run_simulation_parallel.sh \
    sim=base \
    ldm.model.ldm_type=diffusion \
    ldm.eval.run_name=my_diffusion_offline \
    postprocess_sim_envs.pre_path=outputs/checkpoints/diffusion_motion_waymo_sim_500/complete_sim_envs \
    postprocess_sim_envs.post_path=outputs/checkpoints/diffusion_motion_waymo_sim_500/complete_sim_envs_post

Notes:
  - sim=online: only runs tools/simulate_parallel.py
  - sim!=online: first runs tools/preprocess/postprocess_sim_envs.py, then simulation
  - meanflow:
      * does NOT use rel-bias / GCF related parameters
      * forces ldm.train.guidance_scale=1.0
  - flow / diffusion:
      * automatically use the required rel-bias / GCF related parameters
      * default ldm.train.guidance_scale=3.0 (can be overridden)
  - Other Hydra overrides are forwarded to tools/simulate_parallel.py
  - postprocess_sim_envs.* overrides are forwarded to tools/preprocess/postprocess_sim_envs.py
EOF
}

for arg in "$@"; do
  [[ "$arg" == "-h" || "$arg" == "--help" ]] && { usage; exit 0; }
done

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

source scripts/define_env_variables.sh

run_cmd() {
  printf '\n[Run]'
  printf ' %q' "$@"
  printf '\n'
  "$@"
}

PROTECTED_KEYS=(
  ldm.model.use_rel_bias
  ldm.model.use_gcf
  ldm.model.qk_norm
  ldm.model.attn_logit_clip
  ldm.model.lane_rel_dim
  ldm.model.agent_rel_dim
  ldm.model.edge_dim
  ldm.model.use_cross_rel_bias
  ldm.model.use_rel_gate
  ldm.model.gcf_var_scale
)

is_protected_key() {
  local key="${1%%=*}"
  local k
  for k in "${PROTECTED_KEYS[@]}"; do
    [[ "$key" == "$k" ]] && return 0
  done
  return 1
}

DATASET_NAME="waymo"
MODEL_NAME="deltasim"

SIM="online"
SIM_MODE=""
LDM_TYPE="meanflow"
RUN_NAME=""

POST_RUN_NAME="diffusion_motion_waymo_sim_500"
POST_ROUTE_LENGTH="100"
PRE_PATH=""
POST_PATH=""

GUIDANCE_SCALE_OVERRIDE=""
SIM_EXTRA_ARGS=()
POST_EXTRA_ARGS=()

for arg in "$@"; do
  case "$arg" in
    dataset_name=*)
      DATASET_NAME="${arg#dataset_name=}"
      ;;
    model_name=*)
      MODEL_NAME="${arg#model_name=}"
      ;;
    sim=*)
      SIM="${arg#sim=}"
      ;;
    sim.mode=*)
      SIM_MODE="${arg#sim.mode=}"
      ;;
    ldm.model.ldm_type=*)
      LDM_TYPE="${arg#ldm.model.ldm_type=}"
      ;;
    ldm.eval.run_name=*)
      RUN_NAME="${arg#ldm.eval.run_name=}"
      ;;
    ldm.train.guidance_scale=*)
      GUIDANCE_SCALE_OVERRIDE="${arg#ldm.train.guidance_scale=}"
      ;;
    postprocess_sim_envs.run_name=*)
      POST_RUN_NAME="${arg#postprocess_sim_envs.run_name=}"
      ;;
    postprocess_sim_envs.route_length=*)
      POST_ROUTE_LENGTH="${arg#postprocess_sim_envs.route_length=}"
      ;;
    postprocess_sim_envs.pre_path=*)
      PRE_PATH="${arg#postprocess_sim_envs.pre_path=}"
      ;;
    postprocess_sim_envs.post_path=*)
      POST_PATH="${arg#postprocess_sim_envs.post_path=}"
      ;;
    postprocess_sim_envs.*)
      POST_EXTRA_ARGS+=("$arg")
      ;;
    *)
      if is_protected_key "$arg"; then
        echo "Error: ${arg%%=*} is managed automatically by this script. Please do not pass it manually." >&2
        exit 1
      fi
      SIM_EXTRA_ARGS+=("$arg")
      ;;
  esac
done

[[ -n "$SIM_MODE" ]] || SIM_MODE=$([[ "$SIM" == "online" ]] && echo "vectorworld_online" || echo "vectorworld")
[[ -n "$RUN_NAME" ]] || RUN_NAME=$([[ "$SIM" == "online" ]] && echo "vectorworld_${LDM_TYPE}_deltasim" || echo "vectorworld_${LDM_TYPE}_deltasim_offline")
[[ -n "$PRE_PATH" ]] || PRE_PATH="outputs/checkpoints/${POST_RUN_NAME}/complete_sim_envs"
[[ -n "$POST_PATH" ]] || POST_PATH="outputs/checkpoints/${POST_RUN_NAME}/complete_sim_envs_post"

COMMON_ARGS=(
  "dataset_name=${DATASET_NAME}"
  "model_name=${MODEL_NAME}"
  "sim=${SIM}"
  "sim.mode=${SIM_MODE}"
  "sim.num_workers=4"
  "ldm.model.ldm_type=${LDM_TYPE}"
  "ldm.eval.meanflow_num_steps=3"
  "ldm.model.flow_num_steps=32"
  "ldm.model.autoencoder_path=metadata/checkpoints/${DATASET_NAME}/vae/last.ckpt"
  "ldm.eval.run_name=${RUN_NAME}"
  "ldm.eval.ckpt_path=metadata/checkpoints/${DATASET_NAME}/ldm/${LDM_TYPE}/last.ckpt"
  "ldm.eval.sim_envs.route_length=200"
  "ldm.eval.sim_envs.overhead_factor=10"
  "ldm.eval.sim_envs.num_inpainting_candidates=8"
  "ldm.eval.sim_envs.nocturne_compatible_only=false"
  "sim.behaviour_model.variant=ours"
  "sim.behaviour_model.tilt=10"
  "sim.behaviour_model.action_temperature=1.0"
  "sim.behaviour_model.init_history_from_motion=false"
  "sim.behaviour_model.history_from_motion_num_points=6"
  "sim.behaviour_model.motion_history.speed_mode=clamped"
  "sim.behaviour_model.motion_history.speed_scale=1.0"
  "sim.behaviour_model.motion_history.min_speed=0.0"
  "sim.behaviour_model.motion_history.max_speed=30.0"
  "sim.behaviour_model.motion_history.max_scale=2.0"
  "sim.behaviour_model.inference.residual_refine.enabled=true"
  "sim.behaviour_model.inference.residual_refine.use_refine_in_eval=true"
  "sim.behaviour_model.inference.residual_refine.refine_scale=1.0"
  "sim.behaviour_model.inference.collision_avoidance.enabled=true"
  "sim.behaviour_model.inference.collision_avoidance.beta=2.0"
  "sim.behaviour_model.inference.collision_avoidance.margin=0.2"
  "sim.behaviour_model.inference.collision_avoidance.min_dist_clip=0.05"
  "sim.behaviour_model.inference.rtg_inference.mode=one_pass_like"
  "sim.save_trajectory=true"
  "sim.trajectory_path=outputs/viz/${RUN_NAME}/pkl"
  "sim.visualize=true"
  "sim.movie_path=outputs/viz/${RUN_NAME}/vis"
)

case "$LDM_TYPE" in
  meanflow)
    if [[ -n "$GUIDANCE_SCALE_OVERRIDE" && ! "$GUIDANCE_SCALE_OVERRIDE" =~ ^1(\.0+)?$ ]]; then
      echo "Error: meanflow requires ldm.train.guidance_scale=1.0" >&2
      exit 1
    fi
    MODEL_ARGS=(
      "ldm.train.guidance_scale=1.0"
    )
    ;;
  flow|diffusion)
    MODEL_ARGS=(
      "ldm.train.guidance_scale=${GUIDANCE_SCALE_OVERRIDE:-3.0}"
      "ldm.model.use_rel_bias=true"
      "ldm.model.use_gcf=true"
      "ldm.model.qk_norm=true"
      "ldm.model.attn_logit_clip=30.0"
      "ldm.model.lane_rel_dim=64"
      "ldm.model.agent_rel_dim=32"
      "ldm.model.edge_dim=32"
      "ldm.model.use_cross_rel_bias=true"
      "ldm.model.use_rel_gate=true"
      "ldm.model.gcf_var_scale=0.15"
    )
    ;;
  *)
    echo "Error: unsupported ldm.model.ldm_type=${LDM_TYPE}. Expected one of: meanflow, flow, diffusion." >&2
    exit 1
    ;;
esac

echo "[Info] sim=${SIM}, sim.mode=${SIM_MODE}, ldm.model.ldm_type=${LDM_TYPE}, ldm.eval.run_name=${RUN_NAME}"

if [[ "$SIM" == "online" ]]; then
  MODE_ARGS=(
    "sim.num_online_scenarios=4"
    "sim.online_batch_size=4"
  )
else
  echo "[Info] offline mode: postprocess ${PRE_PATH} -> ${POST_PATH}"

  PRE_ARGS=(
    "dataset_name=${DATASET_NAME}"
    "postprocess_sim_envs.run_name=${POST_RUN_NAME}"
    "postprocess_sim_envs.route_length=${POST_ROUTE_LENGTH}"
    "postprocess_sim_envs.pre_path=${PRE_PATH}"
    "postprocess_sim_envs.post_path=${POST_PATH}"
  )

  run_cmd python3 tools/preprocess/postprocess_sim_envs.py \
    "${PRE_ARGS[@]}" \
    "${POST_EXTRA_ARGS[@]}"

  MODE_ARGS=(
    "sim.dataset_path=${POST_PATH}"
  )
fi

run_cmd python3 tools/simulate_parallel.py \
  "${COMMON_ARGS[@]}" \
  "${MODE_ARGS[@]}" \
  "${MODEL_ARGS[@]}" \
  "${SIM_EXTRA_ARGS[@]}"
<!--
Replace the placeholder links below with your official URLs:
- Project page
- arXiv
- Hugging Face
- Demo video
-->

<h1 align="center">VectorWorld: Efficient Streaming World Model via Diffusion Flow on Vector Graphs</h1>

<p align="center">
  Official implementation of <b>VectorWorld</b>, a streaming and fully vectorized world model for autonomous driving simulation.
</p>

<p align="center">
  <a href="https://your-project-page.example.com">
    <img src="https://img.shields.io/badge/Project-Page-2D7FF9?style=flat-square&logo=googlechrome&logoColor=white" alt="Project Page"/>
  </a>
  <a href="https://arxiv.org/abs/2501.00000">
    <img src="https://img.shields.io/badge/arXiv-Paper-B31B1B?style=flat-square&logo=arxiv&logoColor=white" alt="arXiv"/>
  </a>
  <a href="https://huggingface.co/datasets/Jck1998/vectorworld">
    <img src="https://img.shields.io/badge/Hugging%20Face-Checkpoint-FFD21E?style=flat-square&logo=huggingface&logoColor=black" alt="Hugging Face"/>
  </a>
  <a href="assets\mp4\ken_burns_mosaic.mp4">
    <img src="https://img.shields.io/badge/Demo-Video-FF4B4B?style=flat-square&logo=youtube&logoColor=white" alt="Demo Video"/>
  </a>
</p>

<p align="center">
  <a href="assets\mp4\ken_burns_mosaic.mp4">
    <img src="assets\img\overview.png" alt="VectorWorld Teaser" width="100%"/>
  </a>
</p>

<p align="center">
  <em>Click the teaser to watch the demo video.</em>
</p>

---

## Overview

VectorWorld is a streaming, fully vectorized world model for autonomous-driving simulation that incrementally generates ego-centric 64m × 64m lane-agent graph tiles during rollout, avoiding the mismatch of history-free initialization and the latency of multi-step sampling. By combining a motion-aware gated VAE, a one-step edge-gated relational generator, and DeltaSim for physics-aligned NPC control, it improves initialization validity and map-structure fidelity while enabling stable, real-time 1km+ closed-loop rollouts on Waymo Open Motion and nuPlan.

> **VectorWorld** represents driving scenes as structured vector graphs — lanes, agents, and motion codes — and supports closed-loop NPC simulation with a hybrid discrete-continuous behavior policy.

---

## Team

| Name | Affiliation | Google Scholar |
|---|---|---|
| **Chaokang Jiang** | Bosch XC | [Profile](https://scholar.google.com/citations?user=6gZ8vloAAAAJ) |
| **Deshen Zhou** | Bosch XC | [Profile](https://scholar.google.com/citations?user=Ux677B4AAAAJ&hl=en) |
| **Jiuming Liu** | University of Cambridge | [Profile](https://scholar.google.com/citations?user=j4YXCukAAAAJ) |
| **Kevin Li Sun** | Bosch XC | [Profile](https://scholar.google.com/citations?user=JZViN_4AAAAJ) |

---

## Architecture

| Component | Description |
|---|---|
| **VAE** | Motion-aware factorized graph autoencoder that encodes vectorized scenes into compact, policy-compatible latents |
| **EGR-DiT** | Edge-Gated Relational Diffusion Transformer for latent scene generation via MeanFlow / Flow / Diffusion |
| **DeltaSim** | Hybrid NPC behavior model with discrete anchor actions and continuous residual refinement for stable closed-loop rollout |

---

## Workflow

VectorWorld follows a simple and modular pipeline:

1. **Extract raw driving logs** from Waymo Open Motion or nuPlan.
2. **Preprocess vector-graph scenes** for representation learning and simulation.
3. **Train the VAE** to obtain compact scene latents.
4. **Cache latents** and train **EGR-DiT** with MeanFlow / Flow / Diffusion.
5. **Train DeltaSim** for non-ego agent behavior.
6. **Generate streaming simulation environments** and run offline or online closed-loop evaluation.

---

## News

- **[YYYY/MM/DD]** Paper released on [arXiv](https://arxiv.org/abs/2501.00000).
- **[YYYY/MM/DD]** Initial code release.
- **[YYYY/MM/DD]** Project page is live.
- **[2026/03/18]** Model checkpoints released on [Hugging Face](https://huggingface.co/datasets/Jck1998/vectorworld).

---

## Installation

```bash
git clone https://github.com/your-user/vectorworld.git
cd vectorworld

conda create -n vectorworld python=3.10 -y
conda activate vectorworld

# Install PyTorch / PyG according to your CUDA version
pip install torch torchvision
pip install torch-geometric torch-scatter torch-sparse

# Core dependencies
pip install pytorch-lightning hydra-core omegaconf torch-ema
pip install imageio-ffmpeg matplotlib scipy shapely networkx tqdm

# Environment variables
export SCRATCH_ROOT=/path/to/your/scratch
source scripts/define_env_variables.sh
```

> **Note:** Please install the PyTorch Geometric packages that match your local CUDA / PyTorch version.

---

## Pretrained Checkpoints

Released checkpoints can be downloaded from [Hugging Face](https://huggingface.co/datasets/Jck1998/vectorworld) and placed under:

```text
metadata/checkpoints/
├── waymo
│   ├── vae
│   │   └── last.ckpt
│   ├── ldm
│   │   ├── diffusion
│   │   │   └── last.ckpt
│   │   ├── flow
│   │   │   └── last.ckpt
│   │   └── meanflow
│   │       └── last.ckpt
│   └── delta_sim
│       └── last.ckpt
└── nuplan
    ├── vae
    │   └── last.ckpt
    └── ldm
        ├── diffusion
        │   └── last.ckpt
        ├── flow
        │   └── last.ckpt
        └── meanflow
            └── last.ckpt
```

---

## Datasets

VectorWorld currently supports:

- **Waymo Open Motion Dataset v1.1.0**
- **nuPlan**

### Raw-data conversion

Before VectorWorld preprocessing, the raw datasets should first be converted into extracted scenario files. We follow the Scenario Dreamer-style preprocessing pipeline:

- **Waymo conversion script:** [generate_waymo_dataset.py](https://github.com/princeton-computational-imaging/scenario-dreamer/blob/main/data_processing/waymo/generate_waymo_dataset.py)
- **nuPlan conversion script:** [generate_nuplan_dataset.py](https://github.com/princeton-computational-imaging/scenario-dreamer/blob/main/data_processing/nuplan/generate_nuplan_dataset.py)

In this repository, the corresponding extraction entry points are:

- `scripts/extract_waymo_data.sh`
- `scripts/extract_nuplan_data.sh`

### Repository preprocessing scripts

The main preprocessing scripts are:

- `scripts/preprocess_waymo.sh`
- `scripts/preprocess_nuplan.sh`
- `scripts/preprocess_deltasim_waymo.sh`

These scripts wrap the preprocessing logic in `tools/preprocess/` and can be adapted for different splits, shards, and storage paths.

---

## Quick Start

The minimal commands below use the Python entry points directly for clarity and reproducibility. Convenience recipes are also provided in `scripts/`.

### 1. Prepare data

```bash
# Extract raw Waymo scenarios
bash scripts/extract_waymo_data.sh

# Extract raw nuPlan scenarios
bash scripts/extract_nuplan_data.sh

# Preprocess vector-graph data for VAE / LDM
bash scripts/preprocess_waymo.sh
bash scripts/preprocess_nuplan.sh

# Preprocess Waymo data for DeltaSim
bash scripts/preprocess_deltasim_waymo.sh
```

> **Tip:** The shell scripts are templates. For public experiments, you may further customize split names, shard ids, and output directories via Hydra or by editing the scripts.

### 2. Train the VAE

```bash
python3 tools/train.py \
  dataset_name=waymo \
  model_name=vae \
  ae.dataset.preprocess=true \
  ae.dataset.preprocess_dir=metadata/datasets/waymo/sd_ae_motion_preprocess \
  ae.train.devices=1 \
  ae.train.max_steps=85000 \
  ae.train.run_name=vectorworld_vae_waymo
```

### 3. Cache latents for EGR-DiT training

```bash
# Cache train split
python3 tools/generate.py \
  dataset_name=waymo \
  model_name=vae \
  ae.eval.run_name=vectorworld_vae_waymo \
  ae.eval.ckpt_path=outputs/checkpoints/vectorworld_vae_waymo/last.ckpt \
  ae.eval.split_name=train \
  ae.eval.batch_size=64 \
  ae.eval.cache_latents.enable_caching=true \
  ae.eval.cache_latents.split_name=train \
  ae.eval.cache_latents.latent_dir=metadata/datasets/waymo/vae_latents

# Cache validation split
python3 tools/generate.py \
  dataset_name=waymo \
  model_name=vae \
  ae.eval.run_name=vectorworld_vae_waymo \
  ae.eval.ckpt_path=outputs/checkpoints/vectorworld_vae_waymo/last.ckpt \
  ae.eval.split_name=val \
  ae.eval.batch_size=64 \
  ae.eval.cache_latents.enable_caching=true \
  ae.eval.cache_latents.split_name=val \
  ae.eval.cache_latents.latent_dir=metadata/datasets/waymo/vae_latents
```

### 4. Train the latent generator (EGR-DiT)

#### MeanFlow

```bash
python3 tools/train.py \
  dataset_name=waymo \
  model_name=ldm \
  ldm.model.ldm_type=meanflow \
  ldm.model.autoencoder_path=outputs/checkpoints/vectorworld_vae_waymo/last.ckpt \
  ldm.train.devices=1 \
  ldm.train.max_steps=165000 \
  ldm.train.run_name=vectorworld_meanflow_waymo
```

#### Flow / Diffusion

```bash
python3 tools/train.py \
  dataset_name=waymo \
  model_name=ldm \
  ldm.model.ldm_type=flow \
  ldm.model.autoencoder_path=outputs/checkpoints/vectorworld_vae_waymo/last.ckpt \
  ldm.train.devices=1 \
  ldm.train.max_steps=165000 \
  ldm.train.run_name=vectorworld_flow_waymo \
  ldm.model.use_rel_bias=true \
  ldm.model.use_gcf=true \
  ldm.model.qk_norm=true \
  ldm.model.attn_logit_clip=30.0 \
  ldm.model.lane_rel_dim=64 \
  ldm.model.agent_rel_dim=32 \
  ldm.model.edge_dim=32 \
  ldm.model.use_cross_rel_bias=true \
  ldm.model.use_rel_gate=true \
  ldm.model.gcf_var_scale=0.15
```

> Replace `flow` with `diffusion` if desired.  
> See `scripts/train_egr_dit.sh` for the full training recipe.

### 5. Generate scenes with the trained LDM

#### Initial-scene generation

```bash
python3 tools/generate.py \
  dataset_name=waymo \
  model_name=ldm \
  ldm.model.ldm_type=meanflow \
  ldm.model.autoencoder_path=outputs/checkpoints/vectorworld_vae_waymo/last.ckpt \
  ldm.eval.ckpt_path=outputs/checkpoints/vectorworld_meanflow_waymo/last.ckpt \
  ldm.eval.run_name=vectorworld_meanflow_waymo \
  ldm.eval.mode=initial_scene \
  ldm.eval.num_samples=100 \
  ldm.eval.batch_size=16 \
  ldm.eval.meanflow_num_steps=3 \
  ldm.eval.visualize=true
```

#### Streaming simulation-environment generation

```bash
python3 tools/generate.py \
  dataset_name=waymo \
  model_name=ldm \
  ldm.model.ldm_type=flow \
  ldm.model.autoencoder_path=outputs/checkpoints/vectorworld_vae_waymo/last.ckpt \
  ldm.eval.ckpt_path=outputs/checkpoints/vectorworld_flow_waymo/last.ckpt \
  ldm.eval.run_name=vectorworld_flow_waymo \
  ldm.eval.mode=simulation_environments \
  ldm.eval.num_samples=10 \
  ldm.eval.sim_envs.route_length=200 \
  ldm.eval.sim_envs.overhead_factor=8 \
  ldm.eval.sim_envs.num_inpainting_candidates=10 \
  ldm.eval.sim_envs.nocturne_compatible_only=false \
  ldm.eval.visualize=true
```

> See `scripts/infer_egr_dit.sh` for additional generation recipes.

### 6. Train DeltaSim

```bash
python3 tools/train.py \
  dataset_name=waymo \
  model_name=deltasim \
  deltasim.dataset.preprocess=true \
  deltasim.dataset.preprocess_dir=metadata/datasets/waymo/deltasim \
  deltasim.train.devices=1 \
  deltasim.train.max_steps=100000 \
  deltasim.train.run_name=vectorworld_deltasim \
  deltasim.model.dkal.enabled=true \
  deltasim.model.residual_refine.enabled=true \
  deltasim.model.phys_prior.enabled=true
```

> See `scripts/train_deltasim.sh` for the full training configuration with RTG conditioning, residual refinement, and physics priors.

### 7. Closed-loop simulation

#### Offline simulation

```bash
bash scripts/run_simulation_parallel.sh \
  sim=base \
  sim.mode=vectorworld \
  ldm.model.ldm_type=flow \
  ldm.eval.run_name=vectorworld_flow_offline \
  postprocess_sim_envs.run_name=vectorworld_flow_waymo
```

#### Online parallel simulation

```bash
bash scripts/run_simulation_parallel.sh \
  sim=online \
  sim.mode=vectorworld_online \
  ldm.model.ldm_type=meanflow \
  ldm.eval.run_name=vectorworld_meanflow_online
```

The simulation script automatically configures the correct generator settings for `meanflow`, `flow`, and `diffusion`, and uses DeltaSim as the default NPC behavior model.

---

## Configuration

VectorWorld uses [Hydra](https://hydra.cc/) for composable experiment management. Almost every option can be overridden directly from the command line.

Example:

```bash
python3 tools/generate.py \
  dataset_name=waymo \
  model_name=ldm \
  ldm.model.ldm_type=meanflow \
  ldm.eval.mode=simulation_environments \
  ldm.eval.num_samples=32 \
  ldm.eval.sim_envs.route_length=500
```

Useful config knobs:

- `ldm.model.ldm_type`: `meanflow`, `flow`, `diffusion`
- `ldm.eval.mode`: `initial_scene`, `simulation_environments`
- `sim`: `base`, `online`
- `sim.num_workers`: number of simulation workers
- `ldm.eval.sim_envs.route_length`: target rollout route length
- `ae.eval.cache_latents.*`: latent caching options

---

## Repository Structure

```text
vectorworld/
├── assets/                  # Teaser figures, videos, README media
├── configs/                 # Hydra configuration files
├── scripts/                 # Convenience shell recipes
├── tools/                   # Python entry points for train / generate / preprocess / simulate
├── vectorworld/             # Core package
│   ├── models/              # Lightning modules: VAE, EGR-DiT, DeltaSim
│   ├── networks/            # Backbone architectures and heads
│   ├── data/                # Datasets and datamodules
│   ├── simulation/          # Closed-loop simulator and policies
│   └── utils/               # Geometry, visualization, and helper functions
└── metadata/                # Checkpoints, latent stats, logs, and processed assets
```

---

## Acknowledgements

This project is inspired by and builds upon several excellent open-source efforts:

- [SLEDGE](https://github.com/autonomousvision/sledge/tree/main)
- [Scenario Dreamer](https://github.com/princeton-computational-imaging/scenario-dreamer)
- [MeanFlow](https://github.com/haidog-yaqub/MeanFlow)

We thank the authors of these projects for making their work publicly available.

---

## Citation

If you find this repository useful, please consider starring the repo and citing our work:

```bibtex
@article{jiang2025vectorworld,
  title   = {VectorWorld: Efficient Streaming World Model via Diffusion Flow on Vector Graphs},
  author  = {Chaokang Jiang and Deshen Zhou and Jiuming Liu and Kevin Li Sun},
  year    = {2025}
}
```
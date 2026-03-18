<!-- ======================================================================
     VectorWorld — README
     Replace all placeholder URLs (marked with TODO) before public release.
     ====================================================================== -->

<p align="center">
  <img src="assets/img/VectorWorld-icon.png" alt="VectorWorld" width="100%"/>
</p>

<h1 align="center">VectorWorld: Efficient Streaming World Model via<br>Diffusion Flow on Vector Graphs</h1>

<p align="center">
  <a href="https://scholar.google.com/citations?user=6gZ8vloAAAAJ"><b>Chaokang Jiang</b></a><sup>1</sup>&nbsp;&nbsp;
  <a href="https://scholar.google.com/citations?user=Ux677B4AAAAJ&hl=en"><b>Deshen Zhou</b></a><sup>1</sup>&nbsp;&nbsp;
  <a href="https://scholar.google.com/citations?user=j4YXCukAAAAJ"><b>Jiuming Liu</b></a><sup>2</sup>&nbsp;&nbsp;
  <a href="https://scholar.google.com/citations?user=JZViN_4AAAAJ"><b>Kevin Li Sun</b></a><sup>1</sup>
</p>

<p align="center">
  <sup>1</sup>Bosch XC&nbsp;&nbsp;&nbsp;&nbsp;<sup>2</sup>University of Cambridge
</p>

<!-- TODO: replace placeholder URLs before public release -->
<p align="center">
  <a href="https://jiangchaokang.github.io/VectorWorld">
    <img src="https://img.shields.io/badge/Project-Page-2D7FF9?style=flat-square&logo=googlechrome&logoColor=white" alt="Project Page"/>
  </a>
  <a href="https://arxiv.org/abs/2501.00000">
    <img src="https://img.shields.io/badge/arXiv-Paper-B31B1B?style=flat-square&logo=arxiv&logoColor=white" alt="arXiv"/>
  </a>
  <a href="https://huggingface.co/datasets/Jck1998/vectorworld">
    <img src="https://img.shields.io/badge/Hugging%20Face-Checkpoints-FFD21E?style=flat-square&logo=huggingface&logoColor=black" alt="Hugging Face"/>
  </a>
  <a href="assets/mp4/demo.gif">
    <img src="https://img.shields.io/badge/Demo-Video-FF4B4B?style=flat-square&logo=youtube&logoColor=white" alt="Demo Video"/>
  </a>
</p>

<p align="center">
  <img src="assets/mp4/demo.gif" alt="VectorWorld Demo" width="100%"/>
</p>

---

## Overview

**VectorWorld** is a streaming, fully vectorized world model for closed-loop autonomous driving simulation. It incrementally outpaints ego-centric **64 m × 64 m** lane–agent graph tiles during rollout, enabling history-conditioned policies to interact beyond recorded horizons while preserving structured map–agent relations.

The system combines three core components: a **motion-aware gated VAE** for policy-compatible warm starts, an **edge-gated relational DiT (EGR-DiT)** with **interval-conditioned MeanFlow** and **JVP-based supervision** for solver-free one-step generation, and **DeltaSim (∆Sim)** for physics-aligned NPC control. On [Waymo Open Motion](https://waymo.com/open/) and [nuPlan](https://www.nuscenes.org/nuplan), VectorWorld achieves stable **1 km+** closed-loop rollouts at **~6 ms per tile** with improved initialization validity and map fidelity.

---

## Highlights

- 🚗 **Streaming vector generation** — raster-free km-scale closed-loop simulation
- 🧠 **Motion-aware warm starts** — reduced cold-start mismatch
- 🧩 **Relation-aware generation** — EGR-DiT preserves lane topology and lane–agent consistency
- ⚡ **Real-time one-step inference** — solver-free completion at **~6 ms per tile**
- 🔁 **Physics-aligned control** — ∆Sim uses hybrid actions for stable multi-agent rollout
- 📊 **Strong performance** — 1 km+ rollouts and 56.0% stress-test success

---

## News

| Date | Update |
|:---|:---|
| **2026/03/16** | Initial code release. |
| **2026/03/18** | Model checkpoints released on [Hugging Face](https://huggingface.co/datasets/Jck1998/vectorworld). |
| **2026/03/18** | Project page is live. [Project page](https://jiangchaokang.github.io/VectorWorld). |
| **2026/03/20** | Paper released on [arXiv](https://arxiv.org/abs/2501.00000). |

---

## Table of Contents

- [Architecture](#architecture)
- [Installation](#installation)
- [Pretrained Checkpoints](#pretrained-checkpoints)
- [Datasets](#datasets)
- [Quick Start](#quick-start)
- [Configuration](#configuration)
- [Repository Structure](#repository-structure)
- [Acknowledgements](#acknowledgements)
- [Citation](#citation)
- [License](#license)

---

## Architecture

| Component | Description |
|:---|:---|
| **VAE** | Motion-aware factorized graph autoencoder that encodes vectorized scenes into compact, policy-compatible latents. |
| **EGR-DiT** | Edge-Gated Relational Diffusion Transformer for latent scene generation via MeanFlow, Flow, or Diffusion. |
| **DeltaSim** | Hybrid NPC behavior model with discrete anchor actions and continuous residual refinement for stable closed-loop rollout. |

---

## Installation

```bash
git clone https://github.com/your-user/vectorworld.git
cd vectorworld

conda create -n vectorworld python=3.10 -y
conda activate vectorworld

# Install PyTorch and PyG (adjust for your CUDA version)
pip install torch torchvision
pip install torch-geometric torch-scatter torch-sparse

# Core dependencies
pip install pytorch-lightning hydra-core omegaconf torch-ema
pip install imageio-ffmpeg matplotlib scipy shapely networkx tqdm

# Environment variables
export SCRATCH_ROOT=/path/to/your/scratch
source scripts/define_env_variables.sh
```

> **Note:** Please install the [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/) packages that match your local CUDA and PyTorch version.

---

## Pretrained Checkpoints

Released checkpoints can be downloaded from [Hugging Face](https://huggingface.co/datasets/Jck1998/vectorworld) and placed under:

```
metadata/checkpoints/
├── waymo
│   ├── vae/last.ckpt
│   ├── ldm
│   │   ├── diffusion/last.ckpt
│   │   ├── flow/last.ckpt
│   │   └── meanflow/last.ckpt
│   └── delta_sim/last.ckpt
└── nuplan
    ├── vae/last.ckpt
    └── ldm
        ├── diffusion/last.ckpt
        ├── flow/last.ckpt
        └── meanflow/last.ckpt
```

---

## Datasets

VectorWorld currently supports **Waymo Open Motion Dataset v1.1.0** and **nuPlan**.

### Raw-data conversion

Before VectorWorld preprocessing, the raw datasets should first be converted into extracted scenario files. We follow the [Scenario Dreamer](https://github.com/princeton-computational-imaging/scenario-dreamer)-style preprocessing pipeline:

| Dataset | Conversion script |
|:---|:---|
| Waymo | [`generate_waymo_dataset.py`](https://github.com/princeton-computational-imaging/scenario-dreamer/blob/main/data_processing/waymo/generate_waymo_dataset.py) |
| nuPlan | [`generate_nuplan_dataset.py`](https://github.com/princeton-computational-imaging/scenario-dreamer/blob/main/data_processing/nuplan/generate_nuplan_dataset.py) |

The corresponding extraction entry points in this repository are `scripts/extract_waymo_data.sh` and `scripts/extract_nuplan_data.sh`.

### Preprocessing

The main preprocessing scripts are `scripts/preprocess_waymo.sh`, `scripts/preprocess_nuplan.sh`, and `scripts/preprocess_deltasim_waymo.sh`. These scripts wrap the logic in `tools/preprocess/` and can be adapted for different splits, shards, and storage paths.

---

## Quick Start

The commands below use Python entry points directly for clarity and reproducibility. Convenience shell recipes are also provided in `scripts/`.

### Step 1 — Prepare data

```bash
# Extract raw scenarios
bash scripts/extract_waymo_data.sh
bash scripts/extract_nuplan_data.sh

# Preprocess vector-graph data
bash scripts/preprocess_waymo.sh
bash scripts/preprocess_nuplan.sh
bash scripts/preprocess_deltasim_waymo.sh
```

> **Tip:** The shell scripts are templates. Customize split names, shard IDs, and output directories via [Hydra](https://hydra.cc/) overrides or by editing the scripts directly.

### Step 2 — Train the VAE

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

### Step 3 — Cache latents for EGR-DiT training

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

### Step 4 — Train the latent generator (EGR-DiT)

**MeanFlow**

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

**Flow / Diffusion**

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

> Replace `flow` with `diffusion` if desired. See `scripts/train_egr_dit.sh` for the full training recipe.

### Step 5 — Generate scenes

**Initial-scene generation**

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

**Streaming simulation-environment generation**

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

### Step 6 — Train DeltaSim

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

> See `scripts/train_deltasim.sh` for the full configuration with RTG conditioning, residual refinement, and physics priors.

### Step 7 — Closed-loop simulation

**Offline simulation**

```bash
bash scripts/run_simulation_parallel.sh \
  sim=base \
  sim.mode=vectorworld \
  ldm.model.ldm_type=flow \
  ldm.eval.run_name=vectorworld_flow_offline \
  postprocess_sim_envs.run_name=vectorworld_flow_waymo
```

**Online parallel simulation**

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

VectorWorld uses [Hydra](https://hydra.cc/) for composable experiment management. Almost every option can be overridden directly from the command line:

```bash
python3 tools/generate.py \
  dataset_name=waymo \
  model_name=ldm \
  ldm.model.ldm_type=meanflow \
  ldm.eval.mode=simulation_environments \
  ldm.eval.num_samples=32 \
  ldm.eval.sim_envs.route_length=500
```

Frequently used configuration knobs include `ldm.model.ldm_type` (`meanflow` | `flow` | `diffusion`), `ldm.eval.mode` (`initial_scene` | `simulation_environments`), `sim` (`base` | `online`), `sim.num_workers`, `ldm.eval.sim_envs.route_length`, and `ae.eval.cache_latents.*`.

---

## Repository Structure

```
vectorworld/
├── assets/                  # Figures, videos, and README media
├── configs/                 # Hydra configuration files
├── scripts/                 # Convenience shell recipes
├── tools/                   # Entry points: train / generate / preprocess / simulate
├── vectorworld/             # Core package
│   ├── models/              #   Lightning modules — VAE, EGR-DiT, DeltaSim
│   ├── networks/            #   Backbone architectures and heads
│   ├── data/                #   Datasets and data modules
│   ├── simulation/          #   Closed-loop simulator and policies
│   └── utils/               #   Geometry, visualization, helpers
└── metadata/                # Checkpoints, latent stats, logs, processed assets
```

---

## Acknowledgements

This project is inspired by and builds upon several excellent open-source efforts: [SLEDGE](https://github.com/autonomousvision/sledge) (ECCV 2024), [Scenario Dreamer](https://github.com/princeton-computational-imaging/scenario-dreamer), and [MeanFlow](https://github.com/haidog-yaqub/MeanFlow). We thank the authors for making their work publicly available.

---

## Citation

If you find this work useful, please consider citing:

```bibtex
@article{jiang2025vectorworld,
  title   = {VectorWorld: Efficient Streaming World Model via Diffusion Flow on Vector Graphs},
  author  = {Jiang, Chaokang and Zhou, Deshen and Liu, Jiuming and Sun, Kevin Li},
  year    = {2025}
}
```

---

## License

This project is released under the [Apache 2.0 License](LICENSE).

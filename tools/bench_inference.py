from __future__ import annotations

import os
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import hydra
import torch
from omegaconf import OmegaConf

from torch_geometric.loader import DataLoader

from configs.config import CONFIG_PATH, PARTITIONED, NON_PARTITIONED
from vectorworld.utils.train_helpers import set_latent_stats
from vectorworld.utils.bench_utils import (
    benchmark_forward,
    count_parameters,
    ensure_dir,
    get_env_meta,
    resolve_path,
    save_json,
    summarize_ms,
)

from models.scenario_dreamer_ldm import ScenarioDreamerLDM
from models.scenario_dreamer_autoencoder import ScenarioDreamerAutoEncoder
from models.ctrl_sim import CtRLSim
from vectorworld.utils.data_container import CtRLSimData


# -----------------------------------------------------------------------------
# Defaults (no need to touch configs if you don't want)
# -----------------------------------------------------------------------------
_DEFAULT_EVAL_BENCH = {
    "enabled": False,
    # target: ldm_latent | ldm_decode | ae_forward | ae_encoder | ae_decoder | ctrl_sim_forward
    "target": None,
    # LDM-specific: initial_scene | inpainting (synthetic mask) | lane_conditioned
    "ldm_mode": "initial_scene",
    # how many scenes to build as input pool
    "num_scenes": 256,
    # dataloader batch size (scenes per batch)
    "batch_size": 16,
    # repeat timing on each batch to reduce noise
    "num_warmup": 10,
    "num_iters": 50,
    # how many distinct batches to benchmark (<= ceil(num_scenes/batch_size))
    "num_batches": 8,
    # if true, empty CUDA cache between batches (usually keep False)
    "clear_cuda_cache_between_batches": False,
    "amp": {"enabled": True, "dtype": "bf16"},
    # optional explicit save path; if null => auto
    "save_path": None,
    # print per-batch stats
    "verbose": True,
}

_DEFAULT_SIM_BENCH = {
    "enabled": False,
    "target": "ctrl_sim_forward",
    "batch_size": 8,  # number of CtRLSim "motion_datas" stacked
    "num_warmup": 20,
    "num_iters": 200,
    "amp": {"enabled": True, "dtype": "bf16"},
    "save_path": None,
    "verbose": True,
}


def _project_root() -> str:
    # this file lives in <project_root>/bench_inference.py
    return str(Path(__file__).resolve().parent)


def _get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda", 0)
    return torch.device("cpu")


def _maybe_get(node, key: str, default):
    try:
        return getattr(node, key)
    except Exception:
        return default


def _get_bench_cfg(cfg, path: str, default_dict: Dict[str, Any]) -> Dict[str, Any]:
    node = OmegaConf.select(cfg, path)
    if node is None:
        return dict(default_dict)
    # merge defaults (so old configs won't break)
    merged = dict(default_dict)
    merged.update(OmegaConf.to_container(node, resolve=True))
    # nested amp
    if "amp" in default_dict:
        amp_def = dict(default_dict["amp"])
        amp_def.update(merged.get("amp", {}) or {})
        merged["amp"] = amp_def
    return merged


def _auto_save_path_for_eval(cfg_eval, kind: str, target: str) -> str:
    # put under <save_dir>/<run_name>/bench/
    save_dir = resolve_path(str(cfg_eval.save_dir), project_root=_project_root())
    run_name = str(cfg_eval.run_name) if cfg_eval.run_name is not None else "unnamed"
    out_dir = Path(save_dir) / run_name / "bench"
    ensure_dir(out_dir)
    return (out_dir / f"bench_{kind}_{target}.json").as_posix()


def _auto_save_path_for_sim(cfg, target: str) -> str:
    # try to use behaviour_model.run_name for grouping
    run_name = str(getattr(cfg.sim.behaviour_model, "run_name", "behaviour"))
    save_dir = resolve_path("outputs/checkpoints", project_root=_project_root())
    out_dir = Path(save_dir) / run_name / "bench"
    ensure_dir(out_dir)
    return (out_dir / f"bench_sim_{target}.json").as_posix()


def _graph_batch_stats_scenario_dreamer(data) -> Dict[str, int]:
    # HeteroData, already batched
    stats = {
        "batch_size": int(getattr(data, "batch_size", -1)),
        "num_agents_total": int(data["agent"].x.shape[0]) if hasattr(data["agent"], "x") else 0,
        "num_lanes_total": int(data["lane"].x.shape[0]) if hasattr(data["lane"], "x") else 0,
        "num_l2l_edges": int(data["lane", "to", "lane"].edge_index.shape[1]),
        "num_a2a_edges": int(data["agent", "to", "agent"].edge_index.shape[1]),
        "num_l2a_edges": int(data["lane", "to", "agent"].edge_index.shape[1]),
    }
    stats["num_tokens_total"] = stats["num_agents_total"] + stats["num_lanes_total"]
    return stats


def _make_ldm_input_batches(
    model: ScenarioDreamerLDM,
    dataset_name: str,
    mode: str,
    num_scenes: int,
    batch_size: int,
    num_batches: int,
    device: torch.device,
) -> List[Any]:
    """Create up to num_batches batches of ScenarioDreamerData on GPU (no timing here)."""
    mode = str(mode).lower()
    if num_scenes <= 0:
        raise ValueError("num_scenes must be > 0")
    if batch_size <= 0:
        raise ValueError("batch_size must be > 0")
    if num_batches <= 0:
        raise ValueError("num_batches must be > 0")

    # ensure enough scenes to cover requested num_batches
    num_scenes = max(num_scenes, num_batches * batch_size)

    # initial_scene & lane_conditioned can use internal initializer
    if mode in ("initial_scene", "lane_conditioned"):
        data_list, _ = model._initialize_pyg_dset(
            mode=mode,
            num_samples=num_scenes,
            batch_size=batch_size,
            conditioning_path=None if mode == "initial_scene" else model.cfg.eval.conditioning_path,
            nocturne_compatible_only=False if dataset_name == "nuplan" else bool(model.cfg.eval.sim_envs.nocturne_compatible_only),
            conditioning_scenes=None,
        )
    elif mode == "inpainting":
        # Synthetic inpainting batches:
        # We reuse initial_scene structure but add mask+latents; lg_type=PARTITIONED.
        data_list, _ = model._initialize_pyg_dset(
            mode="initial_scene",
            num_samples=num_scenes,
            batch_size=batch_size,
            conditioning_path=None,
            nocturne_compatible_only=False,
            conditioning_scenes=None,
        )
        lane_latent_dim = int(model.cfg.model.lane_latent_dim)
        agent_latent_dim = int(model.cfg.model.agent_latent_dim)

        # Simple, deterministic conditioning fraction
        cond_frac_lane = float(_maybe_get(model.cfg.eval, "bench_cond_frac_lane", 0.5))
        cond_frac_agent = float(_maybe_get(model.cfg.eval, "bench_cond_frac_agent", 0.5))
        cond_frac_lane = min(max(cond_frac_lane, 0.0), 1.0)
        cond_frac_agent = min(max(cond_frac_agent, 0.0), 1.0)

        for d in data_list:
            # mark as partitioned (important: affects DiT scene embedding)
            d["lg_type"] = int(PARTITIONED)

            n_lanes = int(d["num_lanes"])
            n_agents = int(d["num_agents"])

            n_cond_l = max(1, int(math.ceil(n_lanes * cond_frac_lane)))
            n_cond_a = max(1, int(math.ceil(n_agents * cond_frac_agent)))

            lane_mask = torch.zeros(n_lanes, dtype=torch.bool)
            agent_mask = torch.zeros(n_agents, dtype=torch.bool)
            lane_mask[:n_cond_l] = True
            agent_mask[:n_cond_a] = True

            d["lane"].mask = lane_mask
            d["agent"].mask = agent_mask

            # conditioned latents (only used where mask==True; but we fill all for simplicity)
            d["lane"].latents = torch.randn(n_lanes, lane_latent_dim, dtype=torch.float32)
            d["agent"].latents = torch.randn(n_agents, agent_latent_dim, dtype=torch.float32)

    else:
        raise ValueError(f"Unsupported ldm_mode='{mode}'")

    dl = DataLoader(data_list, batch_size=batch_size, shuffle=False, drop_last=False)
    batches: List[Any] = []
    for b, data in enumerate(dl):
        if b >= num_batches:
            break
        batches.append(data.to(device))
    if len(batches) == 0:
        raise RuntimeError("Failed to build any LDM input batches.")
    return batches


def _bench_ldm(
    cfg,
    dataset_name: str,
    bench_cfg: Dict[str, Any],
    device: torch.device,
) -> Dict[str, Any]:
    target = str(bench_cfg["target"])
    mode = str(bench_cfg["ldm_mode"]).lower()

    # Resolve ckpt path under Hydra chdir
    ckpt_path = resolve_path(str(cfg.ldm.eval.ckpt_path), project_root=_project_root())
    assert ckpt_path is not None and os.path.exists(ckpt_path), f"[bench] invalid ldm.eval.ckpt_path: {ckpt_path}"

    # Thread dataset name into sub-configs (same as eval.py)
    ldm_cfg = cfg.ldm
    ae_cfg = cfg.ae
    OmegaConf.set_struct(ldm_cfg, False)
    OmegaConf.set_struct(ae_cfg, False)
    ldm_cfg.dataset_name = dataset_name
    ae_cfg.dataset_name = dataset_name
    OmegaConf.set_struct(ldm_cfg, True)
    OmegaConf.set_struct(ae_cfg, True)

    # latent stats required for un/normalize
    ldm_cfg = set_latent_stats(ldm_cfg)

    print(f"[bench][LDM] Loading ckpt: {ckpt_path}")
    model = ScenarioDreamerLDM.load_from_checkpoint(
        ckpt_path, cfg=ldm_cfg, cfg_ae=ae_cfg, map_location="cpu"
    ).to(device)
    model.eval()

    # MeanFlow: respect eval.meanflow_num_steps (same behavior as generate())
    ldm_type = str(getattr(model.cfg.model, "ldm_type", "diffusion")).lower()
    if ldm_type in ("meanflow", "mf") and hasattr(model.gen_model, "set_num_steps_eval"):
        n_steps = getattr(model.cfg.eval, "meanflow_num_steps", None)
        if n_steps is not None:
            model.gen_model.set_num_steps_eval(int(n_steps))

    batches = _make_ldm_input_batches(
        model=model,
        dataset_name=dataset_name,
        mode=mode,
        num_scenes=int(bench_cfg["num_scenes"]),
        batch_size=int(bench_cfg["batch_size"]),
        num_batches=int(bench_cfg["num_batches"]),
        device=device,
    )

    amp_enabled = bool(bench_cfg["amp"]["enabled"])
    amp_dtype = str(bench_cfg["amp"]["dtype"])

    # Parameters
    params_gen = count_parameters(model.gen_model, trainable_only=False)
    params_ae = count_parameters(model.autoencoder.model, trainable_only=False)
    params_total = params_gen + params_ae

    results_per_batch = []
    all_times = []

    clear_cache = bool(bench_cfg["clear_cuda_cache_between_batches"])
    num_warmup = int(bench_cfg["num_warmup"])
    num_iters = int(bench_cfg["num_iters"])

    with model.ema.average_parameters():
        for bi, data in enumerate(batches):
            if clear_cache and device.type == "cuda":
                torch.cuda.empty_cache()

            # choose forward target
            if target == "ldm_latent":

                def _fn():
                    return model.gen_model.forward(data, mode=mode)

            elif target == "ldm_decode":

                def _fn():
                    agent_lat, lane_lat = model.gen_model.forward(data, mode=mode)
                    # unnormalize latents (DiT space -> AE latent space)
                    from vectorworld.utils.data_helpers import unnormalize_latents, unnormalize_scene

                    agent_lat, lane_lat = unnormalize_latents(
                        agent_lat,
                        lane_lat,
                        model.cfg.dataset.agent_latents_mean,
                        model.cfg.dataset.agent_latents_std,
                        model.cfg.dataset.lane_latents_mean,
                        model.cfg.dataset.lane_latents_std,
                    )
                    # decode to (static+motion) in normalized scene space
                    ae_model = model.autoencoder.model
                    agent_states_full_norm, lane_samples_norm, agent_types, lane_types, lane_conn = ae_model.forward_decoder_with_motion(
                        agent_lat, lane_lat, data
                    )
                    # unnormalize to physical space (what simulator actually needs)
                    _ = unnormalize_scene(
                        agent_states_full_norm,
                        lane_samples_norm,
                        fov=model.cfg.dataset.fov,
                        min_speed=model.cfg.dataset.min_speed,
                        max_speed=model.cfg.dataset.max_speed,
                        min_length=model.cfg.dataset.min_length,
                        max_length=model.cfg.dataset.max_length,
                        min_width=model.cfg.dataset.min_width,
                        max_width=model.cfg.dataset.max_width,
                        min_lane_x=model.cfg.dataset.min_lane_x,
                        min_lane_y=model.cfg.dataset.min_lane_y,
                        max_lane_x=model.cfg.dataset.max_lane_x,
                        max_lane_y=model.cfg.dataset.max_lane_y,
                    )
                    return None

            else:
                raise ValueError(f"[bench][LDM] Unknown target='{target}'")

            times_ms, mem = benchmark_forward(
                fn=_fn,
                device=device,
                num_warmup=num_warmup,
                num_iters=num_iters,
                amp_enabled=amp_enabled,
                amp_dtype=amp_dtype,
                measure_memory=True,
            )

            stats = _graph_batch_stats_scenario_dreamer(data)
            summary = summarize_ms(times_ms)

            # Throughput: scenes/sec based on mean batch time
            bs = int(stats["batch_size"])
            mean_s = float(summary["mean"]) / 1000.0
            throughput = (bs / mean_s) if (mean_s > 0 and bs > 0) else float("nan")
            ms_per_scene = float(summary["mean"]) / float(bs) if bs > 0 else float("nan")

            one = {
                "batch_index": int(bi),
                "ldm_mode": mode,
                "target": target,
                "amp": {"enabled": amp_enabled, "dtype": amp_dtype},
                "timing_ms": summary,
                "throughput_scenes_per_s": float(throughput),
                "ms_per_scene_mean": float(ms_per_scene),
                "graph": stats,
                "cuda_mem": {
                    "baseline_allocated_mb": mem.baseline_allocated_mb,
                    "baseline_reserved_mb": mem.baseline_reserved_mb,
                    "peak_allocated_mb": mem.peak_allocated_mb,
                    "peak_reserved_mb": mem.peak_reserved_mb,
                },
            }
            results_per_batch.append(one)
            all_times.extend(times_ms)

            if bool(bench_cfg["verbose"]):
                print(
                    f"[bench][LDM][{target}][{mode}] batch={bi} "
                    f"p50={one['timing_ms']['p50']:.2f}ms mean={one['timing_ms']['mean']:.2f}ms "
                    f"({one['ms_per_scene_mean']:.2f}ms/scene) "
                    f"peak_alloc={one['cuda_mem']['peak_allocated_mb']:.1f}MB"
                )

    overall = summarize_ms(all_times)
    overall_bs = int(bench_cfg["batch_size"])
    overall_mean_s = float(overall["mean"]) / 1000.0
    overall_throughput = (overall_bs / overall_mean_s) if (overall_mean_s > 0 and overall_bs > 0) else float("nan")

    report = {
        "kind": "ldm",
        "dataset_name": dataset_name,
        "ckpt_path": ckpt_path,
        "ldm_type": str(getattr(model.cfg.model, "ldm_type", "diffusion")),
        "target": target,
        "ldm_mode": mode,
        "bench_cfg": bench_cfg,
        "env": get_env_meta(device),
        "params": {
            "gen_model": int(params_gen),
            "autoencoder": int(params_ae),
            "total": int(params_total),
        },
        "overall": {
            "timing_ms": overall,
            "throughput_scenes_per_s": float(overall_throughput),
            "ms_per_scene_mean": float(overall["mean"]) / float(overall_bs) if overall_bs > 0 else float("nan"),
        },
        "per_batch": results_per_batch,
    }
    return report


def _make_synthetic_ctrlsim_batch(cfg, batch_size: int, device: torch.device) -> CtRLSimData:
    """Create a synthetic CtRLSimData(batch=B) fully on GPU for forward-only benchmarking."""
    dcfg = cfg.ctrl_sim.dataset
    mcfg = cfg.ctrl_sim.model

    B = int(batch_size)
    A = int(dcfg.max_num_agents)
    T = int(dcfg.train_context_length)

    max_num_lanes = int(dcfg.max_num_lanes)
    num_pts_per_lane = int(dcfg.num_points_per_lane)
    map_attr = int(mcfg.map_attr)  # (x,y,existence) = 3 in your config

    vocab_size = int(dcfg.vocab_size)
    rtg_K = int(dcfg.rtg.discretization)
    C = int(mcfg.num_reward_components)

    # agent_states: [B,A,T,9] (last dim is existence; include is_ego in the 8 dims before existence)
    # layout inside agent_states[..., :-1] is opaque to model except it uses:
    #   vx,vy at [:, :, :, 2:4], yaw at [:, :, :, 4], len/width at [:, :, :, 5:7], is_ego inserted before existence.
    agent_states = torch.zeros((B, A, T, 9), device=device, dtype=torch.float32)

    # positions & velocity: keep reasonable ranges
    fov = float(dcfg.fov)
    agent_states[..., 0:2] = (torch.rand((B, A, T, 2), device=device) - 0.5) * fov
    agent_states[..., 2:4] = torch.randn((B, A, T, 2), device=device) * 2.0
    agent_states[..., 4] = (torch.rand((B, A, T), device=device) - 0.5) * 2.0 * math.pi  # heading
    agent_states[..., 5] = 4.5  # length
    agent_states[..., 6] = 1.8  # width

    # is_ego (inserted before existence): we store it at index 7 in the 8-dim block, i.e. agent_states[..., 7]
    agent_states[:, 0, :, 7] = 1.0  # first agent is ego-like

    # existence (last dim)
    agent_states[..., 8] = 1.0

    # agent_types: [B,A,5] one-hot [unset, vehicle, ped, cyc, other]
    agent_types = torch.zeros((B, A, 5), device=device, dtype=torch.float32)
    agent_types[..., 1] = 1.0  # all vehicles

    actions = torch.randint(low=0, high=vocab_size, size=(B, A, T), device=device, dtype=torch.long)
    rtgs = torch.randint(low=0, high=rtg_K, size=(B, A, T, C), device=device, dtype=torch.long)

    # rtg_mask: [B,A,T,1]
    rtg_mask = torch.ones((B, A, T, 1), device=device, dtype=torch.float32)

    # timesteps: [B,A,T,1]
    t_idx = torch.arange(T, device=device, dtype=torch.long).view(1, 1, T, 1).expand(B, A, T, 1)

    moving_agent_mask = torch.ones((B, A), device=device, dtype=torch.bool)

    # map road_points: [B, L, P, map_attr]
    road_points = torch.zeros((B, max_num_lanes, num_pts_per_lane, map_attr), device=device, dtype=torch.float32)
    # random lane polylines
    road_points[..., 0:2] = (torch.rand((B, max_num_lanes, num_pts_per_lane, 2), device=device) - 0.5) * float(dcfg.lane_fov)
    road_points[..., 2] = 1.0  # existence for all points

    data = CtRLSimData()
    data["idx"] = torch.zeros((B,), device=device, dtype=torch.long)

    data["agent"].agent_states = agent_states
    data["agent"].agent_types = agent_types
    data["agent"].actions = actions
    data["agent"].rtgs = rtgs
    data["agent"].rtg_mask = rtg_mask
    data["agent"].timesteps = t_idx
    data["agent"].moving_agent_mask = moving_agent_mask

    data["map"].road_points = road_points
    return data


def _load_ctrlsim_model_robust(ckpt_path: str, cfg_ctrlsim) -> CtRLSim:
    """Same spirit as Simulator._load_ctrl_sim_model but minimal."""
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    ckpt_hparams = ckpt.get("hyper_parameters", {})
    ckpt_cfg = ckpt_hparams.get("cfg", None)

    # Prefer checkpoint cfg if present (avoids discretization mismatch)
    cfg_to_use = ckpt_cfg if ckpt_cfg is not None else cfg_ctrlsim

    try:
        model = CtRLSim.load_from_checkpoint(
            ckpt_path,
            cfg=cfg_to_use,
            map_location="cpu",
            strict=False,
        )
        return model
    except Exception as e:
        print(f"[bench][CtRLSim] load_from_checkpoint failed: {repr(e)}")
        print("[bench][CtRLSim] fallback: init model from cfg and load compatible weights")
        model = CtRLSim(cfg_to_use)
        state = ckpt.get("state_dict", ckpt)
        msd = model.state_dict()
        compatible = {}
        for k, v in state.items():
            if k in msd and isinstance(v, torch.Tensor) and v.shape == msd[k].shape:
                compatible[k] = v
        model.load_state_dict(compatible, strict=False)
        return model


def _bench_ctrlsim(cfg, dataset_name: str, bench_cfg: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    target = str(bench_cfg["target"])
    if target != "ctrl_sim_forward":
        raise ValueError(f"[bench][CtRLSim] only supports target='ctrl_sim_forward', got {target}")

    ckpt_path = resolve_path(str(cfg.sim.behaviour_model.model_path), project_root=_project_root())
    assert ckpt_path is not None and os.path.exists(ckpt_path), f"[bench] invalid sim.behaviour_model.model_path: {ckpt_path}"

    # thread dataset name (for completeness; ctrl_sim cfg uses dataset configs anyway)
    ctrl_cfg = cfg.ctrl_sim
    OmegaConf.set_struct(ctrl_cfg, False)
    ctrl_cfg.dataset_name = dataset_name
    OmegaConf.set_struct(ctrl_cfg, True)

    print(f"[bench][CtRLSim] Loading ckpt: {ckpt_path}")
    model = _load_ctrlsim_model_robust(ckpt_path, ctrl_cfg).to(device)
    model.eval()

    amp_enabled = bool(bench_cfg["amp"]["enabled"])
    amp_dtype = str(bench_cfg["amp"]["dtype"])

    B = int(bench_cfg["batch_size"])
    data = _make_synthetic_ctrlsim_batch(cfg, batch_size=B, device=device)

    params_total = count_parameters(model, trainable_only=False)

    def _fn():
        return model(data, eval=True)

    times_ms, mem = benchmark_forward(
        fn=_fn,
        device=device,
        num_warmup=int(bench_cfg["num_warmup"]),
        num_iters=int(bench_cfg["num_iters"]),
        amp_enabled=amp_enabled,
        amp_dtype=amp_dtype,
        measure_memory=True,
    )

    summary = summarize_ms(times_ms)
    mean_s = float(summary["mean"]) / 1000.0
    throughput = (B / mean_s) if (mean_s > 0 and B > 0) else float("nan")

    report = {
        "kind": "ctrl_sim",
        "dataset_name": dataset_name,
        "ckpt_path": ckpt_path,
        "target": target,
        "bench_cfg": bench_cfg,
        "env": get_env_meta(device),
        "params": {"total": int(params_total)},
        "overall": {
            "timing_ms": summary,
            "throughput_batches_per_s": float(throughput),
            "cuda_mem": {
                "baseline_allocated_mb": mem.baseline_allocated_mb,
                "baseline_reserved_mb": mem.baseline_reserved_mb,
                "peak_allocated_mb": mem.peak_allocated_mb,
                "peak_reserved_mb": mem.peak_reserved_mb,
            },
        },
    }

    if bool(bench_cfg["verbose"]):
        print(
            f"[bench][CtRLSim] mean={summary['mean']:.3f}ms p50={summary['p50']:.3f}ms "
            f"p90={summary['p90']:.3f}ms peak_alloc={mem.peak_allocated_mb:.1f}MB params={params_total/1e6:.2f}M"
        )

    return report


@hydra.main(version_base=None, config_path=CONFIG_PATH, config_name="config")
def main(cfg):
    device = _get_device()

    # dataset name threading follows existing scripts
    dataset_name = cfg.dataset_name.name

    # 1) LDM / AE eval bench config (if you add it to cfgs/eval/base.yaml, no '+' needed)
    ldm_bench = _get_bench_cfg(cfg, "ldm.eval.bench", _DEFAULT_EVAL_BENCH)
    ae_bench = _get_bench_cfg(cfg, "ae.eval.bench", _DEFAULT_EVAL_BENCH)

    # 2) sim bench config (if you add it to cfgs/sim/base.yaml, no '+' needed)
    sim_bench = _get_bench_cfg(cfg, "sim.bench", _DEFAULT_SIM_BENCH)

    reports: List[Dict[str, Any]] = []

    # ---- LDM bench ----
    if bool(ldm_bench.get("enabled", False)) and ldm_bench.get("target"):
        report = _bench_ldm(cfg, dataset_name=dataset_name, bench_cfg=ldm_bench, device=device)
        save_path = ldm_bench.get("save_path") or _auto_save_path_for_eval(cfg.ldm.eval, "ldm", str(ldm_bench["target"]))
        save_path = resolve_path(str(save_path), project_root=_project_root())
        save_json(report, save_path)
        print(f"[bench] Saved LDM report to: {save_path}")
        reports.append(report)

    # ---- AE bench ----
    if bool(ae_bench.get("enabled", False)) and ae_bench.get("target"):
        target = str(ae_bench["target"])
        ckpt_path = resolve_path(str(cfg.ae.eval.ckpt_path), project_root=_project_root())
        assert ckpt_path is not None and os.path.exists(ckpt_path), f"[bench] invalid ae.eval.ckpt_path: {ckpt_path}"

        print(f"[bench][AE] Loading ckpt: {ckpt_path}")
        ae_cfg = cfg.ae
        OmegaConf.set_struct(ae_cfg, False)
        ae_cfg.dataset_name = dataset_name
        OmegaConf.set_struct(ae_cfg, True)

        model = ScenarioDreamerAutoEncoder.load_from_checkpoint(ckpt_path, cfg=ae_cfg, map_location="cpu").to(device)
        model.eval()

        # Build a small eval dataloader (prefetch outside timing)
        split_name = str(getattr(ae_cfg.eval, "split_name", "val"))
        if dataset_name == "waymo":
            from datasets.waymo.dataset_autoencoder_waymo import WaymoDatasetAutoEncoder as _DS
        else:
            from datasets.nuplan.dataset_autoencoder_nuplan import NuplanDatasetAutoEncoder as _DS

        dataset = _DS(ae_cfg.dataset, split_name=split_name, mode="eval")
        dl = DataLoader(
            dataset,
            batch_size=int(ae_bench["batch_size"]),
            shuffle=False,
            num_workers=int(getattr(ae_cfg.datamodule, "num_workers", 0)),
            pin_memory=bool(getattr(ae_cfg.datamodule, "pin_memory", False)),
            drop_last=False,
        )

        # prefetch batches to GPU
        batches = []
        for bi, data in enumerate(dl):
            if bi >= int(ae_bench["num_batches"]):
                break
            batches.append(data.to(device))
        assert len(batches) > 0, "[bench][AE] no batches built; check dataset paths."

        amp_enabled = bool(ae_bench["amp"]["enabled"])
        amp_dtype = str(ae_bench["amp"]["dtype"])

        params_total = count_parameters(model, trainable_only=False)

        all_times = []
        per_batch = []
        for bi, data in enumerate(batches):
            if target == "ae_forward":

                def _fn():
                    return model(data)

            elif target == "ae_encoder":

                def _fn():
                    return model.model.forward_encoder(data)

            elif target == "ae_decoder":
                # precompute latents outside timing
                with torch.inference_mode():
                    a_lat, l_lat, _ = model.model.forward_encoder(data)

                def _fn():
                    return model.model.forward_decoder_with_motion(a_lat, l_lat, data)

            else:
                raise ValueError(f"[bench][AE] unknown target='{target}'")

            times_ms, mem = benchmark_forward(
                fn=_fn,
                device=device,
                num_warmup=int(ae_bench["num_warmup"]),
                num_iters=int(ae_bench["num_iters"]),
                amp_enabled=amp_enabled,
                amp_dtype=amp_dtype,
                measure_memory=True,
            )
            summary = summarize_ms(times_ms)
            per_batch.append(
                {
                    "batch_index": int(bi),
                    "target": target,
                    "timing_ms": summary,
                    "cuda_mem": {
                        "baseline_allocated_mb": mem.baseline_allocated_mb,
                        "baseline_reserved_mb": mem.baseline_reserved_mb,
                        "peak_allocated_mb": mem.peak_allocated_mb,
                        "peak_reserved_mb": mem.peak_reserved_mb,
                    },
                }
            )
            all_times.extend(times_ms)

            if bool(ae_bench["verbose"]):
                print(
                    f"[bench][AE][{target}] batch={bi} mean={summary['mean']:.2f}ms p50={summary['p50']:.2f}ms "
                    f"peak_alloc={mem.peak_allocated_mb:.1f}MB"
                )

        overall = summarize_ms(all_times)
        report = {
            "kind": "autoencoder",
            "dataset_name": dataset_name,
            "ckpt_path": ckpt_path,
            "target": target,
            "bench_cfg": ae_bench,
            "env": get_env_meta(device),
            "params": {"total": int(params_total)},
            "overall": {"timing_ms": overall},
            "per_batch": per_batch,
        }

        save_path = ae_bench.get("save_path") or _auto_save_path_for_eval(cfg.ae.eval, "ae", target)
        save_path = resolve_path(str(save_path), project_root=_project_root())
        save_json(report, save_path)
        print(f"[bench] Saved AE report to: {save_path}")
        reports.append(report)

    # ---- CtRLSim bench ----
    if bool(sim_bench.get("enabled", False)) and sim_bench.get("target"):
        report = _bench_ctrlsim(cfg, dataset_name=dataset_name, bench_cfg=sim_bench, device=device)
        save_path = sim_bench.get("save_path") or _auto_save_path_for_sim(cfg, str(sim_bench["target"]))
        save_path = resolve_path(str(save_path), project_root=_project_root())
        save_json(report, save_path)
        print(f"[bench] Saved CtRLSim report to: {save_path}")
        reports.append(report)

    if len(reports) == 0:
        print(
            "[bench] No benchmark executed. Enable one of:\n"
            "  - ldm.eval.bench.enabled=true (and set ldm.eval.bench.target)\n"
            "  - ae.eval.bench.enabled=true  (and set ae.eval.bench.target)\n"
            "  - sim.bench.enabled=true      (and set sim.bench.target)\n"
        )


if __name__ == "__main__":
    main()
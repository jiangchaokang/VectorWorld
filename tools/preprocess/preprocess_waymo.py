import os
import pickle
import random
import multiprocessing as mp
from typing import Any, Dict, Sequence

import numpy as np
import torch
from tqdm import tqdm
import hydra
from omegaconf import OmegaConf

from configs.config import CONFIG_PATH
from vectorworld.data.waymo.vae_dataset import WaymoDatasetAutoEncoder

torch.set_printoptions(threshold=100000)
np.set_printoptions(suppress=True)


# ─────────────────────────────────────────────────────────
# Global dataset, used for re-use in every worker
# ─────────────────────────────────────────────────────────
_GLOBAL_DATASET: WaymoDatasetAutoEncoder | None = None


def _init_worker(
    dataset_cfg_dict: Dict[str, Any],
    split_name: str,
    mode: str,
    file_list: Sequence[str],
):
    """Pool initializer: Build an independent WaymoDataset AutoEncoder example within each process.

    Reuse the master process here has generated a good list of files to avoid repeating glob in every worker.
    """
    global _GLOBAL_DATASET
    from omegaconf import OmegaConf

    # Reconstruct OmegaConf objects
    cfg_dataset = OmegaConf.create(dataset_cfg_dict)
    OmegaConf.set_struct(cfg_dataset, False)
    cfg_dataset.preprocess = False  # slow path: raw -> preprocessed
    OmegaConf.set_struct(cfg_dataset, True)

    # Mode's not sensitive to slow path. It's the same "train" here.
    _GLOBAL_DATASET = WaymoDatasetAutoEncoder(
        cfg_dataset,
        split_name=split_name,
        mode="train",
        files=file_list,
    )


def _process_single(idx: int) -> Dict[str, Any]:
    """Handle single raw Waymo pkl (called by Pool)."""
    global _GLOBAL_DATASET
    assert _GLOBAL_DATASET is not None, "Global dataset is not initialized in worker."

    dset = _GLOBAL_DATASET
    raw_file = dset.files[idx]

    with open(raw_file, "rb") as f:
        data = pickle.load(f)

    # slow path: you will be able to do lane drag extraction + agent filter + motion code + writepkl/COS
    result = dset.get_data(data, idx)
    # result: {'normalize_statistics': ..., 'valid_scene': bool}
    return result


@hydra.main(version_base=None, config_path=CONFIG_PATH, config_name="config")
def main(cfg):
    """Multi-process Waymo pre-processing entrance.

    Typical usage:
    ----------
    python preprocess_dataset_waymo.py       dataset_name=waymo       preprocess_waymo.mode=train       preprocess_waymo.num_workers=32       ae.dataset.preprocess_dir=outputs/data/sd_ae_waymo_motion       ae.dataset.save_to_cos=True       ae.dataset.cos_prefix="arena-custom/workspace/.../datasets/waymo/ae_motion_12m"

    python data_processing/waymo/preprocess_dataset_waymo.py         dataset_name=waymo         preprocess_waymo.mode=train         preprocess_waymo.num_workers=64         ae.dataset.preprocess_dir=outputs/data/predata         ae.dataset.save_to_cos=False
    """
    # ─────────────────────────────────────────────────────
    # Parsing dataset names
    # ─────────────────────────────────────────────────────
    dataset_name = (
        cfg.dataset_name.name
        if hasattr(cfg.dataset_name, "name")
        else str(cfg.dataset_name)
    )
    assert dataset_name == "waymo", "This preprocessing script currently only supports Waymo."

    # ─────────────────────────────────────────────────────
    # Parsing preprocess configuration (support from)cfg.preprocess_waymoRead Mode & Parallelity)
    # ─────────────────────────────────────────────────────
    preprocess_cfg = getattr(cfg, "preprocess_waymo", None)

    class _Dummy:
        pass

    if preprocess_cfg is None:
        preprocess_cfg = _Dummy()
        preprocess_cfg.mode = "train"
        preprocess_cfg.num_workers = os.cpu_count()
        preprocess_cfg.use_multiprocessing = True
        preprocess_cfg.seed = 1

    split_name: str = getattr(preprocess_cfg, "mode", "train")
    num_workers: int = int(getattr(preprocess_cfg, "num_workers", os.cpu_count()))
    use_multiprocessing: bool = bool(getattr(preprocess_cfg, "use_multiprocessing", True))
    seed: int = int(getattr(preprocess_cfg, "seed", 1))

    # ─────────────────────────────────────────────────────
    # Use AE 's dataset configuration as ground truth
    # ─────────────────────────────────────────────────────
    dataset_cfg = cfg.ae.dataset
    OmegaConf.set_struct(dataset_cfg, False)
    dataset_cfg.preprocess = False  # slow path: raw -> preprocessed

    # Select to add a base torrent for certainty timestep for WaymoDatasetAutoEncoder
    dataset_cfg.preprocess_seed = seed
    OmegaConf.set_struct(dataset_cfg, True)

    save_to_cos = bool(getattr(dataset_cfg, "save_to_cos", False))

    print(f"[Preprocess] dataset_name={dataset_name}")
    print(f"[Preprocess] split_name={split_name}")
    print(f"[Preprocess] num_workers={num_workers}, use_multiprocessing={use_multiprocessing}")
    print(f"[Preprocess] ae.dataset.preprocess_dir={dataset_cfg.preprocess_dir}")
    print(f"[Preprocess] ae.dataset.save_to_cos={save_to_cos}")

    # Fixed random torrents (for a small number of random logic only; real scene timestep selects in
    # WaymoDatasetAutoEncoder internal implementation of the certainty process based on file name hash)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)

    # Convert OmegaConf to a pickling dic, to worker
    dataset_cfg_dict: Dict[str, Any] = OmegaConf.to_container(dataset_cfg, resolve=True)

    # Initialize dataset in main process to get file list / preprocess output path
    tmp_dataset = WaymoDatasetAutoEncoder(dataset_cfg, split_name=split_name, mode="train")
    num_files = tmp_dataset.len()
    file_list = list(tmp_dataset.files)
    print(f"[Preprocess] Found {num_files} raw Waymo scenes in split '{split_name}'.")

    indices = list(range(num_files))

    # ─────────────────────────────────────────────────────
    # Normalized statistical accumulation (to help update dataset yaml)
    # ─────────────────────────────────────────────────────
    normalize_stats = {
        "max_speed": [],
        "min_length": [],
        "max_length": [],
        "min_width": [],
        "max_width": [],
        "min_lane_x": [],
        "max_lane_x": [],
        "min_lane_y": [],
        "max_lane_y": [],
    }

    def _accumulate_stats(res: Dict[str, Any] | None):
        if not res or not isinstance(res, dict):
            return
        if not res.get("valid_scene", False):
            return
        stats = res.get("normalize_statistics", None)
        if stats is None:
            return
        for k in normalize_stats.keys():
            if k in stats:
                normalize_stats[k].append(float(stats[k]))

    # ─────────────────────────────────────────────────────
    # Main logic: single/ multi-process
    # ─────────────────────────────────────────────────────
    if use_multiprocessing and num_workers > 1:
        print(f"[Preprocess] Using multiprocessing with {num_workers} workers.")
        with mp.Pool(
            processes=num_workers,
            initializer=_init_worker,
            initargs=(dataset_cfg_dict, split_name, "train", file_list),
        ) as pool:
            for res in tqdm(
                pool.imap_unordered(_process_single, indices),
                total=len(indices),
                desc=f"Preprocessing {split_name}",
            ):
                _accumulate_stats(res)
    else:
        print("[Preprocess] Running in single-process mode.")
        _init_worker(dataset_cfg_dict, split_name, "train", file_list)
        for idx in tqdm(indices, desc=f"Preprocessing {split_name}"):
            res = _process_single(idx)
            _accumulate_stats(res)

    # ─────────────────────────────────────────────────────
    # Convey and Save Normalishation Statistics
    # ─────────────────────────────────────────────────────
    def _safe_agg(vals, fn, default=None):
        return fn(vals) if len(vals) > 0 else default

    global_stats = {
        "max_speed": _safe_agg(normalize_stats["max_speed"], max),
        "min_length": _safe_agg(normalize_stats["min_length"], min),
        "max_length": _safe_agg(normalize_stats["max_length"], max),
        "min_width": _safe_agg(normalize_stats["min_width"], min),
        "max_width": _safe_agg(normalize_stats["max_width"], max),
        "min_lane_x": _safe_agg(normalize_stats["min_lane_x"], min),
        "max_lane_x": _safe_agg(normalize_stats["max_lane_x"], max),
        "min_lane_y": _safe_agg(normalize_stats["min_lane_y"], min),
        "max_lane_y": _safe_agg(normalize_stats["max_lane_y"], max),
    }

    print("[Preprocess] Dataset-level normalization stats (aggregated over valid scenes):")
    for k, v in global_stats.items():
        print(f"  {k}: {v}")

    # Save to preprocess_dir level directory
    stats_out_dir = os.path.join(os.path.dirname(tmp_dataset.preprocessed_dir))
    os.makedirs(stats_out_dir, exist_ok=True)
    stats_path = os.path.join(stats_out_dir, f"normalize_stats_{split_name}.pkl")
    with open(stats_path, "wb") as f:
        pickle.dump(global_stats, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"[Preprocess] Saved normalization stats to: {stats_path}")


if __name__ == "__main__":
    main()
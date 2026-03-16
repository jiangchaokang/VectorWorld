import glob
import os
import pickle
import random
from typing import Dict, List, Tuple

import hydra
from configs.config import CONFIG_PATH


def _get_dataset_name(cfg) -> str:
    # cfg.dataset_name is a config group with field `name`
    try:
        return str(cfg.dataset_name.name)
    except Exception:
        return str(getattr(cfg, "dataset_name", "unknown"))


def _ensure_dir(path: str) -> None:
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)


def _nuplan_strip_extensions(filename: str) -> str:
    """Strip known extensions for nuPlan lane filenames.
    Handles:
      - xxx.pkl
      - xxx.pkl.gz
      - xxx.gz (rare)
    """
    bn = os.path.basename(filename)
    if bn.endswith(".gz"):
        bn = bn[: -len(".gz")]
    if bn.endswith(".pkl"):
        bn = bn[: -len(".pkl")]
    return bn


def _select_subset(files: List[str], num_samples: int | None, seed: int) -> List[str]:
    rng = random.Random(int(seed))
    files = list(files)
    rng.shuffle(files)
    if num_samples is None or num_samples <= 0 or num_samples >= len(files):
        return files
    return files[: int(num_samples)]


def _create_waymo_eval_set(cfg) -> Tuple[Dict, str]:
    params = cfg.waymo_eval_set

    split: str = str(params.split)
    num_samples_cfg = getattr(params, "num_samples", None)
    num_samples = int(num_samples_cfg) if num_samples_cfg is not None else None
    seed = int(getattr(params, "seed", 42))
    filename_pattern: str = str(getattr(params, "filename_pattern", "*-of-*_*_0_*.pkl"))
    output_path: str = str(getattr(params, "output_path", "data/metadata/waymo_eval_set.pkl"))

    split_dir = os.path.join(str(cfg.scenario_dreamer_ae_preprocess_waymo_dir), split)
    glob_pattern = os.path.join(split_dir, filename_pattern)
    files_full = sorted(glob.glob(glob_pattern))

    if len(files_full) == 0:
        raise RuntimeError(
            "[create_eval_set][waymo] No files found.\n"
            f"  split_dir: {split_dir}\n"
            f"  pattern  : {filename_pattern}\n"
            f"  glob     : {glob_pattern}\n"
            "Check `scenario_dreamer_ae_preprocess_waymo_dir` and `waymo_eval_set.filename_pattern`."
        )

    selected = _select_subset(files_full, num_samples, seed)
    filenames = [os.path.basename(p) for p in selected]

    payload = {
        "dataset": "waymo",
        "split": split,
        "seed": seed,
        "num_selected": len(filenames),
        "num_total_candidates": len(files_full),
        "files": filenames,
    }
    return payload, output_path


def _create_nuplan_eval_set(cfg) -> Tuple[Dict, str]:
    params = cfg.nuplan_eval_set

    split = str(getattr(params, "split", "test"))
    num_samples_cfg = getattr(params, "num_samples", None)
    num_samples = int(num_samples_cfg) if num_samples_cfg is not None else None
    seed = int(getattr(params, "seed", 42))
    lane_pattern = str(getattr(params, "lane_filename_pattern", "*.gz*"))
    agent_suffix = str(getattr(params, "agent_suffix", "_0.pkl"))
    gt_lane_root = str(getattr(params, "gt_lane_root_dir", ""))
    output_path = str(getattr(params, "output_path", "data/metadata/nuplan_eval_set.pkl"))

    lane_split_dir = os.path.join(gt_lane_root, split)

    if not os.path.isdir(lane_split_dir):
        raise RuntimeError(
            f"[create_eval_set][nuplan] gt_lane split dir not found: {lane_split_dir}\n"
            "Override via nuplan_eval_set.gt_lane_root_dir or env SCENARIO_DREAMER_LANE_PREPROCESS_NUPLAN_DIR"
        )

    lane_files = sorted(glob.glob(os.path.join(lane_split_dir, lane_pattern)))

    if not lane_files:
        raise RuntimeError(
            f"[create_eval_set][nuplan] No lane files found.\n"
            f"  dir: {lane_split_dir}, pattern: {lane_pattern}"
        )

    # Select directly, without having to check the agent files
    selected = _select_subset(lane_files, num_samples, seed)
    filenames = [os.path.basename(p) for p in selected]

    return {
        "dataset": "nuplan",
        "split": split,
        "seed": seed,
        "num_selected": len(filenames),
        "num_total_lane_candidates": len(lane_files),
        "agent_suffix": agent_suffix,
        "files": filenames,
    }, output_path


@hydra.main(version_base=None, config_path=CONFIG_PATH, config_name="config")
def main(cfg):
    """Create eval_set.pkl for LDM Metrics.

    Output format (pickle):
        {"files": [...], ...optional metadata...}

    - waymo: samples from cfg.scenario_dreamer_ae_preprocess_waymo_dir/<split> with glob pattern
    - nuplan: samples from cfg.nuplan_eval_set.gt_lane_root_dir/<split> and checks agent existence
    """
    dataset = _get_dataset_name(cfg).lower()

    if dataset == "waymo":
        payload, output_path = _create_waymo_eval_set(cfg)
    elif dataset == "nuplan":
        payload, output_path = _create_nuplan_eval_set(cfg)
    else:
        raise ValueError(
            f"[create_eval_set] Unsupported dataset_name='{dataset}'. "
            "Use dataset_name=waymo or dataset_name=nuplan."
        )

    _ensure_dir(output_path)
    with open(output_path, "wb") as f:
        pickle.dump(payload, f)

    print(
        f"[create_eval_set] Done. dataset='{payload.get('dataset')}', split='{payload.get('split')}', "
        f"selected={payload.get('num_selected')} -> {output_path}"
    )


if __name__ == "__main__":
    main()

"""
Examples
--------
# Waymo
python data_processing/waymo/create_waymo_eval_set.py \
  dataset_name=waymo \
  waymo_eval_set.split=test \
  waymo_eval_set.num_samples=50000 \
  waymo_eval_set.output_path=data/metadata/waymo_eval_set.pkl

# NuPlan
python data_processing/waymo/create_waymo_eval_set.py \
  dataset_name=nuplan \
  nuplan_eval_set.split=test \
  nuplan_eval_set.num_samples=50000 \
  nuplan_eval_set.output_path=data/metadata/nuplan_eval_set_1219.pkl
"""
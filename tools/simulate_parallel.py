import os
import random
import multiprocessing as mp
from typing import List

import numpy as np
import torch
import hydra
from omegaconf import OmegaConf

from configs.config import CONFIG_PATH
from vectorworld.simulation.simulator import Simulator
from vectorworld.simulation.idm_policy import IDMPolicy
from tools.simulate import run_serial, PolicyEvaluator


def _split_indices(num_items: int, num_splits: int) -> List[List[int]]:
    indices = np.arange(num_items)
    splits = np.array_split(indices, num_splits)
    return [s.tolist() for s in splits if len(s) > 0]


def _worker_main(args) -> dict:
    worker_id, cfg_dict, file_slice, episode_ids = args
    cfg = OmegaConf.create(cfg_dict)

    base_seed = int(getattr(cfg.sim, "seed", 0))
    worker_seed = base_seed + worker_id * 1000
    cfg.sim.seed = worker_seed
    torch.manual_seed(worker_seed)
    np.random.seed(worker_seed)
    random.seed(worker_seed)

    if getattr(cfg.sim, "save_trajectory", False):
        traj_root = getattr(cfg.sim, "trajectory_path", None)
        if traj_root is not None:
            cfg.sim.trajectory_path = os.path.join(traj_root, f"worker_{worker_id}")

    if getattr(cfg.sim, "visualize", False):
        movie_root = getattr(cfg.sim, "movie_path", None)
        if movie_root is not None:
            cfg.sim.movie_path = os.path.join(movie_root, f"worker_{worker_id}")

    env = Simulator(cfg)

    if env.mode == "vectorworld":
        if file_slice is not None:
            env.test_files = file_slice
            env.num_test_scenarios = len(env.test_files)
    else:
        if episode_ids is not None:
            env.num_test_scenarios = len(episode_ids)
        else:
            env.num_test_scenarios = int(getattr(cfg.sim, "num_online_scenarios", 1))

    policy = IDMPolicy(cfg, env)
    evaluator = PolicyEvaluator(cfg.sim, policy, env, episode_ids=episode_ids)

    summary_dict, _ = evaluator.evaluate_policy()
    summary_dict["worker_id"] = int(worker_id)
    return summary_dict


@hydra.main(version_base=None, config_path=CONFIG_PATH, config_name="config")
def main(cfg):
    num_workers = int(getattr(cfg.sim, "num_workers", 1))
    if num_workers <= 1:
        print("[run_simulation_parallel] sim.num_workers=1, backtools/simulate.pySerial execution")
        run_serial(cfg)
        return

    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    cfg_dict = OmegaConf.to_container(cfg, resolve=True)

    mode = cfg.sim.mode
    worker_args = []

    if mode == "vectorworld":
        dataset_path = cfg.sim.dataset_path
        assert dataset_path is not None and os.path.isdir(dataset_path), \
            f"sim.mode=vectorworld requires a valid sim.dataset_path, currently: {dataset_path}"

        all_files = sorted(os.listdir(dataset_path))
        total = len(all_files)
        index_slices = _split_indices(total, num_workers)

        for wid, idx_slice in enumerate(index_slices):
            file_slice = [all_files[i] for i in idx_slice]
            global_ids = idx_slice
            worker_args.append((wid, cfg_dict, file_slice, global_ids))

    elif mode == "vectorworld_online":
        total = int(getattr(cfg.sim, "num_online_scenarios", 1))
        index_slices = _split_indices(total, num_workers)
        for wid, ep_ids in enumerate(index_slices):
            worker_args.append((wid, cfg_dict, None, ep_ids))
    else:
        raise ValueError(f"[run_simulation_parallel] unsupported sim.mode='{mode}'")

    print(f"[run_simulation_parallel] mode={mode}, Total number of episodes={total}, worker={len(worker_args)}")

    if len(worker_args) == 0:
        print("[run_simulation_parallel] There is no euisode to be implemented.")
        return

    with mp.Pool(processes=len(worker_args)) as pool:
        results = pool.map(_worker_main, worker_args)

    if len(results) == 0:
        print("[run_simulation_parallel] No worker returns the result.")
        return

    finished_episodes = sum(int(r.get("num_episodes", 0)) for r in results)
    total_sim_steps = sum(int(r.get("total_sim_steps", 0)) for r in results)

    print("\n[Parallel Simulation Done]")
    print(f"planned_episodes: {int(total)}")
    print(f"finished_episodes: {finished_episodes}")
    print(f"total_sim_steps: {total_sim_steps}")

    if finished_episodes != int(total):
        print("[Parallel Simulation Warning] Finished_episodes is inconsistent with planned_episodes. Check the worker log.")


if __name__ == "__main__":
    main()
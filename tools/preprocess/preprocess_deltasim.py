#!/usr/bin/env python3
"""
Ultra-fast Waymo preprocessing with fork + pre-init pattern.
Key: Initialize dataset ONCE in parent, workers inherit via COW.
"""

from __future__ import annotations

import os
import sys

# ============================================================
# Thread limits BEFORE any imports
# ============================================================
for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", 
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_k] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import math
import pickle
import time
import threading
import multiprocessing as mp
import zlib
from pathlib import Path
from typing import List, Tuple, Set
from dataclasses import dataclass

import hydra
from omegaconf import OmegaConf
from configs.config import CONFIG_PATH

# ============================================================
# Config
# ============================================================
@dataclass(frozen=True)
class PConfig:
    num_workers: int
    progress_interval: int
    buffer_size: int
    scan_existing: bool
    shard_id: int
    num_shards: int


def _read_cfg() -> PConfig:
    cpu = os.cpu_count() or 64
    return PConfig(
        num_workers=int(os.environ.get("DELTA_SIM_PREPROCESS_WORKERS", str(min(64, cpu)))),
        progress_interval=int(os.environ.get("DELTA_SIM_PROGRESS_INTERVAL", "500")),
        buffer_size=int(os.environ.get("DELTA_SIM_BUFFER_SIZE", str(1 << 20))),
        scan_existing=os.environ.get("DELTA_SIM_SCAN_EXISTING", "1").strip() not in ("0", "false"),
        shard_id=int(os.environ.get("DELTA_SIM_SHARD_ID", "0")),
        num_shards=int(os.environ.get("DELTA_SIM_NUM_SHARDS", "1")),
    )


# ============================================================
# Globals (parent-initialized, inherited by fork)
# ============================================================
_DSET = None          # CtRLSimDataset instance
_OUT_DIR = ""
_BUFFER_SIZE = 1 << 20
_PROGRESS = None      # shared counter
_PROGRESS_INTERVAL = 500


def _init_parent_dataset(dcfg_dict: dict, split: str) -> None:
    """Initialize dataset ONCE in parent before fork."""
    global _DSET
    print(f"[Init] Loading CtRLSimDataset for split={split} ...")
    t0 = time.time()
    
    from vectorworld.data.waymo.deltasim_dataset import CtRLSimDataset
    dcfg = OmegaConf.create(dcfg_dict)
    dcfg.preprocess = False
    dcfg.collect_state_transitions = False
    _DSET = CtRLSimDataset(dcfg, split_name=split)
    
    print(f"[Init] Dataset ready in {time.time() - t0:.1f}s")


def _worker_init(out_dir: str, progress_arr, buffer_size: int, progress_interval: int) -> None:
    """Minimal worker init - dataset already inherited from parent."""
    global _OUT_DIR, _BUFFER_SIZE, _PROGRESS, _PROGRESS_INTERVAL
    _OUT_DIR = out_dir
    _BUFFER_SIZE = buffer_size
    _PROGRESS = progress_arr
    _PROGRESS_INTERVAL = progress_interval
    
    # Re-apply thread limits
    import torch
    torch.set_num_threads(1)


def _process_chunk(file_paths: List[str]) -> Tuple[int, int, int, List[str]]:
    """Process a chunk of files. Dataset is inherited from parent."""
    ok = skip = err = 0
    errors: List[str] = []
    
    for i, fpath in enumerate(file_paths):
        try:
            out_path = os.path.join(_OUT_DIR, os.path.basename(fpath))
            if os.path.exists(out_path):
                skip += 1
                continue
            
            with open(fpath, "rb", buffering=_BUFFER_SIZE) as f:
                data = pickle.load(f)
            
            if "objects" in data and len(data["objects"]) <= 1:
                skip += 1
                continue
            
            _DSET.files = [fpath]
            _DSET.get_data(data, idx=0)
            ok += 1
            
        except Exception as e:
            err += 1
            errors.append(f"{fpath}\t{type(e).__name__}: {e}")
        
        # Update shared progress
        if _PROGRESS is not None and (i + 1) % _PROGRESS_INTERVAL == 0:
            with _PROGRESS.get_lock():
                _PROGRESS.value += _PROGRESS_INTERVAL
    
    # Final progress update
    remainder = len(file_paths) % _PROGRESS_INTERVAL
    if _PROGRESS is not None and remainder > 0:
        with _PROGRESS.get_lock():
            _PROGRESS.value += remainder
    
    return ok, skip, err, errors


# ============================================================
# Helpers
# ============================================================
def _scan_pkls(dir_path: str) -> List[str]:
    return [e.path for e in os.scandir(dir_path) if e.is_file() and e.name.endswith(".pkl")]


def _scan_existing_names(dir_path: str) -> Set[str]:
    if not os.path.isdir(dir_path):
        return set()
    return {e.name for e in os.scandir(dir_path) if e.is_file() and e.name.endswith(".pkl")}


def _shard_filter(files: List[str], shard_id: int, num_shards: int) -> List[str]:
    if num_shards <= 1:
        return files
    return [f for f in files if zlib.crc32(os.path.basename(f).encode()) % num_shards == shard_id]


def _chunk_round_robin(files: List[str], n_chunks: int) -> List[List[str]]:
    buckets = [[] for _ in range(n_chunks)]
    for i, f in enumerate(files):
        buckets[i % n_chunks].append(f)
    return [b for b in buckets if b]


def _progress_thread(progress_val, total: int, stop_evt: threading.Event) -> None:
    start = time.time()
    while not stop_evt.is_set():
        done = progress_val.value
        elapsed = time.time() - start
        rate = done / elapsed if elapsed > 0 else 0
        eta = (total - done) / rate / 60 if rate > 0 else float("inf")
        pct = 100.0 * done / total if total > 0 else 0
        sys.stdout.write(f"\r[Progress] {done:>7}/{total} ({pct:5.1f}%) | {rate:7.1f} files/sec | ETA: {eta:6.1f} min   ")
        sys.stdout.flush()
        time.sleep(0.5)
    print()


# ============================================================
# Main
# ============================================================
@hydra.main(version_base=None, config_path=CONFIG_PATH, config_name="config")
def main(cfg) -> None:
    if cfg.dataset_name.name != "waymo" or cfg.model_name != "deltasim":
        raise ValueError("Only supports dataset_name=waymo, model_name=deltasim")
    
    dcfg = cfg.deltasim.dataset
    if dcfg.preprocess:
        raise ValueError("Set delta_sim.dataset.preprocess=false")
    
    # Resolve paths
    from hydra.utils import to_absolute_path
    dataset_root = to_absolute_path(dcfg.dataset_path)
    preprocess_root = to_absolute_path(dcfg.preprocess_dir)
    vocab_path = to_absolute_path(dcfg.k_disks_vocab_path)
    
    dcfg_dict = OmegaConf.to_container(dcfg, resolve=True)
    dcfg_dict.update({
        "dataset_path": dataset_root,
        "preprocess_dir": preprocess_root,
        "k_disks_vocab_path": vocab_path,
        "preprocess": False,
        "collect_state_transitions": False,
    })
    
    pcfg = _read_cfg()
    splits = [s.strip() for s in os.environ.get("DELTA_SIM_PREPROCESS_SPLITS", "train").split(",") if s.strip()]
    
    print(f"[Config] workers={pcfg.num_workers}, shard={pcfg.shard_id}/{pcfg.num_shards}")
    print(f"[Config] dataset_root={dataset_root}")
    print(f"[Config] preprocess_root={preprocess_root}")
    
    # MUST use fork for parent-init pattern
    ctx = mp.get_context("fork")
    
    for split in splits:
        raw_dir = os.path.join(dataset_root, split)
        out_dir = os.path.join(preprocess_root, split)
        os.makedirs(out_dir, exist_ok=True)
        
        if not os.path.isdir(raw_dir):
            print(f"[Skip] {raw_dir} not found")
            continue
        
        # Scan files
        print(f"[Scan] {raw_dir} ...")
        all_files = _scan_pkls(raw_dir)
        
        if pcfg.scan_existing:
            print(f"[Scan] existing outputs ...")
            existing = _scan_existing_names(out_dir)
            todo = [f for f in all_files if os.path.basename(f) not in existing]
        else:
            todo = all_files
        
        todo = _shard_filter(todo, pcfg.shard_id, pcfg.num_shards)
        total = len(todo)
        print(f"[{split}] {total} files to process")
        
        if total == 0:
            continue
        
        # *** KEY: Initialize dataset ONCE in parent ***
        global _DSET
        _DSET = None
        _init_parent_dataset(dcfg_dict, split)
        
        # Chunk tasks
        n_workers = min(pcfg.num_workers, total)
        n_chunks = min(total, n_workers * 8)  # 8 chunks per worker for load balance
        tasks = _chunk_round_robin(todo, n_chunks)
        del todo
        
        print(f"[Run] workers={n_workers}, chunks={len(tasks)}")
        
        # Shared progress counter
        progress = ctx.Value("Q", 0)
        
        # Progress monitor
        stop_evt = threading.Event()
        mon = threading.Thread(target=_progress_thread, args=(progress, total, stop_evt), daemon=True)
        mon.start()
        
        t0 = time.time()
        total_ok = total_skip = total_err = 0
        all_errors: List[str] = []
        
        try:
            with ctx.Pool(
                processes=n_workers,
                initializer=_worker_init,
                initargs=(out_dir, progress, pcfg.buffer_size, pcfg.progress_interval),
            ) as pool:
                for ok, skip, err, errs in pool.imap_unordered(_process_chunk, tasks, chunksize=1):
                    total_ok += ok
                    total_skip += skip
                    total_err += err
                    all_errors.extend(errs)
        finally:
            stop_evt.set()
            mon.join(timeout=2)
        
        elapsed = time.time() - t0
        rate = total / elapsed if elapsed > 0 else 0
        print(f"\n[Done] {split}: {elapsed/60:.1f} min, {rate:.1f} files/sec")
        print(f"[Stats] OK={total_ok}, Skip={total_skip}, Error={total_err}")
        
        if all_errors:
            err_log = os.path.join(out_dir, f"errors_shard{pcfg.shard_id:02d}.log")
            with open(err_log, "w") as f:
                f.write("\n".join(all_errors))
            print(f"[Errors] logged to {err_log}")
    
    print("[All done]")


if __name__ == "__main__":
    main()
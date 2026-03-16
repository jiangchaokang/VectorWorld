from __future__ import annotations

import json
import os
import platform
import time
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch


def _now_timestamp() -> str:
    return time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())


def ensure_dir(path: str | os.PathLike) -> str:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p.as_posix()


def resolve_path(path: str, project_root: Optional[str] = None) -> str:
    """Resolve a possibly-relative path robustly under Hydra chdir."""
    if path is None:
        return path
    p = Path(path)
    if p.is_absolute() and p.exists():
        return p.as_posix()
    if p.exists():
        return p.resolve().as_posix()
    if project_root is not None:
        p2 = Path(project_root) / p
        if p2.exists():
            return p2.resolve().as_posix()
    # fallback: return as-is
    return str(path)


def count_parameters(module: torch.nn.Module, trainable_only: bool = False) -> int:
    if trainable_only:
        return sum(p.numel() for p in module.parameters() if p.requires_grad)
    return sum(p.numel() for p in module.parameters())


def summarize_ms(times_ms: List[float]) -> Dict[str, float]:
    if len(times_ms) == 0:
        return {
            "n": 0,
            "mean": float("nan"),
            "std": float("nan"),
            "min": float("nan"),
            "p50": float("nan"),
            "p90": float("nan"),
            "p95": float("nan"),
            "p99": float("nan"),
            "max": float("nan"),
        }
    x = np.asarray(times_ms, dtype=np.float64)
    return {
        "n": int(x.size),
        "mean": float(x.mean()),
        "std": float(x.std(ddof=1)) if x.size > 1 else 0.0,
        "min": float(x.min()),
        "p50": float(np.percentile(x, 50)),
        "p90": float(np.percentile(x, 90)),
        "p95": float(np.percentile(x, 95)),
        "p99": float(np.percentile(x, 99)),
        "max": float(x.max()),
    }


def get_env_meta(device: torch.device) -> Dict[str, Any]:
    meta = {
        "timestamp": _now_timestamp(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "device": str(device),
    }
    if device.type == "cuda" and torch.cuda.is_available():
        idx = device.index if device.index is not None else 0
        meta.update(
            {
                "cuda_device_index": int(idx),
                "cuda_device_name": torch.cuda.get_device_name(idx),
                "cuda_capability": ".".join(map(str, torch.cuda.get_device_capability(idx))),
                "cuda_total_mem_gb": float(torch.cuda.get_device_properties(idx).total_memory / (1024**3)),
            }
        )
    return meta


def autocast_context(device: torch.device, enabled: bool, dtype: str):
    if (not enabled) or (device.type != "cuda"):
        return nullcontext()

    dtype = str(dtype).lower()
    if dtype in ("bf16", "bfloat16"):
        amp_dtype = torch.bfloat16
    elif dtype in ("fp16", "float16"):
        amp_dtype = torch.float16
    else:
        raise ValueError(f"Unknown amp dtype='{dtype}', use bf16/fp16.")

    return torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=True)


@dataclass
class CUDAMemoryStats:
    baseline_allocated_mb: float
    baseline_reserved_mb: float
    peak_allocated_mb: float
    peak_reserved_mb: float

    @staticmethod
    def zeros() -> "CUDAMemoryStats":
        return CUDAMemoryStats(0.0, 0.0, 0.0, 0.0)


def benchmark_forward(
    fn: Callable[[], Any],
    device: torch.device,
    num_warmup: int = 10,
    num_iters: int = 50,
    amp_enabled: bool = True,
    amp_dtype: str = "bf16",
    measure_memory: bool = True,
) -> Tuple[List[float], CUDAMemoryStats]:
    """Benchmark forward-only latency.

    - If CUDA: use torch.cuda.Event timing (ms).
    - Warmup runs are excluded.
    - If measure_memory=True and CUDA: reports peak allocated/reserved (MB) during timed region.
    """
    if num_warmup < 0 or num_iters <= 0:
        raise ValueError(f"Invalid warmup/iters: warmup={num_warmup}, iters={num_iters}")

    # Warmup
    with torch.inference_mode():
        with autocast_context(device, amp_enabled, amp_dtype):
            for _ in range(num_warmup):
                _ = fn()

    if device.type == "cuda":
        torch.cuda.synchronize(device)

    mem = CUDAMemoryStats.zeros()
    if measure_memory and device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        baseline_alloc = torch.cuda.memory_allocated(device) / (1024**2)
        baseline_res = torch.cuda.memory_reserved(device) / (1024**2)
        mem.baseline_allocated_mb = float(baseline_alloc)
        mem.baseline_reserved_mb = float(baseline_res)

    times_ms: List[float] = []

    if device.type == "cuda":
        starter = torch.cuda.Event(enable_timing=True)
        ender = torch.cuda.Event(enable_timing=True)
        with torch.inference_mode():
            with autocast_context(device, amp_enabled, amp_dtype):
                for _ in range(num_iters):
                    starter.record()
                    _ = fn()
                    ender.record()
                    ender.synchronize()
                    times_ms.append(float(starter.elapsed_time(ender)))
    else:
        with torch.inference_mode():
            for _ in range(num_iters):
                t0 = time.perf_counter()
                _ = fn()
                t1 = time.perf_counter()
                times_ms.append(float((t1 - t0) * 1000.0))

    if measure_memory and device.type == "cuda":
        peak_alloc = torch.cuda.max_memory_allocated(device) / (1024**2)
        peak_res = torch.cuda.max_memory_reserved(device) / (1024**2)
        mem.peak_allocated_mb = float(peak_alloc)
        mem.peak_reserved_mb = float(peak_res)

    return times_ms, mem


def save_json(obj: Dict[str, Any], path: str) -> None:
    p = Path(path)
    ensure_dir(p.parent)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def flatten_dict(d: Dict[str, Any], prefix: str = "", sep: str = ".") -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k, v in d.items():
        kk = f"{prefix}{sep}{k}" if prefix else str(k)
        if isinstance(v, dict):
            out.update(flatten_dict(v, prefix=kk, sep=sep))
        else:
            out[kk] = v
    return out
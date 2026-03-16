from __future__ import annotations

import os
import pickle
import hashlib
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, List

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

from configs.config import PROJECT_ROOT


def resolve_path(p: str) -> str:
    """Resolve a possibly-relative path against PROJECT_ROOT."""
    if p is None:
        return p
    return p if os.path.isabs(p) else str(PROJECT_ROOT / p)


def _load_pkl(path: str) -> Any:
    with open(path, "rb") as f:
        return pickle.load(f)


def load_kdisks_V(vocab_path: str) -> np.ndarray:
    vocab_path = resolve_path(str(vocab_path))
    if not os.path.exists(vocab_path):
        raise FileNotFoundError(f"[ckpt_loader] k-disks vocab not found: {vocab_path}")
    payload = _load_pkl(vocab_path)
    V = np.asarray(payload["V"], dtype=np.float32)
    if V.ndim != 2 or V.shape[1] != 3:
        raise ValueError(f"[ckpt_loader] Invalid vocab V shape {V.shape}, expected [K,3]")
    return V

@dataclass
class LoadReport:
    ckpt_path: str
    strict: bool
    legacy_renames: Dict[str, str]
    missing_keys: List[str]
    unexpected_keys: List[str]
    shape_mismatch: Dict[str, Tuple[Tuple[int, ...], Tuple[int, ...]]]


def _ensure_dictconfig(cfg: Any) -> DictConfig:
    if isinstance(cfg, DictConfig):
        return cfg
    return OmegaConf.create(cfg)


def _extract_state_dict(ckpt: Dict[str, Any]) -> Dict[str, torch.Tensor]:
    sd = ckpt.get("state_dict", None)
    if sd is None:
        # some checkpoints may store weights at root
        if all(isinstance(v, torch.Tensor) for v in ckpt.values()):
            return ckpt  # type: ignore
        raise RuntimeError("[ckpt_loader] checkpoint has no 'state_dict'")
    return sd


def apply_legacy_key_map(state_dict: Dict[str, torch.Tensor]) -> Tuple[Dict[str, torch.Tensor], Dict[str, str]]:
    """
    Map known legacy parameter names to current names.
    IMPORTANT: We only apply a mapping when it's unambiguous.
    """
    sd = dict(state_dict)
    renames: Dict[str, str] = {}

    # Legacy RTG embedding table rename:
    #   encoder.embed_rtg_veh.weight  -> encoder.embed_rtg_bin.weight
    target = "encoder.embed_rtg_bin.weight"
    if target not in sd:
        candidates = [
            "encoder.embed_rtg_veh.weight",
            "encoder.embed_rtg_goal.weight",
            "encoder.embed_rtg_road.weight",
            "encoder.embed_rtg_edge.weight",
        ]
        present = [k for k in candidates if k in sd]
        if len(present) == 1:
            src = present[0]
            sd[target] = sd.pop(src)
            renames[src] = target

    return sd, renames


def diff_state_dict(
    model: torch.nn.Module, state_dict: Dict[str, torch.Tensor]
) -> Tuple[List[str], List[str], Dict[str, Tuple[Tuple[int, ...], Tuple[int, ...]]]]:
    msd = model.state_dict()
    missing = [k for k in msd.keys() if k not in state_dict]
    unexpected = [k for k in state_dict.keys() if k not in msd]
    shape_mismatch: Dict[str, Tuple[Tuple[int, ...], Tuple[int, ...]]] = {}
    for k, v in state_dict.items():
        if k in msd and tuple(v.shape) != tuple(msd[k].shape):
            shape_mismatch[k] = (tuple(v.shape), tuple(msd[k].shape))
    return missing, unexpected, shape_mismatch


def _format_key_list(keys: List[str], max_items: int = 30) -> str:
    if not keys:
        return "[]"
    head = keys[:max_items]
    tail = keys[max_items:]
    s = "\n".join(f"  - {k}" for k in head)
    if tail:
        s += f"\n  ... +{len(tail)} more"
    return s


def load_ctrlsim_model_strict(
    ckpt_path: str,
    *,
    device: str = "cuda",
    strict: bool = True,
    allow_legacy_key_map: bool = True,
    override_kdisks_vocab_path: Optional[str] = None,
) -> Tuple["CtRLSim", DictConfig, LoadReport]:
    """
    Strictly load CtRLSim from a Lightning checkpoint.
    - By default, any missing/unexpected/shape mismatch => RuntimeError.
    - Supports limited legacy key renames.
    - Optional: override vocab path, but you SHOULD verify hash externally.

    Returns: (model, cfg_from_ckpt, report)
    """
    from models.ctrl_sim import CtRLSim  # local import to avoid circular deps

    ckpt_path = resolve_path(ckpt_path)
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"[ckpt_loader] checkpoint not found: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    ckpt_cfg = ckpt.get("hyper_parameters", {}).get("cfg", None)
    if ckpt_cfg is None:
        raise RuntimeError(f"[ckpt_loader] checkpoint missing hyper_parameters.cfg: {ckpt_path}")
    cfg = _ensure_dictconfig(ckpt_cfg)

    # Resolve / override vocab path (NO silent override)
    OmegaConf.set_struct(cfg, False)
    cfg.dataset.k_disks_vocab_path = resolve_path(str(override_kdisks_vocab_path))
    OmegaConf.set_struct(cfg, True)

    # Build model from ckpt cfg
    model = CtRLSim(cfg)

    # Load weights
    sd = _extract_state_dict(ckpt)
    legacy_renames: Dict[str, str] = {}
    if allow_legacy_key_map:
        sd, legacy_renames = apply_legacy_key_map(sd)

    missing, unexpected, shape_mismatch = diff_state_dict(model, sd)
    report = LoadReport(
        ckpt_path=ckpt_path,
        strict=strict,
        legacy_renames=legacy_renames,
        missing_keys=missing,
        unexpected_keys=unexpected,
        shape_mismatch=shape_mismatch,
    )

    if strict and (missing or unexpected or shape_mismatch):
        msg = (
            "[ckpt_loader] STRICT load failed (state_dict does not exactly match model).\n"
            f"ckpt: {ckpt_path}\n"
            f"legacy_renames_applied: {legacy_renames}\n"
            f"missing_keys ({len(missing)}):\n{_format_key_list(missing)}\n"
            f"unexpected_keys ({len(unexpected)}):\n{_format_key_list(unexpected)}\n"
            f"shape_mismatch ({len(shape_mismatch)}):\n"
        )
        for k, (s_ckpt, s_model) in list(shape_mismatch.items())[:30]:
            msg += f"  - {k}: ckpt={s_ckpt} vs model={s_model}\n"
        if len(shape_mismatch) > 30:
            msg += f"  ... +{len(shape_mismatch) - 30} more\n"
        raise RuntimeError(msg)

    # Non-strict load (still filters out shape mismatches to avoid runtime error)
    if not strict:
        if shape_mismatch:
            for k in shape_mismatch.keys():
                sd.pop(k, None)
        model.load_state_dict(sd, strict=False)
    else:
        model.load_state_dict(sd, strict=True)

    model.to(device)
    model.eval()
    return model, cfg, report
import os
import pickle
from typing import Any, Dict

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from pytorch_lightning.utilities import grad_norm
from torch import nn

from vectorworld.networks.deltasim_net import CtRLSimEncoder, CtRLSimDecoder
from vectorworld.utils.train_helpers import create_lambda_lr_linear

def _wrap_to_pi_torch(x: torch.Tensor) -> torch.Tensor:
    return torch.atan2(torch.sin(x), torch.cos(x))


def _linear_warmup_factor(step: int, warmup_steps: int) -> float:
    if warmup_steps <= 0:
        return 1.0
    return float(min(max(step, 0), warmup_steps)) / float(warmup_steps)


_HARD_MASK_LOGIT_SAFE_DEFAULT = -1.0e4


class DeltaSim(pl.LightningModule):
    """CtRL-Sim LightningModule with DKAL, Physics Prior, and Residual Refinement."""

    def __init__(self, cfg):
        super().__init__()
        self.save_hyperparameters()

        self.cfg = cfg
        self.cfg_model = self.cfg.model
        self.cfg_dataset = self.cfg.dataset

        self.action_dim = int(self.cfg_dataset.vocab_size)
        self.seq_len = int(self.cfg_dataset.train_context_length)
        self.delta_t = 1.0 / float(self.cfg_dataset.simulation_hz)

        self.encoder = CtRLSimEncoder(self.cfg)
        self.decoder = CtRLSimDecoder(self.cfg)
        self.runtime_rtg_tilt = 0.0

        self._init_token_vocab_and_features()

        self._phys_cfg = self._parse_phys_prior_cfg()
        self._exp_phys_cfg = self._parse_expected_phys_loss_cfg()
        self._risk_se2_cfg = self._parse_risk_se2_loss_cfg()
        self._smooth_cfg = self._parse_temporal_smooth_loss_cfg()
        self._actw_cfg = self._parse_action_loss_weighting_cfg()
        self._dkal_cfg = self._parse_dkal_cfg()
        
        # ============================================================
        # Residual Refinement Config (NEW)
        # ============================================================
        self._refine_cfg = self._parse_residual_refine_cfg()

    # ============================================================
    # NEW: Parse residual refine config
    # ============================================================
    def _parse_residual_refine_cfg(self) -> Dict[str, Any]:
        c = self._safe_get(self.cfg_model, "residual_refine", None)
        if c is None:
            return {
                "enabled": False,
                "loss_coef": 0.1,
                "loss_coef_anneal_steps": 20000,
                "w_xy": 1.0,
                "w_yaw": 0.3,
                "use_huber": True,
                "huber_delta": 0.1,
                "apply_to": "vehicle",
                "require_exists_both": True,
                "use_refine_in_eval": True,
                "refine_scale": 1.0,
            }
        return {
            "enabled": bool(self._safe_get(c, "enabled", False)),
            "loss_coef": float(self._safe_get(c, "loss_coef", 0.1)),
            "loss_coef_anneal_steps": int(self._safe_get(c, "loss_coef_anneal_steps", 20000)),
            "w_xy": float(self._safe_get(c, "w_xy", 1.0)),
            "w_yaw": float(self._safe_get(c, "w_yaw", 0.3)),
            "use_huber": bool(self._safe_get(c, "use_huber", True)),
            "huber_delta": float(self._safe_get(c, "huber_delta", 0.1)),
            "apply_to": str(self._safe_get(c, "apply_to", "vehicle")),
            "require_exists_both": bool(self._safe_get(c, "require_exists_both", True)),
            "use_refine_in_eval": bool(self._safe_get(c, "use_refine_in_eval", True)),
            "refine_scale": float(self._safe_get(c, "refine_scale", 1.0)),
        }

    @staticmethod
    def _safe_get(node: Any, key: str, default: Any) -> Any:
        if node is None:
            return default
        try:
            val = getattr(node, key, None)
            return val if val is not None else default
        except Exception:
            return default

    def _parse_action_loss_weighting_cfg(self) -> Dict[str, Any]:
        c = self._safe_get(self.cfg_model, "action_loss_weighting", None)
        if c is None:
            return {"enabled": False, "mode": "last_k", "last_k": 8, "early_weight": 0.2}
        return {
            "enabled": bool(self._safe_get(c, "enabled", False)),
            "mode": str(self._safe_get(c, "mode", "last_k")),
            "last_k": int(self._safe_get(c, "last_k", 8)),
            "early_weight": float(self._safe_get(c, "early_weight", 0.2)),
        }

    def _parse_dkal_cfg(self) -> Dict[str, Any]:
        c = self._safe_get(self.cfg_model, "dkal", None)
        if c is None:
            return {
                "enabled": False,
                "coef": 0.05,
                "coef_anneal_steps": 30000,
                "beta": 1.0,
                "apply_to": "vehicle",
                "cost_mode": "match_phys",
                "center_logits": True,
                "center_cost": True,
                "detach_cost": True,
                "use_logits": "raw",
            }
        return {
            "enabled": bool(self._safe_get(c, "enabled", False)),
            "coef": float(self._safe_get(c, "coef", 0.05)),
            "coef_anneal_steps": int(self._safe_get(c, "coef_anneal_steps", 30000)),
            "beta": float(self._safe_get(c, "beta", 1.0)),
            "apply_to": str(self._safe_get(c, "apply_to", "vehicle")),
            "cost_mode": str(self._safe_get(c, "cost_mode", "match_phys")),
            "center_logits": bool(self._safe_get(c, "center_logits", True)),
            "center_cost": bool(self._safe_get(c, "center_cost", True)),
            "detach_cost": bool(self._safe_get(c, "detach_cost", True)),
            "use_logits": str(self._safe_get(c, "use_logits", "raw")),
        }

    def _parse_phys_prior_cfg(self) -> Dict[str, Any]:
        phys = self._safe_get(self.cfg_model, "phys_prior", None)
        if phys is None:
            return self._default_phys_prior_cfg()

        hard = self._safe_get(phys, "hard_mask", None)
        weights = self._safe_get(phys, "weights", None)

        return {
            "enabled": bool(self._safe_get(phys, "enabled", False)),
            "apply_in_train": bool(self._safe_get(phys, "apply_in_train", False)),
            "apply_in_eval": bool(self._safe_get(phys, "apply_in_eval", False)),
            "beta": float(self._safe_get(phys, "beta", 1.0)),
            "beta_anneal_steps": int(self._safe_get(phys, "beta_anneal_steps", 30000)),
            "apply_to": str(self._safe_get(phys, "apply_to", "vehicle")),
            "cost_mode": str(self._safe_get(phys, "cost_mode", "normalized")),
            "only_apply_on_exists": bool(self._safe_get(phys, "only_apply_on_exists", True)),
            "yaw_rate_max": float(self._safe_get(phys, "yaw_rate_max", 1.5)),
            "arc_res_max": float(self._safe_get(phys, "arc_res_max", 0.35)),
            "kappa_max": float(self._safe_get(phys, "kappa_max", 0.25)),
            "a_lat_max": float(self._safe_get(phys, "a_lat_max", 8.0)),
            "a_long_max": float(self._safe_get(phys, "a_long_max", 6.0)),
            "reverse_allow": float(self._safe_get(phys, "reverse_allow", 0.05)),
            "w_reverse": float(self._safe_get(weights, "reverse", 1.0)) if weights else 1.0,
            "w_yaw_rate": float(self._safe_get(weights, "yaw_rate", 1.0)) if weights else 1.0,
            "w_arc": float(self._safe_get(weights, "arc", 0.8)) if weights else 0.8,
            "w_kappa": float(self._safe_get(weights, "curvature", 0.2)) if weights else 0.2,
            "w_a_lat": float(self._safe_get(weights, "a_lat", 0.5)) if weights else 0.5,
            "w_a_long": float(self._safe_get(weights, "a_long", 0.3)) if weights else 0.3,
            "hard_mask_enabled": bool(self._safe_get(hard, "enabled", False)) if hard else False,
            "yaw_rate_hard_max": float(self._safe_get(hard, "yaw_rate_hard_max", 6.0)) if hard else 6.0,
            "arc_res_hard_max": float(self._safe_get(hard, "arc_res_hard_max", 1.2)) if hard else 1.2,
            "max_tokens_to_hard_mask": int(self._safe_get(hard, "max_tokens_to_mask", 64)) if hard else 64,
            "hard_mask_logit_value": float(self._safe_get(hard, "logit_value", _HARD_MASK_LOGIT_SAFE_DEFAULT)) if hard else _HARD_MASK_LOGIT_SAFE_DEFAULT,
            "log_stats": bool(self._safe_get(phys, "log_stats", False)),
        }

    def _default_phys_prior_cfg(self) -> Dict[str, Any]:
        return {
            "enabled": False,
            "apply_in_train": False,
            "apply_in_eval": False,
            "beta": 1.0,
            "beta_anneal_steps": 30000,
            "apply_to": "vehicle",
            "cost_mode": "normalized",
            "only_apply_on_exists": True,
            "yaw_rate_max": 1.5,
            "arc_res_max": 0.35,
            "kappa_max": 0.25,
            "a_lat_max": 8.0,
            "a_long_max": 6.0,
            "reverse_allow": 0.05,
            "w_reverse": 1.0,
            "w_yaw_rate": 1.0,
            "w_arc": 0.8,
            "w_kappa": 0.2,
            "w_a_lat": 0.5,
            "w_a_long": 0.3,
            "hard_mask_enabled": False,
            "yaw_rate_hard_max": 6.0,
            "arc_res_hard_max": 1.2,
            "max_tokens_to_hard_mask": 64,
            "hard_mask_logit_value": _HARD_MASK_LOGIT_SAFE_DEFAULT,
            "log_stats": False,
        }

    def _parse_expected_phys_loss_cfg(self) -> Dict[str, Any]:
        c = self._safe_get(self.cfg_model, "expected_phys_loss", None)
        if c is None:
            return {"enabled": False, "coef": 0.01, "coef_anneal_steps": 30000, "use_shaped_logits": False, "apply_to": "vehicle", "use_cost_mode": "match_phys"}
        return {
            "enabled": bool(self._safe_get(c, "enabled", False)),
            "coef": float(self._safe_get(c, "coef", 0.01)),
            "coef_anneal_steps": int(self._safe_get(c, "coef_anneal_steps", 30000)),
            "use_shaped_logits": bool(self._safe_get(c, "use_shaped_logits", False)),
            "apply_to": str(self._safe_get(c, "apply_to", "vehicle")),
            "use_cost_mode": str(self._safe_get(c, "use_cost_mode", "match_phys")),
        }

    def _parse_risk_se2_loss_cfg(self) -> Dict[str, Any]:
        c = self._safe_get(self.cfg_model, "risk_se2_loss", None)
        if c is None:
            return {"enabled": False, "coef": 0.02, "w_xy": 1.0, "w_yaw": 0.2, "use_shaped_logits": True, "apply_to": "vehicle"}
        return {
            "enabled": bool(self._safe_get(c, "enabled", False)),
            "coef": float(self._safe_get(c, "coef", 0.02)),
            "w_xy": float(self._safe_get(c, "w_xy", 1.0)),
            "w_yaw": float(self._safe_get(c, "w_yaw", 0.2)),
            "use_shaped_logits": bool(self._safe_get(c, "use_shaped_logits", True)),
            "apply_to": str(self._safe_get(c, "apply_to", "vehicle")),
        }

    def _parse_temporal_smooth_loss_cfg(self) -> Dict[str, Any]:
        c = self._safe_get(self.cfg_model, "temporal_smooth_loss", None)
        if c is None:
            return {"enabled": False, "coef": 0.03, "coef_anneal_steps": 30000, "w_xy": 1.0, "w_yaw": 0.2, "use_shaped_logits": False, "apply_to": "vehicle", "require_exists_t_and_tm1": True}
        return {
            "enabled": bool(self._safe_get(c, "enabled", False)),
            "coef": float(self._safe_get(c, "coef", 0.03)),
            "coef_anneal_steps": int(self._safe_get(c, "coef_anneal_steps", 30000)),
            "w_xy": float(self._safe_get(c, "w_xy", 1.0)),
            "w_yaw": float(self._safe_get(c, "w_yaw", 0.2)),
            "use_shaped_logits": bool(self._safe_get(c, "use_shaped_logits", False)),
            "apply_to": str(self._safe_get(c, "apply_to", "vehicle")),
            "require_exists_t_and_tm1": bool(self._safe_get(c, "require_exists_t_and_tm1", True)),
        }

    def _init_token_vocab_and_features(self) -> None:
        vocab_path = self.cfg_dataset.k_disks_vocab_path
        if not os.path.exists(vocab_path):
            raise FileNotFoundError(f"k_disks_vocab_path not found: {vocab_path}")

        with open(vocab_path, "rb") as f:
            V_np = np.array(pickle.load(f)["V"], dtype=np.float32)

        if V_np.ndim != 2 or V_np.shape[1] != 3:
            raise ValueError(f"Invalid vocab shape: {V_np.shape}")

        V = torch.from_numpy(V_np)
        if int(V.shape[0]) != int(self.action_dim):
            raise ValueError(f"vocab_size mismatch: {self.action_dim} vs {V.shape[0]}")

        self.register_buffer("k_disks_vocab", V, persistent=False)

        dx, dy = V[:, 0], V[:, 1]
        dpsi = _wrap_to_pi_torch(V[:, 2])
        d = torch.sqrt(dx * dx + dy * dy).clamp(min=1e-6)

        self.register_buffer("token_dx", dx, persistent=False)
        self.register_buffer("token_dy", dy, persistent=False)
        self.register_buffer("token_dpsi", dpsi, persistent=False)
        self.register_buffer("token_d", d, persistent=False)
        self.register_buffer("token_speed", d / float(self.delta_t), persistent=False)
        self.register_buffer("token_yaw_rate_abs", torch.abs(dpsi) / float(self.delta_t), persistent=False)
        self.register_buffer("token_kappa_abs", torch.abs(2.0 * torch.sin(dpsi / 2.0) / d), persistent=False)
        self.register_buffer("token_arc_res_abs", torch.abs(_wrap_to_pi_torch(torch.atan2(dy, dx) - dpsi / 2.0)), persistent=False)

    # ============================================================
    # NEW: Compute GT local delta from agent states (GPU tensor)
    # ============================================================
    def _compute_gt_local_delta(
        self,
        agent_states: torch.Tensor,  # [B, A, T, D]
    ) -> torch.Tensor:
        """Compute ground truth local delta in body frame.
        
        For each agent at each timestep t, compute the local transform
        from state[t] to state[t+1] in the body frame of state[t].
        
        Returns: [B, A, T-1, 3] where dim=-1 is (dx, dy, dtheta)
        """
        # agent_states layout: [pos_x, pos_y, vel_x, vel_y, heading, length, width, is_ego, existence]
        # Indices: 0=pos_x, 1=pos_y, 4=heading
        
        B, A, T, D = agent_states.shape
        if T < 2:
            return torch.zeros((B, A, 0, 3), device=agent_states.device, dtype=agent_states.dtype)
        
        # Extract positions and headings
        x_t = agent_states[:, :, :-1, 0]      # [B, A, T-1]
        y_t = agent_states[:, :, :-1, 1]
        theta_t = agent_states[:, :, :-1, 4]
        
        x_tp1 = agent_states[:, :, 1:, 0]     # [B, A, T-1]
        y_tp1 = agent_states[:, :, 1:, 1]
        theta_tp1 = agent_states[:, :, 1:, 4]
        
        # Global displacement
        global_dx = x_tp1 - x_t
        global_dy = y_tp1 - y_t
        
        # Rotate to body frame (rotate by -theta_t)
        cos_theta = torch.cos(-theta_t)
        sin_theta = torch.sin(-theta_t)
        local_dx = cos_theta * global_dx - sin_theta * global_dy
        local_dy = sin_theta * global_dx + cos_theta * global_dy
        
        # Heading change (wrap to [-pi, pi])
        local_dtheta = _wrap_to_pi_torch(theta_tp1 - theta_t)
        
        return torch.stack([local_dx, local_dy, local_dtheta], dim=-1)  # [B, A, T-1, 3]

    # ============================================================
    # NEW: Compute residual target
    # ============================================================
    def _compute_residual_target(
        self,
        gt_local_delta: torch.Tensor,  # [B, A, T-1, 3]
        action_tokens: torch.Tensor,    # [B, A, T] or [B, A, T-1]
    ) -> torch.Tensor:
        """Compute residual target = GT delta - Anchor delta.
        
        Returns: [B, A, T-1, 3]
        """
        device = gt_local_delta.device
        dtype = gt_local_delta.dtype
        
        # Handle action token dimensions
        if action_tokens.shape[-1] == gt_local_delta.shape[-2] + 1:
            # Standard mode: actions has T tokens, we need T-1
            action_tokens = action_tokens[:, :, :-1]  # [B, A, T-1]
        
        # Clamp token indices
        action_tokens = action_tokens.long().clamp(min=0, max=self.action_dim - 1)
        
        # Lookup anchor delta from vocabulary
        vocab = self.k_disks_vocab.to(device=device, dtype=dtype)  # [K, 3]
        anchor_delta = vocab[action_tokens]  # [B, A, T-1, 3]
        
        # Residual = GT - Anchor
        residual = gt_local_delta - anchor_delta
        
        return residual

    # ============================================================
    # NEW: Residual refinement loss
    # ============================================================
    def _residual_refine_loss(
        self,
        refine_preds: torch.Tensor,     # [B, A, T, 3]
        agent_states: torch.Tensor,     # [B, A, T, D]
        action_tokens: torch.Tensor,    # [B, A, T]
        agent_types: torch.Tensor,      # [B, A, type_dim]
        cfg_refine: Dict[str, Any],
        trajeglish: bool = False,
    ) -> torch.Tensor:
        """Compute residual refinement loss.
        
        Loss = weighted_huber(refine_preds, residual_target)
        """
        if not bool(cfg_refine["enabled"]):
            return torch.zeros((), device=refine_preds.device, dtype=refine_preds.dtype)
        
        device = refine_preds.device
        dtype = refine_preds.dtype
        B, A, T, _ = refine_preds.shape
        
        # Compute GT local delta
        gt_local_delta = self._compute_gt_local_delta(agent_states)  # [B, A, T-1, 3]
        
        # Handle trajeglish time alignment
        if trajeglish:
            # In trajeglish, refine_preds[:, :, :-1] corresponds to action t predicting state t+1
            refine_preds_aligned = refine_preds[:, :, :-1, :]  # [B, A, T-1, 3]
            actions_aligned = action_tokens[:, :, 1:]          # [B, A, T-1]
            exists_t = agent_states[:, :, :-1, -1] > 0
            exists_tp1 = agent_states[:, :, 1:, -1] > 0
        else:
            # Standard mode: refine_preds[:, :, t] is for transition t->t+1
            # We can only supervise up to T-1
            if T < 2:
                return torch.zeros((), device=device, dtype=dtype)
            refine_preds_aligned = refine_preds[:, :, :-1, :]  # [B, A, T-1, 3]
            actions_aligned = action_tokens[:, :, :-1]         # [B, A, T-1]
            exists_t = agent_states[:, :, :-1, -1] > 0
            exists_tp1 = agent_states[:, :, 1:, -1] > 0
        
        # Compute residual target
        residual_target = self._compute_residual_target(gt_local_delta, actions_aligned)  # [B, A, T-1, 3]
        
        # Build mask
        if bool(cfg_refine["require_exists_both"]):
            mask = (exists_t & exists_tp1).float()  # [B, A, T-1]
        else:
            mask = exists_t.float()
        
        # Apply agent type filter
        apply_mask = self._apply_to_mask(cfg_refine["apply_to"], agent_types).to(device=device)  # [B, A]
        mask = mask * apply_mask.unsqueeze(-1).float()  # [B, A, T-1]
        
        # Weights for different components
        w_xy = float(cfg_refine["w_xy"])
        w_yaw = float(cfg_refine["w_yaw"])
        
        # Compute loss per component
        diff_xy = refine_preds_aligned[..., :2] - residual_target[..., :2]  # [B, A, T-1, 2]
        diff_yaw = _wrap_to_pi_torch(refine_preds_aligned[..., 2] - residual_target[..., 2])  # [B, A, T-1]
        
        if bool(cfg_refine["use_huber"]):
            delta = float(cfg_refine["huber_delta"])
            loss_xy = F.huber_loss(diff_xy, torch.zeros_like(diff_xy), reduction='none', delta=delta)
            loss_yaw = F.huber_loss(diff_yaw, torch.zeros_like(diff_yaw), reduction='none', delta=delta)
        else:
            loss_xy = diff_xy ** 2
            loss_yaw = diff_yaw ** 2
        
        # Combine with weights
        loss_per_sample = w_xy * loss_xy.sum(dim=-1) + w_yaw * loss_yaw  # [B, A, T-1]
        
        # Apply mask and reduce
        if mask.sum() < 1:
            return torch.zeros((), device=device, dtype=dtype)
        
        loss = (loss_per_sample * mask).sum() / mask.sum().clamp(min=1.0)
        
        return loss

    @staticmethod
    def _vehicle_mask_from_agent_types(agent_types_flat: torch.Tensor) -> torch.Tensor:
        return agent_types_flat[:, 1] > 0.5

    @staticmethod
    def _apply_to_mask(apply_to: str, agent_types: torch.Tensor) -> torch.Tensor:
        """Return mask indicating which agents the loss applies to.
        
        agent_types can be:
          - [N, type_dim] for flat input
          - [B, A, type_dim] for batched input
        """
        if apply_to.lower() == "all":
            if agent_types.dim() == 2:
                return torch.ones((agent_types.shape[0],), dtype=torch.bool, device=agent_types.device)
            else:  # [B, A, type_dim]
                return torch.ones((agent_types.shape[0], agent_types.shape[1]), dtype=torch.bool, device=agent_types.device)
        if apply_to.lower() == "vehicle":
            if agent_types.dim() == 2:
                return agent_types[:, 1] > 0.5
            else:  # [B, A, type_dim]
                return agent_types[:, :, 1] > 0.5
        raise ValueError(f"Unknown apply_to='{apply_to}'")

    def _compute_physics_cost_matrix(self, agent_states_flat, agent_types_flat, apply_to, cfg_phys, dtype, cost_mode):
        device = agent_states_flat.device
        N, V = agent_states_flat.shape[0], int(self.action_dim)

        if V == 0 or self.k_disks_vocab.numel() == 0:
            return torch.zeros((N, V), device=device, dtype=dtype)

        vx, vy = agent_states_flat[:, 2].to(dtype), agent_states_flat[:, 3].to(dtype)
        v_curr = torch.sqrt(vx * vx + vy * vy).unsqueeze(-1)

        dx = self.token_dx.to(device=device, dtype=dtype).unsqueeze(0)
        yaw_rate_abs = self.token_yaw_rate_abs.to(device=device, dtype=dtype).unsqueeze(0)
        arc_res_abs = self.token_arc_res_abs.to(device=device, dtype=dtype).unsqueeze(0)
        kappa_abs = self.token_kappa_abs.to(device=device, dtype=dtype).unsqueeze(0)
        v_tok = self.token_speed.to(device=device, dtype=dtype).unsqueeze(0)

        v_ref = torch.maximum(v_curr, v_tok)
        reverse_allow = float(cfg_phys["reverse_allow"])

        reverse_violation = F.relu(-(dx + reverse_allow))
        yaw_rate_violation = F.relu(yaw_rate_abs - float(cfg_phys["yaw_rate_max"]))
        arc_violation = F.relu(arc_res_abs - float(cfg_phys["arc_res_max"]))
        kappa_violation = F.relu(kappa_abs - float(cfg_phys["kappa_max"]))
        a_lat = (v_ref * v_ref) * kappa_abs
        a_lat_violation = F.relu(a_lat - float(cfg_phys["a_lat_max"]))
        a_long = (v_tok - v_curr) / float(self.delta_t)
        a_long_violation = F.relu(torch.abs(a_long) - float(cfg_phys["a_long_max"]))

        if cost_mode.lower() == "normalized":
            eps = 1e-6
            cost = (
                float(cfg_phys["w_reverse"]) * (reverse_violation / max(reverse_allow, eps)) ** 2
                + float(cfg_phys["w_yaw_rate"]) * (yaw_rate_violation / max(float(cfg_phys["yaw_rate_max"]), eps)) ** 2
                + float(cfg_phys["w_arc"]) * (arc_violation / max(float(cfg_phys["arc_res_max"]), eps)) ** 2
                + float(cfg_phys["w_kappa"]) * (kappa_violation / max(float(cfg_phys["kappa_max"]), eps)) ** 2
                + float(cfg_phys["w_a_lat"]) * (a_lat_violation / max(float(cfg_phys["a_lat_max"]), eps)) ** 2
                + float(cfg_phys["w_a_long"]) * (a_long_violation / max(float(cfg_phys["a_long_max"]), eps)) ** 2
            )
        else:
            cost = (
                float(cfg_phys["w_reverse"]) * (reverse_violation ** 2)
                + float(cfg_phys["w_yaw_rate"]) * (yaw_rate_violation ** 2)
                + float(cfg_phys["w_arc"]) * (arc_violation ** 2)
                + float(cfg_phys["w_kappa"]) * (kappa_violation ** 2)
                + float(cfg_phys["w_a_lat"]) * (a_lat_violation ** 2)
                + float(cfg_phys["w_a_long"]) * (a_long_violation ** 2)
            )

        # For flat input
        if agent_types_flat.dim() == 2:
            apply_mask = self._vehicle_mask_from_agent_types(agent_types_flat) if apply_to.lower() == "vehicle" else torch.ones(N, device=device, dtype=torch.bool)
        else:
            apply_mask = torch.ones(N, device=device, dtype=torch.bool)
        
        return cost * apply_mask.to(dtype=dtype).unsqueeze(-1)

    def _hard_token_mask(self, cfg_phys, device, dtype):
        yaw_rate_abs = self.token_yaw_rate_abs.to(device=device, dtype=dtype)
        arc_res_abs = self.token_arc_res_abs.to(device=device, dtype=dtype)

        hard = (yaw_rate_abs > float(cfg_phys["yaw_rate_hard_max"])) | (arc_res_abs > float(cfg_phys["arc_res_hard_max"]))
        if hard.numel() > 0:
            hard = hard.clone()
            hard[0] = False

        max_k = int(cfg_phys["max_tokens_to_hard_mask"])
        if max_k > 0 and int(hard.sum().item()) > max_k:
            score = yaw_rate_abs + 0.5 * arc_res_abs
            score = score.clone()
            if score.numel() > 0:
                score[0] = -1e9
            topk = torch.topk(score, k=max_k).indices
            hard = torch.zeros_like(hard)
            hard[topk] = True
        return hard

    def _apply_hard_mask(self, logits_flat, hard_tokens, row_apply, logit_value):
        if logits_flat.numel() == 0 or hard_tokens is None or not hard_tokens.any():
            return logits_flat
        if row_apply is None or not row_apply.any():
            return logits_flat
        mask2d = row_apply.unsqueeze(1) & hard_tokens.unsqueeze(0)
        return logits_flat.masked_fill(mask2d, float(logit_value))

    def _maybe_shape_action_logits(self, data, action_logits, eval_mode):
        cfg_phys = self._phys_cfg
        B, A, T, V = action_logits.shape

        apply_shape = bool(cfg_phys["enabled"])
        if apply_shape and eval_mode and (not bool(cfg_phys["apply_in_eval"])):
            apply_shape = False
        if apply_shape and (not eval_mode) and (not bool(cfg_phys["apply_in_train"])):
            apply_shape = False

        agent_states = data["agent"].agent_states[:, :, :T, :]
        agent_types = data["agent"].agent_types
        agent_types_rep = agent_types.unsqueeze(2).expand(B, A, T, agent_types.shape[-1])

        agent_states_flat = agent_states.reshape(B * A * T, -1)
        agent_types_flat = agent_types_rep.reshape(B * A * T, -1)
        logits_flat = action_logits.reshape(B * A * T, V).float()

        raw = logits_flat

        if not apply_shape:
            shaped = logits_flat
            return shaped.reshape(B, A, T, V), raw.reshape(B, A, T, V), shaped.reshape(B, A, T, V), None

        cost = self._compute_physics_cost_matrix(
            agent_states_flat,
            agent_types_flat,
            cfg_phys["apply_to"],
            cfg_phys,
            logits_flat.dtype,
            cfg_phys["cost_mode"],
        )

        if bool(cfg_phys["only_apply_on_exists"]):
            exists = (agent_states_flat[:, -1] > 0).to(dtype=logits_flat.dtype)
            cost = cost * exists.unsqueeze(-1)

        beta = float(cfg_phys["beta"])
        if not eval_mode:
            beta *= _linear_warmup_factor(int(self.global_step), int(cfg_phys["beta_anneal_steps"]))

        shaped_no_hard = logits_flat - beta * cost
        shaped_out = shaped_no_hard
        hard_tokens = None

        if eval_mode and bool(cfg_phys["hard_mask_enabled"]):
            hard_tokens = self._hard_token_mask(cfg_phys, logits_flat.device, logits_flat.dtype)
            row_apply = self._vehicle_mask_from_agent_types(agent_types_flat) if cfg_phys["apply_to"].lower() == "vehicle" else torch.ones(agent_types_flat.shape[0], device=logits_flat.device, dtype=torch.bool)
            if bool(cfg_phys["only_apply_on_exists"]):
                row_apply = row_apply & (agent_states_flat[:, -1] > 0)
            shaped_out = self._apply_hard_mask(shaped_no_hard, hard_tokens, row_apply, float(cfg_phys["hard_mask_logit_value"]))

        return (
            shaped_out.reshape(B, A, T, V),
            raw.reshape(B, A, T, V),
            shaped_no_hard.reshape(B, A, T, V),
            hard_tokens,
        )

    def forward(self, data, eval: bool = False) -> Dict[str, torch.Tensor]:
        if hasattr(self.decoder, "set_rtg_schedule"):
            self.decoder.set_rtg_schedule(int(self.global_step), eval=eval)
        self.decoder.runtime_rtg_tilt = float(self.runtime_rtg_tilt)

        scene_enc = self.encoder(data, eval)
        pred = self.decoder(data, scene_enc, eval)

        shaped_out, raw, shaped_no_hard, hard_tokens = self._maybe_shape_action_logits(data, pred["action_preds"], eval)

        pred["action_preds"] = shaped_out
        pred["action_preds_shaped"] = shaped_no_hard
        pred["action_preds_raw"] = raw
        if hard_tokens is not None:
            pred["action_hard_token_mask"] = hard_tokens
        return pred

    def _flatten_action_supervision(self, data, preds):
        agent_states = data["agent"].agent_states
        agent_types = data["agent"].agent_types
        trajeglish = bool(self._safe_get(self.cfg_model, "trajeglish", False))

        if trajeglish:
            action_logits = preds["action_preds"][:, :, :-1, :]
            targets = data["agent"].actions[:, :, 1:]
            exists_t = agent_states[:, :, :-1, -1] > 0
            exists_tp1 = agent_states[:, :, 1:, -1] > 0
            mask = (exists_t & exists_tp1).float()
            states_used = agent_states[:, :, :-1, :]
        else:
            action_logits = preds["action_preds"]
            targets = data["agent"].actions
            mask = (agent_states[:, :, :, -1] > 0).float()
            states_used = agent_states

        B, A, T, V = action_logits.shape
        types_rep = agent_types.unsqueeze(2).expand(B, A, T, agent_types.shape[-1])

        return (
            action_logits.reshape(B * A * T, V),
            targets.reshape(-1),
            mask.reshape(-1),
            states_used.reshape(B * A * T, -1),
            types_rep.reshape(B * A * T, -1),
            (B, A, T, V),
        )

    def _dkal_loss(self, logits_flat, states_flat, types_flat, mask_flat, cfg_dkal: Dict[str, Any]):
        if (not bool(cfg_dkal["enabled"])) or logits_flat.numel() == 0:
            return torch.zeros((), device=logits_flat.device, dtype=logits_flat.dtype)

        cost_mode = str(cfg_dkal["cost_mode"]).lower()
        if cost_mode == "match_phys":
            cost_mode = str(self._phys_cfg["cost_mode"]).lower()

        cost = self._compute_physics_cost_matrix(
            states_flat,
            types_flat,
            cfg_dkal["apply_to"],
            self._phys_cfg,
            logits_flat.dtype,
            cost_mode,
        )

        if bool(cfg_dkal.get("detach_cost", True)):
            cost = cost.detach()

        beta = float(cfg_dkal["beta"])

        if bool(cfg_dkal.get("center_logits", True)):
            logits_c = logits_flat - logits_flat.mean(dim=-1, keepdim=True)
        else:
            logits_c = logits_flat

        if bool(cfg_dkal.get("center_cost", True)):
            cost_c = cost - cost.mean(dim=-1, keepdim=True)
        else:
            cost_c = cost

        diff = logits_c + beta * cost_c
        per_row = (diff * diff).mean(dim=-1)

        apply_mask = self._vehicle_mask_from_agent_types(types_flat) if cfg_dkal["apply_to"].lower() == "vehicle" else torch.ones(types_flat.shape[0], device=types_flat.device, dtype=torch.bool)
        active = (mask_flat > 0) & apply_mask
        if int(active.sum().item()) == 0:
            return torch.zeros((), device=logits_flat.device, dtype=logits_flat.dtype)

        return (per_row * active.float()).sum() / active.float().sum().clamp(min=1.0)

    def _expected_phys_loss(self, logits, states_flat, types_flat, mask, cfg_loss):
        if not bool(cfg_loss["enabled"]) or logits.numel() == 0:
            return torch.zeros((), device=logits.device, dtype=logits.dtype)

        cost_mode = cfg_loss["use_cost_mode"]
        if cost_mode == "match_phys":
            cost_mode = self._phys_cfg["cost_mode"]

        cost = self._compute_physics_cost_matrix(states_flat, types_flat, cfg_loss["apply_to"], self._phys_cfg, logits.dtype, cost_mode)
        p = torch.softmax(logits.float(), dim=-1)
        exp_cost = (p * cost).sum(dim=-1)
        return (exp_cost * mask).sum() / mask.sum().clamp(min=1.0)

    def _risk_se2_loss(self, logits, targets, types_flat, mask, cfg_loss):
        if not bool(cfg_loss["enabled"]) or logits.numel() == 0:
            return torch.zeros((), device=logits.device, dtype=logits.dtype)

        device, dtype = logits.device, logits.dtype
        apply_mask = self._vehicle_mask_from_agent_types(types_flat) if cfg_loss["apply_to"].lower() == "vehicle" else torch.ones(types_flat.shape[0], device=device, dtype=torch.bool)
        active = (mask > 0) & apply_mask
        if int(active.sum().item()) == 0:
            return torch.zeros((), device=device, dtype=dtype)

        targets = targets.long().clamp(min=0, max=self.action_dim - 1)
        delta_gt = self.k_disks_vocab.to(device=device, dtype=dtype)[targets]

        dx = self.token_dx.to(device=device, dtype=dtype).unsqueeze(0) - delta_gt[:, 0:1]
        dy = self.token_dy.to(device=device, dtype=dtype).unsqueeze(0) - delta_gt[:, 1:2]
        ddpsi = _wrap_to_pi_torch(self.token_dpsi.to(device=device, dtype=dtype).unsqueeze(0) - delta_gt[:, 2:3])

        dist = float(cfg_loss["w_xy"]) * (dx * dx + dy * dy) + float(cfg_loss["w_yaw"]) * (ddpsi * ddpsi)
        p = torch.softmax(logits.float(), dim=-1)
        exp_dist = (p * dist).sum(dim=-1)
        return (exp_dist * active.float() * mask).sum() / (active.float() * mask).sum().clamp(min=1.0)

    def _temporal_smooth_loss(self, logits_bat, agent_states, agent_types, cfg_loss):
        if not bool(cfg_loss["enabled"]) or logits_bat.numel() == 0:
            return torch.zeros((), device=logits_bat.device, dtype=logits_bat.dtype)

        B, A, T, V = logits_bat.shape
        if T <= 1:
            return torch.zeros((), device=logits_bat.device, dtype=logits_bat.dtype)

        device, dtype = logits_bat.device, logits_bat.dtype
        exists = agent_states[:, :, :, -1] > 0

        if bool(cfg_loss["require_exists_t_and_tm1"]):
            active = exists[:, :, 1:] & exists[:, :, :-1]
        else:
            active = exists[:, :, 1:]

        apply_flat = self._apply_to_mask(cfg_loss["apply_to"], agent_types)
        active = active & apply_flat.unsqueeze(-1)

        if int(active.sum().item()) == 0:
            return torch.zeros((), device=device, dtype=dtype)

        p = torch.softmax(logits_bat.float(), dim=-1).to(dtype=dtype)
        dx = self.token_dx.to(device=device, dtype=dtype)
        dy = self.token_dy.to(device=device, dtype=dtype)

        ex = torch.einsum("batv,v->bat", p, dx)
        ey = torch.einsum("batv,v->bat", p, dy)
        ex2 = torch.einsum("batv,v->bat", p, dx * dx)
        ey2 = torch.einsum("batv,v->bat", p, dy * dy)

        cos_dpsi = torch.cos(self.token_dpsi.to(device=device, dtype=dtype))
        sin_dpsi = torch.sin(self.token_dpsi.to(device=device, dtype=dtype))
        e_cos = torch.einsum("batv,v->bat", p, cos_dpsi)
        e_sin = torch.einsum("batv,v->bat", p, sin_dpsi)

        w_xy, w_yaw = float(cfg_loss["w_xy"]), float(cfg_loss["w_yaw"])
        en2_xy = w_xy * (ex2 + ey2)
        edot_xy = w_xy * (ex[:, :, 1:] * ex[:, :, :-1] + ey[:, :, 1:] * ey[:, :, :-1])
        diff2_xy = (en2_xy[:, :, 1:] + en2_xy[:, :, :-1] - 2.0 * edot_xy).clamp(min=0.0)
        diff2_yaw = (e_cos[:, :, 1:] - e_cos[:, :, :-1]) ** 2 + (e_sin[:, :, 1:] - e_sin[:, :, :-1]) ** 2

        return ((diff2_xy + w_yaw * diff2_yaw) * active.float()).sum() / active.float().sum().clamp(min=1.0)

    def compute_loss(self, data, preds) -> Dict[str, torch.Tensor]:
        loss_dict = {}
        preds_for_loss = dict(preds)

        if "action_preds_shaped" in preds_for_loss:
            preds_for_loss["action_preds"] = preds_for_loss["action_preds_shaped"]

        logits_flat, targets, mask_flat, states_flat, types_flat, shape_info = self._flatten_action_supervision(data, preds_for_loss)
        B, A, T, V = shape_info

        ce = F.cross_entropy(logits_flat.float(), targets.long(), reduction="none")

        if bool(self._actw_cfg["enabled"]):
            trajeglish = bool(self._safe_get(self.cfg_model, "trajeglish", False))
            T_full = data["agent"].agent_states.shape[2]
            T_used = T_full - 1 if trajeglish else T_full
            last_k = min(max(1, int(self._actw_cfg["last_k"])), T_used)

            w_t = torch.full((T_used,), float(self._actw_cfg["early_weight"]), device=logits_flat.device, dtype=mask_flat.dtype)
            w_t[T_used - last_k :] = 1.0
            w = w_t.view(1, 1, T_used).expand(B, A, T_used).reshape(-1)
            loss_actions = (ce * mask_flat * w).sum() / (mask_flat * w).sum().clamp(min=1.0)
        else:
            loss_actions = (ce * mask_flat).sum() / mask_flat.sum().clamp(min=1.0)

        loss_actions = float(self._safe_get(self.cfg_model, "loss_action_coef", 1.0)) * loss_actions
        loss_dict["loss_actions"] = loss_actions

        final_loss = loss_actions

        # RTG loss
        if bool(self._safe_get(self.cfg_model, "predict_rtg", True)):
            rtg_K = int(self._safe_get(self.cfg_dataset, "rtg_discretization", 350))
            if hasattr(self.cfg_dataset, "rtg") and hasattr(self.cfg_dataset.rtg, "discretization"):
                rtg_K = int(self.cfg_dataset.rtg.discretization)
            C = int(self._safe_get(self.cfg_model, "num_reward_components", 1))

            rtg_logits = preds["rtg_preds"].reshape(-1, rtg_K, C)
            rtg_targets = data["agent"].rtgs.reshape(-1, C).long()
            exist = (data["agent"].agent_states[..., -1:] > 0).float()
            rtg_mask = (data["agent"].rtg_mask * exist).reshape(-1, 1).float()

            loss_rtg_total = torch.zeros((), device=logits_flat.device, dtype=logits_flat.dtype)
            for c in range(C):
                lc = F.cross_entropy(rtg_logits[:, :, c].float(), rtg_targets[:, c], reduction="none")
                lc = (lc * rtg_mask[:, 0]).sum() / rtg_mask.sum().clamp(min=1.0)
                loss_dict[f"loss_rtg_c{c}"] = lc
                loss_rtg_total = loss_rtg_total + lc
            loss_dict["loss_rtg"] = loss_rtg_total
            final_loss = final_loss + loss_rtg_total

        # ============================================================
        # Residual Refinement Loss (NEW)
        # ============================================================
        if bool(self._refine_cfg["enabled"]) and ("refine_preds" in preds):
            trajeglish = bool(self._safe_get(self.cfg_model, "trajeglish", False))
            
            coef = float(self._refine_cfg["loss_coef"]) * _linear_warmup_factor(
                int(self.global_step), 
                int(self._refine_cfg["loss_coef_anneal_steps"])
            )
            
            loss_refine = self._residual_refine_loss(
                refine_preds=preds["refine_preds"],
                agent_states=data["agent"].agent_states,
                action_tokens=data["agent"].actions,
                agent_types=data["agent"].agent_types,
                cfg_refine=self._refine_cfg,
                trajeglish=trajeglish,
            )
            
            loss_dict["loss_refine"] = loss_refine.detach()
            final_loss = final_loss + coef * loss_refine

        # DKAL
        if bool(self._dkal_cfg["enabled"]):
            trajeglish = bool(self._safe_get(self.cfg_model, "trajeglish", False))
            use_logits = str(self._dkal_cfg.get("use_logits", "raw")).lower()

            if use_logits == "shaped":
                logits_dkal_bat = preds_for_loss["action_preds"]
            else:
                logits_dkal_bat = preds.get("action_preds_raw", preds_for_loss["action_preds"])

            if trajeglish:
                logits_dkal_bat = logits_dkal_bat[:, :, :-1, :]

            logits_dkal_flat = logits_dkal_bat.reshape(B * A * T, V)

            coef = float(self._dkal_cfg["coef"]) * _linear_warmup_factor(int(self.global_step), int(self._dkal_cfg["coef_anneal_steps"]))
            loss_dkal = self._dkal_loss(logits_dkal_flat, states_flat, types_flat, mask_flat, self._dkal_cfg)

            loss_dict["loss_dkal"] = loss_dkal.detach()
            final_loss = final_loss + coef * loss_dkal

        # Optional auxiliary losses
        if bool(self._exp_phys_cfg["enabled"]):
            logits_used = logits_flat if bool(self._exp_phys_cfg["use_shaped_logits"]) else preds.get("action_preds_raw", logits_flat).reshape_as(logits_flat)
            loss_exp = self._expected_phys_loss(logits_used, states_flat, types_flat, mask_flat, self._exp_phys_cfg)
            coef = float(self._exp_phys_cfg["coef"]) * _linear_warmup_factor(int(self.global_step), int(self._exp_phys_cfg["coef_anneal_steps"]))
            loss_dict["loss_expected_phys"] = loss_exp
            final_loss = final_loss + coef * loss_exp

        if bool(self._risk_se2_cfg["enabled"]):
            logits_used = logits_flat if bool(self._risk_se2_cfg["use_shaped_logits"]) else preds.get("action_preds_raw", logits_flat).reshape_as(logits_flat)
            loss_risk = self._risk_se2_loss(logits_used, targets, types_flat, mask_flat, self._risk_se2_cfg)
            loss_dict["loss_risk_se2"] = loss_risk
            final_loss = final_loss + float(self._risk_se2_cfg["coef"]) * loss_risk

        if bool(self._smooth_cfg["enabled"]):
            logits_bat = preds_for_loss["action_preds"]
            loss_smooth = self._temporal_smooth_loss(logits_bat, data["agent"].agent_states, data["agent"].agent_types, self._smooth_cfg)
            coef = float(self._smooth_cfg["coef"]) * _linear_warmup_factor(int(self.global_step), int(self._smooth_cfg["coef_anneal_steps"]))
            loss_dict["loss_temporal_smooth"] = loss_smooth
            final_loss = final_loss + coef * loss_smooth

        loss_dict["loss_total"] = final_loss
        return loss_dict

    def training_step(self, data, batch_idx):
        preds = self(data)
        loss_dict = self.compute_loss(data, preds)

        self.log("loss", loss_dict["loss_actions"], prog_bar=True, on_step=True, sync_dist=True)
        self.log("loss_total", loss_dict["loss_total"], on_step=True, sync_dist=True)
        if "loss_rtg" in loss_dict:
            self.log("loss_rtg", loss_dict["loss_rtg"], prog_bar=True, on_step=True, sync_dist=True)
        if "loss_refine" in loss_dict:
            self.log("loss_refine", loss_dict["loss_refine"], prog_bar=False, on_step=True, sync_dist=True)
        if "loss_dkal" in loss_dict:
            self.log("loss_dkal", loss_dict["loss_dkal"], prog_bar=False, on_step=True, sync_dist=True)
        self.log("lr", self.trainer.optimizers[0].param_groups[0]["lr"], prog_bar=True, on_step=True, sync_dist=True)
        return loss_dict["loss_total"]

    def validation_step(self, data, batch_idx):
        preds = self(data, eval=True)
        loss_dict = self.compute_loss(data, preds)
        B = preds["action_preds"].shape[0]
        self.log("val_loss", loss_dict["loss_actions"], prog_bar=True, on_epoch=True, sync_dist=True, batch_size=B)
        if "loss_rtg" in loss_dict:
            self.log("val_loss_rtg", loss_dict["loss_rtg"], prog_bar=True, on_epoch=True, sync_dist=True, batch_size=B)
        if "loss_refine" in loss_dict:
            self.log("val_loss_refine", loss_dict["loss_refine"], prog_bar=False, on_epoch=True, sync_dist=True, batch_size=B)

    def on_before_optimizer_step(self, optimizer):
        self.log_dict(grad_norm(self.encoder, norm_type=2))
        self.log_dict(grad_norm(self.decoder, norm_type=2))

    def configure_optimizers(self):
        decay, no_decay = set(), set()
        whitelist = (nn.Linear, nn.Conv1d, nn.Conv2d, nn.MultiheadAttention, nn.LSTM, nn.GRU)
        blacklist = (nn.BatchNorm1d, nn.BatchNorm2d, nn.LayerNorm, nn.Embedding)

        for mn, m in self.named_modules():
            for pn, _ in m.named_parameters(recurse=False):
                fpn = f"{mn}.{pn}" if mn else pn
                if "bias" in pn:
                    no_decay.add(fpn)
                elif "weight" in pn:
                    (decay if isinstance(m, whitelist) else no_decay).add(fpn)
                else:
                    no_decay.add(fpn)

        param_dict = {pn: p for pn, p in self.named_parameters()}
        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(decay)], "weight_decay": float(self.cfg.train.weight_decay)},
            {"params": [param_dict[pn] for pn in sorted(no_decay)], "weight_decay": 0.0},
        ]
        optimizer = torch.optim.AdamW(optim_groups, lr=float(self.cfg.train.lr))
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, create_lambda_lr_linear(self.cfg))
        return [optimizer], {"scheduler": scheduler, "interval": "step", "frequency": 1}
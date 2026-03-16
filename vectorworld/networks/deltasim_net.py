import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from vectorworld.utils.train_helpers import weight_init, get_causal_mask
from vectorworld.utils.layers import ResidualMLP


class GatedMultiheadAttention(nn.Module):
    """Multihead Attention with optional SDPA-output gating (G1)."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float,
        gate_enabled: bool,
        gate_init_bias: float,
    ) -> None:
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads")

        self.embed_dim = int(embed_dim)
        self.num_heads = int(num_heads)
        self.head_dim = self.embed_dim // self.num_heads
        self.dropout = float(dropout)
        self.batch_first = True

        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=True)
        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=True)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=True)

        self.gate_enabled = bool(gate_enabled)
        self.gate_init_bias = float(gate_init_bias)
        self.gate_proj = nn.Linear(self.embed_dim, self.num_heads, bias=True) if self.gate_enabled else None

    def reset_gate_parameters(self) -> None:
        if not self.gate_enabled or self.gate_proj is None:
            return
        nn.init.zeros_(self.gate_proj.weight)
        nn.init.constant_(self.gate_proj.bias, self.gate_init_bias)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        key_padding_mask=None,
        need_weights: bool = False,
        attn_mask=None,
        average_attn_weights: bool = True,
        is_causal: bool = False,
    ):
        if query.dim() != 3:
            raise ValueError("Expected query shape [B,L,E] with batch_first=True")

        B, L, E = query.shape
        _, S, _ = key.shape
        device = query.device
        dtype = query.dtype

        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)

        q = q.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)

        if key_padding_mask is not None:
            kpm = key_padding_mask.to(device=device)
            kpm_add = torch.zeros((B, 1, 1, S), device=device, dtype=dtype)
            kpm_add = kpm_add.masked_fill(kpm.view(B, 1, 1, S), torch.finfo(dtype).min)
            if attn_mask is None:
                attn_mask = kpm_add
            else:
                attn_mask = attn_mask.to(device=device, dtype=dtype)
                if attn_mask.dim() == 2:
                    attn_mask = attn_mask.view(1, 1, L, S) + kpm_add
                else:
                    attn_mask = attn_mask + kpm_add
        else:
            if attn_mask is not None:
                attn_mask = attn_mask.to(device=device, dtype=dtype)

        if hasattr(F, "scaled_dot_product_attention"):
            attn_out = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=attn_mask,
                dropout_p=(self.dropout if self.training else 0.0),
                is_causal=is_causal,
            )
        else:
            scale = 1.0 / math.sqrt(self.head_dim)
            scores = torch.matmul(q, k.transpose(-2, -1)) * scale
            if attn_mask is not None:
                scores = scores + attn_mask
            attn = torch.softmax(scores, dim=-1)
            if self.training and self.dropout > 0.0:
                attn = F.dropout(attn, p=self.dropout, training=True)
            attn_out = torch.matmul(attn, v)

        if self.gate_enabled and self.gate_proj is not None:
            g = torch.sigmoid(self.gate_proj(query))
            g = g.transpose(1, 2).unsqueeze(-1)
            attn_out = attn_out * g

        attn_out = attn_out.transpose(1, 2).contiguous().view(B, L, E)
        out = self.out_proj(attn_out)

        return out, None


class GatedTransformerDecoderLayer(nn.Module):
    """TransformerDecoderLayer with optional gated self-attn (G1)."""

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        dropout: float,
        gate_enabled: bool,
        gate_init_bias: float,
    ) -> None:
        super().__init__()
        self.self_attn = GatedMultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            gate_enabled=gate_enabled,
            gate_init_bias=gate_init_bias,
        )
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model, eps=1e-5)
        self.norm2 = nn.LayerNorm(d_model, eps=1e-5)
        self.norm3 = nn.LayerNorm(d_model, eps=1e-5)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = nn.ReLU(inplace=True)

    def forward(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask=None,
        memory_mask=None,
        tgt_key_padding_mask=None,
        memory_key_padding_mask=None,
        tgt_is_causal: bool = False,
        memory_is_causal: bool = False,
    ) -> torch.Tensor:
        x = tgt

        sa, _ = self.self_attn(
            x, x, x,
            attn_mask=tgt_mask,
            key_padding_mask=tgt_key_padding_mask,
            need_weights=False,
            is_causal=False,
        )
        x = x + self.dropout1(sa)
        x = self.norm1(x)

        ca, _ = self.multihead_attn(
            x, memory, memory,
            attn_mask=memory_mask,
            key_padding_mask=memory_key_padding_mask,
            need_weights=False,
        )
        x = x + self.dropout2(ca)
        x = self.norm2(x)

        ff = self.linear2(self.dropout(self.activation(self.linear1(x))))
        x = x + self.dropout3(ff)
        x = self.norm3(x)
        return x


class CtRLSimMapEncoder(nn.Module):
    """Map Encoder for CtRL-Sim with optional point position embedding."""

    def __init__(self, cfg) -> None:
        super().__init__()
        self.cfg = cfg
        self.cfg_model = cfg.model
        self.cfg_dataset = cfg.dataset

        # Optional point position embedding (new feature)
        map_enc_cfg = getattr(self.cfg_model, "map_encoder", None)
        self.use_point_pos_emb = bool(getattr(map_enc_cfg, "use_point_pos_emb", False)) if map_enc_cfg else False

        self.map_seeds = nn.Parameter(torch.Tensor(1, 1, self.cfg_model.hidden_dim), requires_grad=True)
        nn.init.xavier_uniform_(self.map_seeds)

        self.road_pts_encoder = ResidualMLP(self.cfg_model.map_attr, self.cfg_model.hidden_dim)

        if self.use_point_pos_emb:
            self.pt_pos_emb = nn.Embedding(int(self.cfg_dataset.num_points_per_lane), self.cfg_model.hidden_dim)
        else:
            self.pt_pos_emb = None

        self.road_pts_attn_layer = nn.MultiheadAttention(
            self.cfg_model.hidden_dim,
            num_heads=self.cfg_model.num_heads,
            dropout=float(self.cfg_model.dropout),
            batch_first=False,
        )
        self.norm1 = nn.LayerNorm(self.cfg_model.hidden_dim, eps=1e-5)
        self.norm2 = nn.LayerNorm(self.cfg_model.hidden_dim, eps=1e-5)
        self.map_feats = ResidualMLP(self.cfg_model.hidden_dim, self.cfg_model.hidden_dim)

        self.apply(weight_init)

    def get_road_pts_mask(self, roads: torch.Tensor):
        road_segment_mask = torch.sum(roads[:, :, :, -1], dim=2) == 0
        road_pts_mask = (1.0 - roads[:, :, :, -1]).to(dtype=torch.bool, device=roads.device).view(-1, roads.shape[2])
        road_pts_mask[:, 0][road_pts_mask.sum(-1) == roads.shape[2]] = False
        return road_segment_mask, road_pts_mask

    def forward(self, data):
        road_points = data["map"].road_points.float()
        batch_size = road_points.shape[0]

        road_segment_mask, road_pts_mask = self.get_road_pts_mask(road_points)

        road_pts_feats = self.road_pts_encoder(road_points[:, :, :, : self.cfg_model.map_attr])
        road_pts_feats = road_pts_feats.view(
            batch_size * int(self.cfg_dataset.max_num_lanes),
            int(self.cfg_dataset.num_points_per_lane),
            -1,
        ).permute(1, 0, 2)

        if self.use_point_pos_emb and self.pt_pos_emb is not None:
            idx = torch.arange(int(self.cfg_dataset.num_points_per_lane), device=road_pts_feats.device)
            pos = self.pt_pos_emb(idx)[:, None, :]
            road_pts_feats = road_pts_feats + pos

        map_seeds = self.map_seeds.repeat(1, batch_size * int(self.cfg_dataset.max_num_lanes), 1)

        road_seg_emb = self.road_pts_attn_layer(
            query=map_seeds, key=road_pts_feats, value=road_pts_feats, key_padding_mask=road_pts_mask
        )[0]
        road_seg_emb = self.norm1(road_seg_emb)
        road_seg_emb2 = road_seg_emb + self.map_feats(road_seg_emb)
        road_seg_emb2 = self.norm2(road_seg_emb2)

        road_seg_emb = road_seg_emb2.view(1, batch_size, int(self.cfg_dataset.max_num_lanes), -1)[0]
        road_segment_mask = ~road_segment_mask
        return road_seg_emb, road_segment_mask.bool()


class CtRLSimEncoder(nn.Module):
    """CtRL-Sim Encoder with optional action/rtg token corruption."""

    def __init__(self, cfg) -> None:
        super().__init__()
        self.cfg = cfg
        self.cfg_model = cfg.model
        self.cfg_dataset = cfg.dataset

        self.action_dim = int(self.cfg_dataset.vocab_size)
        
        # RTG discretization - handle both new and legacy config formats
        if hasattr(self.cfg_dataset, 'rtg') and hasattr(self.cfg_dataset.rtg, 'discretization'):
            self.rtg_K = int(self.cfg_dataset.rtg.discretization)
        elif hasattr(self.cfg_dataset, 'rtg_discretization'):
            self.rtg_K = int(self.cfg_dataset.rtg_discretization)
        else:
            self.rtg_K = 350  # Original CtRL-Sim default
            
        self.num_reward_components = int(getattr(self.cfg_model, "num_reward_components", 1))

        # Action corruption config (optional, new feature)
        acfg = getattr(self.cfg_model, "action_corruption", None)
        if acfg is not None:
            self.action_corrupt_enabled = bool(getattr(acfg, "enabled", False))
            self.action_corrupt_prob = float(getattr(acfg, "prob", 0.0))
            self.action_corrupt_mode = str(getattr(acfg, "mode", "random"))
            self.action_corrupt_apply_to = str(getattr(acfg, "apply_to", "vehicle"))
            self.action_corrupt_keep_t0 = bool(getattr(acfg, "keep_t0", True))
        else:
            self.action_corrupt_enabled = False
            self.action_corrupt_prob = 0.0
            self.action_corrupt_mode = "random"
            self.action_corrupt_apply_to = "vehicle"
            self.action_corrupt_keep_t0 = True

        # RTG corruption config (optional, new feature)
        rcfg = getattr(self.cfg_model, "rtg_corruption", None)
        if rcfg is not None:
            self.rtg_corrupt_enabled = bool(getattr(rcfg, "enabled", False))
            self.rtg_corrupt_prob = float(getattr(rcfg, "prob", 0.0))
            self.rtg_corrupt_mode = str(getattr(rcfg, "mode", "random"))
            self.rtg_corrupt_apply_to = str(getattr(rcfg, "apply_to", "vehicle"))
            self.rtg_corrupt_keep_t0 = bool(getattr(rcfg, "keep_t0", True))
        else:
            self.rtg_corrupt_enabled = False
            self.rtg_corrupt_prob = 0.0
            self.rtg_corrupt_mode = "random"
            self.rtg_corrupt_apply_to = "vehicle"
            self.rtg_corrupt_keep_t0 = True

        self.map_encoder = CtRLSimMapEncoder(self.cfg)
        self.embed_state = ResidualMLP(self.cfg_model.state_dim, self.cfg_model.hidden_dim)
        self.embed_action = nn.Embedding(int(self.action_dim), self.cfg_model.hidden_dim)

        # RTG embedding
        self.embed_rtg_bin = nn.Embedding(int(self.rtg_K), self.cfg_model.hidden_dim)
        self.embed_rtg = nn.Linear(self.cfg_model.hidden_dim * self.num_reward_components, self.cfg_model.hidden_dim)

        self.embed_timestep = nn.Embedding(int(self.cfg_dataset.train_context_length), self.cfg_model.hidden_dim)
        self.embed_agent_id = nn.Embedding(int(self.cfg_dataset.max_num_agents), self.cfg_model.hidden_dim)
        self.embed_ln = nn.LayerNorm(self.cfg_model.hidden_dim)

        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=self.cfg_model.hidden_dim,
                nhead=self.cfg_model.num_heads,
                dim_feedforward=self.cfg_model.dim_feedforward,
                dropout=float(self.cfg_model.dropout),
                batch_first=True,
            ),
            num_layers=self.cfg_model.num_transformer_encoder_layers,
        )
        self.apply(weight_init)

    def _apply_to_mask(self, apply_to: str, agent_types_time_major: torch.Tensor) -> torch.Tensor:
        apply_to = apply_to.lower()
        if apply_to == "all":
            return torch.ones(agent_types_time_major.shape[:-1], device=agent_types_time_major.device, dtype=torch.bool)
        if apply_to == "vehicle":
            return agent_types_time_major[..., 1] > 0.5
        raise ValueError(f"Unknown apply_to='{apply_to}'")

    def _maybe_corrupt_actions(self, actions_flat, existence_flat_bool, agent_types_time_major, T, A, eval):
        if eval or (not self.action_corrupt_enabled) or (self.action_corrupt_prob <= 0.0):
            return actions_flat

        B = actions_flat.shape[0]
        device = actions_flat.device

        corrupt = (torch.rand((B, T * A), device=device) < self.action_corrupt_prob) & existence_flat_bool

        if self.action_corrupt_keep_t0:
            corrupt[:, :A] = False

        apply_mask = self._apply_to_mask(self.action_corrupt_apply_to, agent_types_time_major).reshape(B, T * A)
        corrupt = corrupt & apply_mask

        if not corrupt.any():
            return actions_flat

        out = actions_flat.clone()
        if self.action_corrupt_mode == "random":
            repl = torch.randint(low=0, high=self.action_dim, size=actions_flat.shape, device=device, dtype=actions_flat.dtype)
            out[corrupt] = repl[corrupt]
        elif self.action_corrupt_mode == "zero":
            out[corrupt] = 0
        return out

    def _maybe_corrupt_rtgs(self, rtgs_flat, rtg_mask_flat_bool, agent_types_time_major, T, A, eval):
        if eval or (not self.rtg_corrupt_enabled) or (self.rtg_corrupt_prob <= 0.0):
            return rtgs_flat

        B = rtgs_flat.shape[0]
        device = rtgs_flat.device

        corrupt = (torch.rand((B, T * A), device=device) < self.rtg_corrupt_prob) & rtg_mask_flat_bool

        if self.rtg_corrupt_keep_t0:
            corrupt[:, :A] = False

        apply_mask = self._apply_to_mask(self.rtg_corrupt_apply_to, agent_types_time_major).reshape(B, T * A)
        corrupt = corrupt & apply_mask

        if not corrupt.any():
            return rtgs_flat

        out = rtgs_flat.clone()
        if self.rtg_corrupt_mode == "random":
            repl = torch.randint(low=0, high=self.rtg_K, size=out.shape, device=device)
            out[corrupt] = repl[corrupt]
        elif self.rtg_corrupt_mode == "zero":
            out[corrupt] = 0
        elif self.rtg_corrupt_mode == "max":
            out[corrupt] = int(self.rtg_K - 1)
        return out

    def forward(self, data, eval: bool):
        agent_states = data["agent"].agent_states
        batch_size = agent_states.shape[0]
        seq_len = agent_states.shape[2]

        existence_mask = agent_states[:, :, :, -1:]
        agent_types = data["agent"].agent_types
        actions = data["agent"].actions
        rtgs = data["agent"].rtgs
        timesteps = data["agent"].timesteps
        rtg_mask = data["agent"].rtg_mask * existence_mask

        agent_ids = torch.arange(int(self.cfg_dataset.max_num_agents), device=agent_states.device)
        agent_ids = agent_ids.unsqueeze(0).unsqueeze(2).repeat(batch_size, 1, seq_len)

        agent_types_time_major = agent_types.unsqueeze(2).repeat(1, 1, seq_len, 1).transpose(1, 2)
        actions_time_major = actions.transpose(1, 2)
        rtgs_time_major = rtgs.transpose(1, 2)
        agent_states_time_major = agent_states[:, :, :, :-1].transpose(1, 2)
        timesteps_time_major = timesteps.transpose(1, 2)
        agent_ids_time_major = agent_ids.transpose(1, 2)

        existence_time_major = existence_mask.transpose(1, 2)
        rtg_mask_time_major = rtg_mask.transpose(1, 2)

        states = torch.cat([agent_states_time_major, agent_types_time_major], dim=-1)

        if bool(getattr(self.cfg_model, "encode_initial_state", True)):
            initial_existence_mask = existence_time_major[:, 0]

        existence_flat = existence_time_major.reshape(batch_size, seq_len * int(self.cfg_dataset.max_num_agents), 1)
        rtg_mask_flat = rtg_mask_time_major.reshape(batch_size, seq_len * int(self.cfg_dataset.max_num_agents), 1)

        timesteps_flat = timesteps_time_major.reshape(batch_size, seq_len * int(self.cfg_dataset.max_num_agents))
        agent_ids_flat = agent_ids_time_major.reshape(batch_size, seq_len * int(self.cfg_dataset.max_num_agents))
        states_flat = states.reshape(batch_size, seq_len * int(self.cfg_dataset.max_num_agents), int(self.cfg_model.state_dim)).float()
        actions_flat = actions_time_major.reshape(batch_size, seq_len * int(self.cfg_dataset.max_num_agents))
        rtgs_flat = rtgs_time_major.reshape(
            batch_size,
            seq_len * int(self.cfg_dataset.max_num_agents),
            int(self.num_reward_components),
        ).long()

        # Optional corruption
        existence_bool = existence_flat[:, :, 0].bool()
        if self.action_corrupt_enabled and (not eval):
            actions_flat = self._maybe_corrupt_actions(
                actions_flat, existence_bool, agent_types_time_major,
                seq_len, int(self.cfg_dataset.max_num_agents), eval
            )

        if self.rtg_corrupt_enabled and (not eval):
            rtg_mask_bool = rtg_mask_flat[:, :, 0].bool()
            rtgs_flat = self._maybe_corrupt_rtgs(
                rtgs_flat, rtg_mask_bool, agent_types_time_major,
                seq_len, int(self.cfg_dataset.max_num_agents), eval
            )

        timestep_embeddings = self.embed_timestep(timesteps_flat)
        agent_id_embeddings = self.embed_agent_id(agent_ids_flat)
        state_embeddings = self.embed_state(states_flat) + timestep_embeddings + agent_id_embeddings

        if bool(getattr(self.cfg_model, "encode_initial_state", True)):
            initial_state_embeddings = state_embeddings[:, 0 : int(self.cfg_dataset.max_num_agents)]

        action_embeddings = self.embed_action(actions_flat.long()) + timestep_embeddings + agent_id_embeddings

        # RTG embeddings
        rtg_emb_list = []
        for c in range(int(self.num_reward_components)):
            rtg_emb_list.append(self.embed_rtg_bin(rtgs_flat[:, :, c].clamp(min=0, max=self.rtg_K - 1)))
        rtg_cat = torch.cat(rtg_emb_list, dim=-1)
        rtg_embeddings = self.embed_rtg(rtg_cat) + timestep_embeddings + agent_id_embeddings

        state_embeddings = state_embeddings * existence_flat.float()
        action_embeddings = action_embeddings * existence_flat.float()
        rtg_embeddings = rtg_embeddings * rtg_mask_flat.float()

        if bool(getattr(self.cfg_model, "encode_initial_state", True)):
            initial_state_embeddings = initial_state_embeddings * initial_existence_mask.float()
            initial_existence_mask = initial_existence_mask[:, :, 0].bool()

        trajeglish = bool(getattr(self.cfg_model, "trajeglish", False))
        il = bool(getattr(self.cfg_model, "il", False))
        
        if trajeglish:
            stacked_embeddings = (
                action_embeddings.unsqueeze(1)
                .permute(0, 2, 1, 3)
                .reshape(batch_size, 1 * seq_len * int(self.cfg_dataset.max_num_agents), self.cfg_model.hidden_dim)
            )
        elif il:
            stacked_embeddings = (
                torch.stack((state_embeddings, action_embeddings), dim=1)
                .permute(0, 2, 1, 3)
                .reshape(batch_size, seq_len * int(self.cfg_dataset.max_num_agents) * 2, self.cfg_model.hidden_dim)
            )
        else:
            stacked_embeddings = (
                torch.stack((state_embeddings, rtg_embeddings, action_embeddings), dim=1)
                .permute(0, 2, 1, 3)
                .reshape(batch_size, seq_len * int(self.cfg_dataset.max_num_agents) * 3, self.cfg_model.hidden_dim)
            )

        stacked_embeddings = self.embed_ln(stacked_embeddings)

        polyline_embeddings, valid_mask = self.map_encoder(data)

        if bool(getattr(self.cfg_model, "encode_initial_state", True)):
            pre_encoder_embeddings = torch.cat([polyline_embeddings, initial_state_embeddings], dim=1)
            src_key_padding_mask = ~torch.cat([valid_mask, initial_existence_mask], dim=1)
        else:
            pre_encoder_embeddings = polyline_embeddings
            src_key_padding_mask = ~valid_mask

        encoder_embeddings = self.transformer_encoder(pre_encoder_embeddings, src_key_padding_mask=src_key_padding_mask)
        return {
            "stacked_embeddings": stacked_embeddings,
            "encoder_embeddings": encoder_embeddings,
            "src_key_padding_mask": src_key_padding_mask,
        }

class CtRLSimDecoder(nn.Module):
    """CtRL-Sim Decoder with optional gated self-attn + RTG conditioning + Residual Refinement."""

    def __init__(self, cfg) -> None:
        super().__init__()
        self.cfg = cfg
        self.cfg_model = cfg.model
        self.cfg_dataset = cfg.dataset

        self.action_dim = int(self.cfg_dataset.vocab_size)
        
        # RTG discretization - handle both new and legacy config formats
        if hasattr(self.cfg_dataset, 'rtg') and hasattr(self.cfg_dataset.rtg, 'discretization'):
            self.rtg_K = int(self.cfg_dataset.rtg.discretization)
        elif hasattr(self.cfg_dataset, 'rtg_discretization'):
            self.rtg_K = int(self.cfg_dataset.rtg_discretization)
        else:
            self.rtg_K = 350
            
        self.num_reward_components = int(getattr(self.cfg_model, "num_reward_components", 1))

        # Gated attention config (optional)
        gate_cfg = getattr(self.cfg_model, "attn_gate", None)
        if gate_cfg is not None:
            self.attn_gate_enabled = bool(getattr(gate_cfg, "enabled", False))
            self.attn_gate_init_bias = float(getattr(gate_cfg, "init_bias", -2.0))
        else:
            self.attn_gate_enabled = False
            self.attn_gate_init_bias = -2.0

        # RTG conditioning config (optional)
        rcfg = getattr(self.cfg_model, "rtg_conditioning", None)
        if rcfg is not None:
            self.rtg_cond_enabled = bool(getattr(rcfg, "enabled", False))
            self.rtg_cond_anneal_steps = int(getattr(rcfg, "anneal_steps", 30000))
            self.rtg_cond_alpha_start = float(getattr(rcfg, "alpha_start", 1.0))
            self.rtg_cond_alpha_end = float(getattr(rcfg, "alpha_end", 0.0))
            self.rtg_cond_stopgrad_pred = bool(getattr(rcfg, "stopgrad_pred", True))
            self.rtg_cond_use_tilt_in_eval = bool(getattr(rcfg, "use_tilt_in_eval", True))
        else:
            self.rtg_cond_enabled = False
            self.rtg_cond_anneal_steps = 30000
            self.rtg_cond_alpha_start = 1.0
            self.rtg_cond_alpha_end = 0.0
            self.rtg_cond_stopgrad_pred = True
            self.rtg_cond_use_tilt_in_eval = True

        if self.rtg_cond_enabled and (not bool(getattr(self.cfg_model, "predict_rtg", True))):
            raise ValueError("rtg_conditioning.enabled=true requires model.predict_rtg=true")

        # ============================================================
        # Residual Refinement Head Config (NEW)
        # ============================================================
        refine_cfg = getattr(self.cfg_model, "residual_refine", None)
        if refine_cfg is not None:
            self.refine_enabled = bool(getattr(refine_cfg, "enabled", False))
            self.refine_use_in_eval = bool(getattr(refine_cfg, "use_refine_in_eval", True))
            self.refine_scale = float(getattr(refine_cfg, "refine_scale", 1.0))
        else:
            self.refine_enabled = False
            self.refine_use_in_eval = True
            self.refine_scale = 1.0

        self._rtg_alpha = 0.0
        self.runtime_rtg_tilt = 0.0

        base_dropout = float(getattr(self.cfg_model, "dropout", 0.0))

        # Build decoder - use gated layers only if enabled
        if self.attn_gate_enabled:
            decoder_layer = GatedTransformerDecoderLayer(
                d_model=self.cfg_model.hidden_dim,
                dim_feedforward=self.cfg_model.dim_feedforward,
                nhead=self.cfg_model.num_heads,
                dropout=base_dropout,
                gate_enabled=True,
                gate_init_bias=self.attn_gate_init_bias,
            )
        else:
            decoder_layer = nn.TransformerDecoderLayer(
                d_model=self.cfg_model.hidden_dim,
                nhead=self.cfg_model.num_heads,
                dim_feedforward=self.cfg_model.dim_feedforward,
                dropout=base_dropout,
                batch_first=True,
            )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=self.cfg_model.num_decoder_layers)

        # Action classification head (discrete tokens)
        self.predict_action = ResidualMLP(
            input_dim=self.cfg_model.hidden_dim,
            hidden_dim=self.cfg_model.hidden_dim,
            n_hidden=2,
            output_dim=self.action_dim,
        )

        # ============================================================
        # Residual Refinement Head (NEW)
        # Output: (delta_x, delta_y, delta_theta) in body frame
        # ============================================================
        if self.refine_enabled:
            self.predict_refine = ResidualMLP(
                input_dim=self.cfg_model.hidden_dim,
                hidden_dim=self.cfg_model.hidden_dim,
                n_hidden=2,
                output_dim=3,  # dx, dy, dtheta
            )
        else:
            self.predict_refine = None

        # RTG prediction head
        if bool(getattr(self.cfg_model, "predict_rtg", True)):
            self.predict_rtg = ResidualMLP(
                input_dim=self.cfg_model.hidden_dim,
                hidden_dim=self.cfg_model.hidden_dim,
                n_hidden=2,
                output_dim=self.rtg_K * self.num_reward_components,
            )

        # RTG conditioning layers (only created if enabled)
        if self.rtg_cond_enabled:
            self.rtg_bin_emb = nn.Embedding(self.rtg_K, self.cfg_model.hidden_dim)
            self.rtg_emb_proj = nn.Linear(self.cfg_model.hidden_dim * self.num_reward_components, self.cfg_model.hidden_dim)
            self.action_cond_norm = nn.LayerNorm(self.cfg_model.hidden_dim, eps=1e-5)
            self.rtg_to_film = nn.Sequential(
                nn.Linear(self.cfg_model.hidden_dim, self.cfg_model.hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(self.cfg_model.hidden_dim, 2 * self.cfg_model.hidden_dim),
            )

        trajeglish = bool(getattr(self.cfg_model, "trajeglish", False))
        il = bool(getattr(self.cfg_model, "il", False))
        
        if not (trajeglish or il):
            num_types = 3
        elif trajeglish:
            num_types = 1
        else:
            num_types = 2

        self.causal_mask = get_causal_mask(self.cfg, int(self.cfg_dataset.train_context_length), num_types)

        self.apply(weight_init)
        self._reset_all_decoder_self_attn_gates()

    def _reset_all_decoder_self_attn_gates(self) -> None:
        if not self.attn_gate_enabled:
            return
        for layer in self.transformer_decoder.layers:
            sa = getattr(layer, "self_attn", None)
            if isinstance(sa, GatedMultiheadAttention):
                sa.reset_gate_parameters()

    def set_rtg_schedule(self, global_step: int, eval: bool) -> None:
        if (not self.rtg_cond_enabled) or eval:
            self._rtg_alpha = 0.0
            return
        steps = int(self.rtg_cond_anneal_steps)
        if steps <= 0:
            self._rtg_alpha = 0.0
            return
        s = min(max(int(global_step), 0), steps)
        frac = float(s) / float(steps)
        self._rtg_alpha = (1.0 - frac) * self.rtg_cond_alpha_start + frac * self.rtg_cond_alpha_end

    def _soft_rtg_embedding_from_logits(self, rtg_logits_bnkc: torch.Tensor, eval: bool) -> torch.Tensor:
        B, N, K, C = rtg_logits_bnkc.shape
        device = rtg_logits_bnkc.device
        dtype = rtg_logits_bnkc.dtype

        if eval and self.rtg_cond_use_tilt_in_eval:
            tilt = float(self.runtime_rtg_tilt)
            if abs(tilt) > 1e-6:
                tilt_vec = torch.linspace(0.0, 1.0, steps=K, device=device, dtype=dtype) * tilt
                rtg_logits_bnkc = rtg_logits_bnkc.clone()
                rtg_logits_bnkc[:, :, :, 0] = rtg_logits_bnkc[:, :, :, 0] + tilt_vec.view(1, 1, K)

        embs = []
        for c in range(C):
            p = torch.softmax(rtg_logits_bnkc[:, :, :, c].float(), dim=-1).to(dtype=dtype)
            E = self.rtg_bin_emb.weight.to(device=device, dtype=dtype)
            e = torch.matmul(p, E)
            embs.append(e)
        e_cat = torch.cat(embs, dim=-1)
        return self.rtg_emb_proj(e_cat)

    def _gt_rtg_embedding(self, data, seq_len: int) -> torch.Tensor:
        rtg_bins = data["agent"].rtgs.long()
        B, A, T, C = rtg_bins.shape

        rtg_bins_tm = rtg_bins.transpose(1, 2).reshape(B, T * A, C)
        rtg_mask = data["agent"].rtg_mask
        rtg_mask_tm = rtg_mask.transpose(1, 2).reshape(B, T * A, 1).float()

        embs = []
        for c in range(C):
            e = self.rtg_bin_emb(rtg_bins_tm[:, :, c].clamp(min=0, max=self.rtg_K - 1))
            embs.append(e)
        e_cat = torch.cat(embs, dim=-1)
        e = self.rtg_emb_proj(e_cat)
        return e * rtg_mask_tm

    def forward(self, data, scene_enc, eval: bool = False):
        agent_states = data["agent"].agent_states
        batch_size = agent_states.shape[0]
        seq_len = agent_states.shape[2]

        stacked_embeddings = scene_enc["stacked_embeddings"]
        encoder_embeddings = scene_enc["encoder_embeddings"]
        src_key_padding_mask = scene_enc["src_key_padding_mask"]

        output = self.transformer_decoder(
            stacked_embeddings,
            encoder_embeddings,
            tgt_mask=self.causal_mask.to(stacked_embeddings.device),
            memory_key_padding_mask=src_key_padding_mask,
        )

        preds = {}

        trajeglish = bool(getattr(self.cfg_model, "trajeglish", False))
        il = bool(getattr(self.cfg_model, "il", False))

        if not (trajeglish or il):
            output = output.reshape(
                batch_size,
                seq_len * int(self.cfg_dataset.max_num_agents),
                3,
                self.cfg_model.hidden_dim,
            ).permute(0, 2, 1, 3)

            state_h = output[:, 0]
            rtg_h = output[:, 1]

            if bool(getattr(self.cfg_model, "predict_rtg", True)):
                rtg_logits = self.predict_rtg(state_h)
                rtg_preds = rtg_logits.reshape(
                    batch_size, seq_len, int(self.cfg_dataset.max_num_agents),
                    self.rtg_K * self.num_reward_components,
                ).permute(0, 2, 1, 3)
                preds["rtg_preds"] = rtg_preds

            # RTG-conditioned action head
            if self.rtg_cond_enabled and bool(getattr(self.cfg_model, "predict_rtg", True)):
                rtg_logits_bnkc = rtg_logits.view(
                    batch_size, seq_len * int(self.cfg_dataset.max_num_agents),
                    self.rtg_K, self.num_reward_components
                )

                e_pred = self._soft_rtg_embedding_from_logits(rtg_logits_bnkc, eval=eval)
                if (not eval) and self.rtg_cond_stopgrad_pred:
                    e_pred = e_pred.detach()

                alpha = float(self._rtg_alpha) if (not eval) else 0.0
                if (not eval) and (alpha > 0.0):
                    e_gt = self._gt_rtg_embedding(data, seq_len=seq_len)
                    e_mix = alpha * e_gt + (1.0 - alpha) * e_pred
                else:
                    e_mix = e_pred

                film = self.rtg_to_film(e_mix)
                gamma, beta = film.chunk(2, dim=-1)
                gamma = torch.tanh(gamma)

                h = self.action_cond_norm(state_h)
                h = h * (1.0 + gamma) + beta
                action_logits = self.predict_action(h)
                action_hidden = h  # Save for refine head
            else:
                action_logits = self.predict_action(rtg_h)
                action_hidden = rtg_h  # Save for refine head

            action_preds = action_logits.reshape(
                batch_size, seq_len, int(self.cfg_dataset.max_num_agents), self.action_dim
            ).permute(0, 2, 1, 3)
            preds["action_preds"] = action_preds

            # ============================================================
            # Residual Refinement Prediction (NEW)
            # ============================================================
            if self.predict_refine is not None:
                # Predict refine delta: (dx, dy, dtheta) in body frame
                refine_logits = self.predict_refine(action_hidden)
                refine_preds = refine_logits.reshape(
                    batch_size, seq_len, int(self.cfg_dataset.max_num_agents), 3
                ).permute(0, 2, 1, 3)
                preds["refine_preds"] = refine_preds

            return preds

        if trajeglish:
            output = output.reshape(
                batch_size, seq_len * int(self.cfg_dataset.max_num_agents), 1, self.cfg_model.hidden_dim
            ).permute(0, 2, 1, 3)
            action_hidden = output[:, 0]
            action_logits = self.predict_action(action_hidden)
            action_preds = action_logits.reshape(
                batch_size, seq_len, int(self.cfg_dataset.max_num_agents), self.action_dim
            ).permute(0, 2, 1, 3)
            preds["action_preds"] = action_preds
            
            # Refine head for trajeglish
            if self.predict_refine is not None:
                refine_logits = self.predict_refine(action_hidden)
                refine_preds = refine_logits.reshape(
                    batch_size, seq_len, int(self.cfg_dataset.max_num_agents), 3
                ).permute(0, 2, 1, 3)
                preds["refine_preds"] = refine_preds
            
            return preds

        # IL mode
        output = output.reshape(
            batch_size, seq_len * int(self.cfg_dataset.max_num_agents), 2, self.cfg_model.hidden_dim
        ).permute(0, 2, 1, 3)
        action_hidden = output[:, 0]
        action_logits = self.predict_action(action_hidden)
        action_preds = action_logits.reshape(
            batch_size, seq_len, int(self.cfg_dataset.max_num_agents), self.action_dim
        ).permute(0, 2, 1, 3)
        preds["action_preds"] = action_preds
        
        # Refine head for IL
        if self.predict_refine is not None:
            refine_logits = self.predict_refine(action_hidden)
            refine_preds = refine_logits.reshape(
                batch_size, seq_len, int(self.cfg_dataset.max_num_agents), 3
            ).permute(0, 2, 1, 3)
            preds["refine_preds"] = refine_preds
        
        return preds
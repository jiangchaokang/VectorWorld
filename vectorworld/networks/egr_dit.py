"""EGR-DiT: Edge-Gated Relational Diffusion Transformer.

This is the backbone network for VectorWorld's latent generative model.
It processes heterogeneous scene graphs (lanes + agents) with factorized
attention blocks augmented by:
  - Per-edge relational bias (improves topology metrics)
  - Cross-type relational bias (improves agent-lane metrics)
  - Optional relational value gating (zero-init for stability)
  - Global Context Fusion (scene-level statistics)
  - Optional QK normalization and logit clipping

IMPORTANT: This class is instantiated as `self.model` inside LDM/FlowLDM/MeanFlowLDM,
which stores parameters under `gen_model.model.*`. Renaming this class is safe —
only the attribute path matters for checkpoint loading.
"""
import torch
import torch.nn as nn
import numpy as np

from vectorworld.networks.dit_layers import (
    FactorizedDiTBlock,
    FinalLayer,
    LabelEmbedder,
    TimestepEmbedder,
    TwoLayerResMLP,
    GlobalContextFusion,
    get_1d_sincos_pos_embed_from_grid,
)
from vectorworld.utils.pyg_helpers import get_indices_within_scene


class EGRDiT(nn.Module):
    """Edge-Gated Relational Diffusion Transformer for scene latent generation.

    Supports three time-conditioning modes:
    - Diffusion/Flow: single timestep channel t
    - MeanFlow (two_times=True): two channels (t, h=t-r) for step-size awareness
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.cfg_model = cfg.model
        self.cfg_dataset = cfg.dataset

        ldm_type = str(getattr(self.cfg_model, "ldm_type", "diffusion")).lower()
        self.ldm_type = ldm_type
        use_two = bool(getattr(self.cfg_model, "meanflow_use_two_times", False))
        self.use_two_times = (ldm_type in ("meanflow", "mf")) and use_two

        # Relational config
        self.use_rel_bias = bool(getattr(self.cfg_model, "use_rel_bias", False))
        self.use_cross_rel_bias = bool(getattr(self.cfg_model, "use_cross_rel_bias", False))
        self.use_rel_gate = bool(getattr(self.cfg_model, "use_rel_gate", False))
        self.lane_rel_dim = int(getattr(self.cfg_model, "lane_rel_dim", 64))
        self.agent_rel_dim = int(getattr(self.cfg_model, "agent_rel_dim", 32))
        self.edge_dim = int(getattr(self.cfg_model, "edge_dim", 32))
        self.qk_norm = bool(getattr(self.cfg_model, "qk_norm", False))
        self.attn_logit_clip = getattr(self.cfg_model, "attn_logit_clip", None)
        if self.attn_logit_clip is not None:
            self.attn_logit_clip = float(self.attn_logit_clip)

        # Global Context Fusion
        self.use_gcf = bool(getattr(self.cfg_model, "use_gcf", False))
        self.gcf_var_scale = float(getattr(self.cfg_model, "gcf_var_scale", 0.0))

        self.emb_drop = nn.Dropout(self.cfg_model.dropout)

        # Scene conditioning
        self.scene_type_embedder = LabelEmbedder(
            self.cfg_dataset.num_map_ids * 2,
            self.cfg_model.hidden_dim,
            self.cfg_model.label_dropout,
        )
        self.num_agents_embedder = LabelEmbedder(
            self.cfg_dataset.max_num_agents + 1, self.cfg_model.hidden_dim, 0,
        )
        self.num_lanes_embedder = LabelEmbedder(
            self.cfg_dataset.max_num_lanes + 1, self.cfg_model.hidden_dim, 0,
        )

        # Time embedding
        freq_size = 512 if self.use_two_times else 256
        self.t_embedder = TimestepEmbedder(self.cfg_model.hidden_dim, freq_size)

        # Latent embedders
        self.downsample_c = nn.Linear(self.cfg_model.hidden_dim, self.cfg_model.agent_hidden_dim)
        self.lane_embedder = TwoLayerResMLP(self.cfg_model.lane_latent_dim, self.cfg_model.hidden_dim)
        self.agent_embedder = TwoLayerResMLP(self.cfg_model.agent_latent_dim, self.cfg_model.agent_hidden_dim)

        # Positional encodings (frozen sin-cos)
        self.pos_emb_lane = nn.Parameter(
            torch.zeros(self.cfg_dataset.max_num_lanes, self.cfg_model.hidden_dim),
            requires_grad=False,
        )
        self.pos_emb_agent = nn.Parameter(
            torch.zeros(self.cfg_dataset.max_num_agents, self.cfg_model.agent_hidden_dim),
            requires_grad=False,
        )

        # Factorized DiT blocks
        self.blocks = nn.ModuleList([
            FactorizedDiTBlock(
                self.cfg_model.hidden_dim,
                self.cfg_model.agent_hidden_dim,
                self.cfg_model.num_heads,
                self.cfg_model.agent_num_heads,
                self.cfg_model.dropout,
                mlp_ratio=4,
                num_l2l_blocks=self.cfg_model.num_l2l_blocks,
                use_rel_bias=self.use_rel_bias,
                use_cross_rel_bias=self.use_cross_rel_bias,
                use_rel_gate=self.use_rel_gate,
                lane_rel_dim=self.lane_rel_dim,
                agent_rel_dim=self.agent_rel_dim,
                edge_dim=self.edge_dim,
                qk_norm=self.qk_norm,
                attn_logit_clip=self.attn_logit_clip,
            )
            for _ in range(self.cfg_model.num_factorized_dit_blocks)
        ])

        # Global Context Fusion
        if self.use_gcf:
            self.gcf = GlobalContextFusion(
                self.cfg_model.hidden_dim,
                self.cfg_model.agent_hidden_dim,
                self.cfg_model.hidden_dim,
                var_scale=self.gcf_var_scale,
            )
            self.gcf_downsample = nn.Linear(self.cfg_model.hidden_dim, self.cfg_model.agent_hidden_dim)

        # Output heads
        self.pred_agent_noise = FinalLayer(self.cfg_model.agent_hidden_dim, self.cfg_model.agent_latent_dim)
        self.pred_lane_noise = FinalLayer(self.cfg_model.hidden_dim, self.cfg_model.lane_latent_dim)

        self.initialize_weights()

    def initialize_weights(self):
        """Custom weight initialization following DiT convention."""
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Sin-cos positional embeddings
        pos_lane = get_1d_sincos_pos_embed_from_grid(
            self.pos_emb_lane.shape[-1], np.arange(self.pos_emb_lane.shape[0])
        )
        self.pos_emb_lane.data.copy_(torch.from_numpy(pos_lane).float())

        pos_agent = get_1d_sincos_pos_embed_from_grid(
            self.pos_emb_agent.shape[-1],
            self.cfg_dataset.max_num_lanes + np.arange(self.pos_emb_agent.shape[0]),
        )
        self.pos_emb_agent.data.copy_(torch.from_numpy(pos_agent).float())

        # Label embeddings
        nn.init.normal_(self.scene_type_embedder.embedding_table.weight, std=0.02)
        nn.init.normal_(self.num_agents_embedder.embedding_table.weight, std=0.02)
        nn.init.normal_(self.num_lanes_embedder.embedding_table.weight, std=0.02)

        # Timestep MLP
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Zero-init adaLN modulation and output layers
        for block in self.blocks:
            for l2l in block.l2l_blocks:
                nn.init.constant_(l2l.adaLN_modulation[-1].weight, 0)
                nn.init.constant_(l2l.adaLN_modulation[-1].bias, 0)
                if self.use_rel_bias:
                    l2l.attn._zero_init_rel_bias()

            nn.init.constant_(block.a2a_block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.a2a_block.adaLN_modulation[-1].bias, 0)
            if self.use_rel_bias:
                block.a2a_block.attn._zero_init_rel_bias()

            nn.init.constant_(block.l2a_block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.l2a_block.adaLN_modulation[-1].bias, 0)
            nn.init.constant_(block.a2l_block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.a2l_block.adaLN_modulation[-1].bias, 0)

            if self.use_cross_rel_bias:
                block.l2a_block.attn._zero_init_rel_bias()
                block.a2l_block.attn._zero_init_rel_bias()

        for layer in (self.pred_agent_noise, self.pred_lane_noise):
            nn.init.constant_(layer.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(layer.adaLN_modulation[-1].bias, 0)
            nn.init.constant_(layer.linear.weight, 0)
            nn.init.constant_(layer.linear.bias, 0)

        if self.use_gcf:
            self.gcf._zero_init()

    def forward(
        self, x_agent, x_lane, data,
        agent_timestep, lane_timestep,
        unconditional=False,
        agent_h=None, lane_h=None,
        scene_drop_mask=None,
    ):
        """Forward pass through EGR-DiT.

        Args:
            x_agent: (Na, 1, Da) noised agent latents
            x_lane: (Nl, 1, Dl) noised lane latents
            agent_timestep, lane_timestep: per-node timesteps
            unconditional: force CFG unconditional branch
            agent_h, lane_h: MeanFlow second time channel (h = t - r)
            scene_drop_mask: (B,) fixed label-drop mask for CFG
        """
        device = x_agent.device

        lane_idx = get_indices_within_scene(data["lane"].batch)
        agent_idx = get_indices_within_scene(data["agent"].batch)

        # Embed latents + positional encoding
        x_lane = self.lane_embedder(x_lane[:, 0]) + self.pos_emb_lane[lane_idx]
        x_agent = self.agent_embedder(x_agent[:, 0]) + self.pos_emb_agent[agent_idx]

        # Scene type conditioning (with CFG label dropout)
        scene_idx = self.cfg_dataset.num_map_ids * data["lg_type"].long() + data["map_id"].long()

        force_drop_ids = None
        if unconditional:
            force_drop_ids = torch.ones_like(scene_idx, dtype=torch.long)
        elif scene_drop_mask is not None:
            force_drop_ids = scene_drop_mask.to(device=scene_idx.device).long()

        scene_type = self.scene_type_embedder(scene_idx.long(), train=self.training, force_drop_ids=force_drop_ids)

        agent_batch = data["agent"].batch
        lane_batch = data["lane"].batch

        # Count embeddings
        num_agents_emb = self.num_agents_embedder(data["num_agents"].long(), train=self.training)[agent_batch]
        num_lanes_emb = self.num_lanes_embedder(data["num_lanes"].long(), train=self.training)[lane_batch]

        # Time embedding
        t_cat = torch.cat([lane_timestep, agent_timestep], dim=0).view(-1)

        if self.use_two_times and agent_h is not None and lane_h is not None:
            h_cat = torch.cat([lane_h, agent_h], dim=0).view(-1)
            t_freq = TimestepEmbedder.timestep_embedding(t_cat, dim=256).to(device)
            h_freq = TimestepEmbedder.timestep_embedding(h_cat, dim=256).to(device)
            time_emb = self.t_embedder.mlp(torch.cat([t_freq, h_freq], dim=-1))
        else:
            time_emb = self.t_embedder(t_cat)

        # Conditioning vector
        n = torch.cat([num_lanes_emb, num_agents_emb], dim=0)
        y = torch.cat([scene_type[lane_batch], scene_type[agent_batch]], dim=0)
        c = time_emb + y + n
        c_small = self.downsample_c(c)

        # Edge indices
        l2l_ei = data["lane", "to", "lane"].edge_index
        a2a_ei = data["agent", "to", "agent"].edge_index
        l2a_ei = data["lane", "to", "agent"].edge_index.clone()
        l2a_ei[1] = l2a_ei[1] + x_lane.shape[0]

        x_lane = self.emb_drop(x_lane)
        x_agent = self.emb_drop(x_agent)

        # Process through factorized blocks
        for block in self.blocks:
            if self.use_gcf:
                ctx = self.gcf(x_lane, x_agent, lane_batch, agent_batch)
                ctx_lane = ctx[lane_batch]
                ctx_agent = ctx[agent_batch]
                c_gcf = c + torch.cat([ctx_lane, ctx_agent], dim=0)
                c_small_gcf = c_small + torch.cat([
                    self.gcf_downsample(ctx_lane),
                    self.gcf_downsample(ctx_agent),
                ], dim=0)
                x_lane, x_agent = block(x_lane, x_agent, c_gcf, c_small_gcf, l2l_ei, a2a_ei, l2a_ei)
            else:
                x_lane, x_agent = block(x_lane, x_agent, c, c_small, l2l_ei, a2a_ei, l2a_ei)

        # Predict noise/velocity
        c_lane = c[:x_lane.shape[0]]
        c_agent = c_small[x_lane.shape[0]:]
        x_lane = self.pred_lane_noise(x_lane, c_lane).unsqueeze(1)
        x_agent = self.pred_agent_noise(x_agent, c_agent).unsqueeze(1)

        return x_agent, x_lane
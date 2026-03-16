import torch
import torch.nn as nn
import numpy as np

from typing import Tuple, Union

import torch.nn.functional as F
from vectorworld.utils.layers import (
    ResidualMLP,
    AttentionLayer,
    AutoEncoderFactorizedAttentionBlock,
)
from vectorworld.utils.train_helpers import weight_init
from vectorworld.utils.losses import GeometricLosses
from vectorworld.utils.data_container import (
    get_batches,
    get_features,              # for downstream / convert_batch_to_scenarios
    get_features_with_motion,  # for AE encoder/decoder with motion code
    get_edge_indices,
    get_encoder_edge_indices,
)
from vectorworld.utils.data_helpers import reparameterize
from configs.config import NON_PARTITIONED


class ScenarioDreamerEncoder(nn.Module):
    """Encoder of the Scenario Dreamer AutoEncoder."""

    def __init__(self, cfg):
        super(ScenarioDreamerEncoder, self).__init__()
        self.cfg = cfg
        self.motion_dim = int(getattr(self.cfg, "motion_dim", 0))

        # ------------------------------------------------------------------ #
        # Learnable query token used for lane-conditional distribution head  #
        # ------------------------------------------------------------------ #
        self.Q = nn.Parameter(torch.Tensor(1, self.cfg.hidden_dim), requires_grad=True)
        nn.init.xavier_uniform_(self.Q)
        # Lane-conditional distribution head
        self.pred_lane_cond_dis = ResidualMLP(
            input_dim=self.cfg.hidden_dim,
            hidden_dim=self.cfg.hidden_dim,
            n_hidden=2,
            output_dim=self.cfg.max_num_lanes + 1,
        )
        # Fuse lane information into the query token
        self.l2q_transformer_layer = AttentionLayer(
            hidden_dim=self.cfg.hidden_dim,
            num_heads=self.cfg.num_heads,
            head_dim=self.cfg.hidden_dim // self.cfg.num_heads,
            feedforward_dim=self.cfg.dim_f,
            dropout=self.cfg.dropout,
            bipartite=True,
            has_pos_emb=False,
            pos_emb_hidden_dim=None,
        )

        # ------------------------------------------------------------------ #
        # Agent embedding: Static / positionMLP+ gating self-adaptation#
        # ------------------------------------------------------------------ #
        # Enter layout: x_agent = [7D state, movement_dim,type_onehot(3)]
        static_dim = self.cfg.state_dim
        motion_dim = self.motion_dim
        type_dim = self.cfg.num_agent_types

        self.agent_static_input_dim = static_dim + type_dim
        self.agent_motion_input_dim = motion_dim + type_dim if motion_dim > 0 else type_dim
        self.agent_full_input_dim = static_dim + motion_dim + type_dim

        # Static branch: focus on the current 7D status + type
        self.agent_static_mlp = ResidualMLP(
            input_dim=self.agent_static_input_dim,
            hidden_dim=self.cfg.agent_hidden_dim,
        )
        # Movement Branch: View only Polyline + Type
        self.agent_motion_mlp = ResidualMLP(
            input_dim=self.agent_motion_input_dim,
            hidden_dim=self.cfg.agent_hidden_dim,
        )
        # Gating: Output of Gate per dimension (0,1) according to [static, movement, type]
        self.agent_gate_mlp = ResidualMLP(
            input_dim=self.agent_full_input_dim,
            hidden_dim=self.cfg.agent_hidden_dim,
            n_hidden=1,
            output_dim=self.cfg.agent_hidden_dim,
        )

        # Lane / lane-connection embeddings
        self.lane_mlp = ResidualMLP(
            input_dim=self.cfg.num_points_per_lane * self.cfg.lane_attr
            + self.cfg.num_lane_types,
            hidden_dim=self.cfg.hidden_dim,
        )
        self.lane_conn_mlp = ResidualMLP(
            input_dim=self.cfg.lane_conn_attr,
            hidden_dim=self.cfg.lane_conn_hidden_dim,
        )

        # Factorised attention encoder blocks
        self.encoder_transformer_blocks = []
        for _ in range(self.cfg.num_encoder_blocks):
            encoder_transformer_block = AutoEncoderFactorizedAttentionBlock(
                lane_hidden_dim=self.cfg.hidden_dim,
                lane_feedforward_dim=self.cfg.dim_f,
                lane_num_heads=self.cfg.num_heads,
                agent_hidden_dim=self.cfg.agent_hidden_dim,
                agent_feedforward_dim=self.cfg.agent_dim_f,
                agent_num_heads=self.cfg.agent_num_heads,
                lane_conn_hidden_dim=self.cfg.lane_conn_hidden_dim,
                dropout=self.cfg.dropout,
            )
            self.encoder_transformer_blocks.append(encoder_transformer_block)
        self.encoder_transformer_blocks = nn.ModuleList(
            self.encoder_transformer_blocks
        )

        # Gaussian latent variable heads
        self.agent_mu = nn.Linear(self.cfg.agent_hidden_dim, self.cfg.agent_latent_dim)
        self.lane_mu = nn.Linear(self.cfg.hidden_dim, self.cfg.lane_latent_dim)
        self.agent_log_var = nn.Linear(
            self.cfg.agent_hidden_dim, self.cfg.agent_latent_dim
        )
        self.lane_log_var = nn.Linear(self.cfg.hidden_dim, self.cfg.lane_latent_dim)

        self.apply(weight_init)

    def forward(
        self,
        x_agent: torch.Tensor,
        x_lane: torch.Tensor,
        x_lane_conn: torch.Tensor,
        a2a_edge_index: torch.Tensor,
        l2l_edge_index: torch.Tensor,
        l2a_edge_index: torch.Tensor,
        l2q_edge_index: torch.Tensor,
        agent_batch: torch.Tensor,
        return_lane_embeddings: bool = False,
    ) -> Union[
        Tuple[
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
        ],
        np.ndarray,
    ]:
        """Encode a batch of vectorized scenes (agents + lanes + lane connectivity)."""

        assert x_agent.dtype == torch.float32
        assert x_lane.dtype == torch.float32
        assert x_lane_conn.dtype == torch.float32

        # -----------agents + gating -------------------
        static_dim = self.cfg.state_dim
        motion_dim = self.motion_dim
        type_dim = self.cfg.num_agent_types

        x_states = x_agent[:, :static_dim]
        x_motion = (
            x_agent[:, static_dim : static_dim + motion_dim] if motion_dim > 0 else None
        )
        x_types = x_agent[:, static_dim + motion_dim : static_dim + motion_dim + type_dim]

        static_input = torch.cat([x_states, x_types], dim=-1)
        h_static = self.agent_static_mlp(static_input)

        if motion_dim > 0:
            motion_input = torch.cat([x_motion, x_types], dim=-1)
            h_motion = self.agent_motion_mlp(motion_input)
        else:
            h_motion = torch.zeros_like(h_static)

        gate_logits = self.agent_gate_mlp(x_agent)  # (Na, agent_hidden_dim)
        g = torch.sigmoid(gate_logits)
        agent_embeddings = (1.0 - g) * h_static + g * h_motion

        # ------------------------ lane / lane-conn ------------------------
        lane_embeddings = self.lane_mlp(x_lane)
        lane_conn_embeddings = self.lane_conn_mlp(x_lane_conn)

        batch_size = int((agent_batch.max() + 1).item()) if agent_batch.numel() > 0 else 0
        if batch_size <= 0:
            # edge case: no agents (shouldn't happen in your datasets)
            batch_size = 1
        query_embeddings = self.Q.repeat(batch_size, 1)

        for _ in range(self.cfg.num_encoder_blocks):
            (
                agent_embeddings,
                lane_embeddings,
                lane_conn_embeddings,
            ) = self.encoder_transformer_blocks[_](
                agent_embeddings,
                lane_embeddings,
                lane_conn_embeddings,
                lane_conn_embeddings,
                a2a_edge_index,
                l2l_edge_index,
                l2a_edge_index,
            )
            lane_query_embeddings = torch.cat([lane_embeddings, query_embeddings], dim=0)
            lane_query_embeddings = self.l2q_transformer_layer(
                lane_query_embeddings, None, l2q_edge_index
            )
            query_embeddings = lane_query_embeddings[lane_embeddings.shape[0] :]

        # ------------------------ FD metric uses this ------------------------
        if return_lane_embeddings:
            return lane_embeddings  # torch.Tensor on current device

        lane_cond_dis_logits = self.pred_lane_cond_dis(query_embeddings)
        lane_cond_dis_prob = F.softmax(lane_cond_dis_logits, -1)
        agent_mu = self.agent_mu(agent_embeddings)
        lane_mu = self.lane_mu(lane_embeddings)
        agent_log_var = self.agent_log_var(agent_embeddings)
        lane_log_var = self.lane_log_var(lane_embeddings)

        return (
            agent_mu,
            lane_mu,
            agent_log_var,
            lane_log_var,
            lane_cond_dis_logits,
            lane_cond_dis_prob,
        )


class ScenarioDreamerDecoder(nn.Module):
    """Decoder of the Scenario Dreamer AutoEncoder."""

    def __init__(self, cfg):
        super(ScenarioDreamerDecoder, self).__init__()
        self.cfg = cfg
        self.motion_dim = int(getattr(self.cfg, "motion_dim", 0))

        # ------------------- linear projections from latent space -------- #
        self.lane_mlp = nn.Linear(self.cfg.lane_latent_dim, self.cfg.hidden_dim)
        self.agent_mlp = nn.Linear(self.cfg.agent_latent_dim, self.cfg.agent_hidden_dim)
        self.downsample_lane_mlp = nn.Linear(
            self.cfg.hidden_dim, self.cfg.lane_conn_hidden_dim
        )
        self.lane_conn_mlp = nn.Linear(
            self.cfg.lane_conn_hidden_dim * 2, self.cfg.lane_conn_hidden_dim
        )

        # ------------------- factorized attention decoder blocks ---------------------- #
        self.decoder_transformer_blocks = []
        for _ in range(self.cfg.num_decoder_blocks):
            decoder_transformer_block = AutoEncoderFactorizedAttentionBlock(
                lane_hidden_dim=self.cfg.hidden_dim,
                lane_feedforward_dim=self.cfg.dim_f,
                lane_num_heads=self.cfg.num_heads,
                agent_hidden_dim=self.cfg.agent_hidden_dim,
                agent_feedforward_dim=self.cfg.agent_dim_f,
                agent_num_heads=self.cfg.agent_num_heads,
                lane_conn_hidden_dim=self.cfg.lane_conn_hidden_dim,
                dropout=self.cfg.dropout,
            )
            self.decoder_transformer_blocks.append(decoder_transformer_block)
        self.decoder_transformer_blocks = nn.ModuleList(
            self.decoder_transformer_blocks
        )

        # ------------------- output heads -------------------------------- #
        # Static 7D Head
        self.pred_agent_static = ResidualMLP(
            input_dim=self.cfg.agent_hidden_dim,
            hidden_dim=self.cfg.agent_hidden_dim,
            n_hidden=3,
            output_dim=self.cfg.state_dim,
        )
        # position head (K tableline points)
        if self.motion_dim > 0:
            self.pred_agent_motion = ResidualMLP(
                input_dim=self.cfg.agent_hidden_dim,
                hidden_dim=self.cfg.agent_hidden_dim,
                n_hidden=3,
                output_dim=self.motion_dim,
            )
        else:
            self.pred_agent_motion = None

        self.pred_agent_types = ResidualMLP(
            input_dim=self.cfg.agent_hidden_dim,
            hidden_dim=self.cfg.agent_hidden_dim,
            n_hidden=2,
            output_dim=self.cfg.num_agent_types,
        )
        if self.cfg.num_lane_types > 0:
            self.pred_lane_types = ResidualMLP(
                input_dim=self.cfg.hidden_dim,
                hidden_dim=self.cfg.hidden_dim,
                n_hidden=2,
                output_dim=self.cfg.num_lane_types,
            )

        self.pred_lane_states = ResidualMLP(
            input_dim=self.cfg.hidden_dim,
            hidden_dim=self.cfg.hidden_dim,
            n_hidden=3,
            output_dim=self.cfg.num_points_per_lane * self.cfg.lane_attr,
        )
        self.pred_lane_conn = ResidualMLP(
            input_dim=self.cfg.lane_conn_hidden_dim,
            hidden_dim=self.cfg.lane_conn_hidden_dim,
            n_hidden=2,
            output_dim=self.cfg.lane_conn_attr,
        )
        self.apply(weight_init)

    def forward(
        self,
        x_agent: torch.Tensor,
        x_lane: torch.Tensor,
        a2a_edge_index: torch.Tensor,
        l2l_edge_index: torch.Tensor,
        l2a_edge_index: torch.Tensor,
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """Decode latent embeddings into vectorized driving scenes."""

        agent_embeddings = self.agent_mlp(x_agent)
        lane_embeddings = self.lane_mlp(x_lane)

        lane_embeddings_downsampled = self.downsample_lane_mlp(lane_embeddings)
        src_lane_conn_embedding = lane_embeddings_downsampled[l2l_edge_index[0]]
        dst_lane_conn_embedding = lane_embeddings_downsampled[l2l_edge_index[1]]
        lane_conn_embeddings = self.lane_conn_mlp(
            torch.cat([src_lane_conn_embedding, dst_lane_conn_embedding], dim=-1)
        )

        for _ in range(self.cfg.num_decoder_blocks):
            (
                agent_embeddings,
                lane_embeddings,
                lane_conn_embeddings,
            ) = self.decoder_transformer_blocks[_](
                agent_embeddings,
                lane_embeddings,
                lane_conn_embeddings,
                lane_conn_embeddings,
                a2a_edge_index,
                l2l_edge_index,
                l2a_edge_index,
            )

        # Static + position
        agent_static_pred = self.pred_agent_static(agent_embeddings)  # (Na,7)
        if self.motion_dim > 0:
            agent_motion_pred = self.pred_agent_motion(agent_embeddings)  # (Na, motion_dim)
            agent_states_pred_full = torch.cat(
                [agent_static_pred, agent_motion_pred], dim=-1
            )
        else:
            agent_motion_pred = None
            agent_states_pred_full = agent_static_pred

        lane_states_pred = self.pred_lane_states(lane_embeddings).reshape(
            x_lane.shape[0], self.cfg.num_points_per_lane, self.cfg.lane_attr
        )

        agent_types_logits = self.pred_agent_types(agent_embeddings)
        agent_types_pred = torch.argmax(agent_types_logits, dim=1)

        if self.cfg.num_lane_types > 0:
            lane_types_logits = self.pred_lane_types(lane_embeddings)
            lane_types_pred = torch.argmax(lane_types_logits, dim=1)
        else:
            lane_types_logits = None
            lane_types_pred = None

        lane_conn_logits = self.pred_lane_conn(lane_conn_embeddings)
        lane_conn_pred = torch.argmax(lane_conn_logits, dim=1)
        lane_conn_pred = F.one_hot(
            lane_conn_pred, num_classes=self.cfg.lane_conn_attr
        )

        return (
            agent_states_pred_full,
            agent_types_logits,
            agent_types_pred,
            lane_states_pred,
            lane_types_logits,
            lane_types_pred,
            lane_conn_logits,
            lane_conn_pred,
        )


class AutoEncoder(nn.Module):
    """Scenario Dreamer AutoEncoder."""

    def __init__(self, cfg):
        super(AutoEncoder, self).__init__()
        self.cfg = cfg
        self.motion_dim = int(getattr(self.cfg, "motion_dim", 0))
        self.encoder = ScenarioDreamerEncoder(self.cfg)
        self.decoder = ScenarioDreamerDecoder(self.cfg)

        # loss functions for training variational autoencoder
        self.agent_loss_fn = GeometricLosses["l1"]()
        # lane loss over point dimensions
        self.lane_loss_fn = GeometricLosses["l1"]((1, 2))
        self.agent_type_loss_fn = GeometricLosses["cross_entropy"](apply_mean=False)
        self.lane_type_loss_fn = GeometricLosses["cross_entropy"](apply_mean=False)
        self.lane_conn_loss_fn = GeometricLosses["cross_entropy"](apply_mean=False)
        self.kl_loss_fn = GeometricLosses["kl"]()

        # Motion related loss weight
        self.lambda_motion = float(getattr(self.cfg, "motion_loss_weight"))
        self.lambda_smooth = float(getattr(self.cfg, "motion_smooth_weight"))
        self.static_xy_weight = float(getattr(self.cfg, "static_xy_weight"))
        self.static_other_weight = float(getattr(self.cfg, "static_other_weight"))
        # col: Soft bumps are now only available on **the current frame 7D**, instead of history
        self.lambda_collision = float(
            getattr(self.cfg, "motion_collision_weight", 0.0)
        )
        self.motion_num_points = int(
            getattr(self.cfg, "motion_num_points", max(self.motion_dim // 2, 0))
        )
        self.motion_x_range = float(getattr(self.cfg, "motion_x_range", 12.0))
        self.motion_y_range = float(
            getattr(self.cfg, "motion_y_range", self.motion_x_range / 2.0)
        )

        # Geostationary Modem Associated Super-Specific
        self.motion_static_weight = float(
            getattr(self.cfg, "motion_static_weight", 3.0)
        )
        self.motion_static_eps = float(
            getattr(self.cfg, "motion_static_eps", 0.03)
        )

        # To use a state 7D from [m]-1Inversely normalized to physical coordinates (collapse regulars)
        self.fov = float(getattr(self.cfg, "fov", 64.0))
        self.min_length = float(getattr(self.cfg, "min_length", 0.0))
        self.max_length = float(getattr(self.cfg, "max_length", 30.0))
        self.min_width = float(getattr(self.cfg, "min_width", 0.0))
        self.max_width = float(getattr(self.cfg, "max_width", 10.0))

        # Continue weight of the plane succ end
        self.lane_endpoint_weight = float(
            getattr(self.cfg, "lane_endpoint_weight", 0.0)
        )
        # Late noise enhancement std
        self.latent_noise_std = float(
            getattr(self.cfg, "latent_noise_std", 0.0)
        )
        # Softly avoid buffer (m)
        self.collision_margin = float(
            getattr(self.cfg, "collision_margin", 1.0)
        )

        self.apply(weight_init)

    def loss(self, data):
        agent_batch, lane_batch, lane_conn_batch = get_batches(data)
        # x_agent_states_full = [7D static, motion_dim]
        (
            x_agent,
            x_agent_states_full,
            x_agent_types,
            x_lane,
            x_lane_states,
            x_lane_types,
            x_lane_conn,
        ) = get_features_with_motion(data)
        a2a_edge_index, l2l_edge_index, l2a_edge_index = get_edge_indices(data)
        (
            a2a_edge_index_encoder,
            l2l_edge_index_encoder,
            l2a_edge_index_encoder,
            l2q_edge_index_encoder,
            x_lane_conn_encoder,
        ) = get_encoder_edge_indices(data)

        # --------------------------------------------
        # scene-wiseAggregation
        # --------------------------------------------
        def _scene_mean(values: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
            """Compute mean over scenes given node-wise values and batch indices."""
            if values.numel() == 0:
                return values.new_tensor(0.0)
            if batch.numel() == 0:
                return values.mean()
            B = int(batch.max().item()) + 1
            per_scene = []
            for b in range(B):
                mask = batch == b
                if mask.any():
                    per_scene.append(values[mask].mean())
            if len(per_scene) == 0:
                return values.mean()
            return torch.stack(per_scene).mean()

        (
            agent_mu,
            lane_mu,
            agent_log_var,
            lane_log_var,
            lane_cond_dis_logits,
            lane_cond_dis_prob,
        ) = self.encoder(
            x_agent,
            x_lane,
            x_lane_conn_encoder,
            a2a_edge_index_encoder,
            l2l_edge_index_encoder,
            l2a_edge_index_encoder,
            l2q_edge_index_encoder,
            agent_batch,
        )

        # ------------------------------------------------------------------ #
        # VAEreparameterize + latent noise enhancement (training only)#
        # ------------------------------------------------------------------ #
        agent_latents = reparameterize(agent_mu, agent_log_var)
        lane_latents = reparameterize(lane_mu, lane_log_var)

        if self.latent_noise_std > 0.0 and self.training:
            noise_agent = torch.randn_like(agent_latents) * self.latent_noise_std
            noise_lane = torch.randn_like(lane_latents) * self.latent_noise_std
            agent_latents = agent_latents + noise_agent
            lane_latents = lane_latents + noise_lane

        (
            agent_states_pred_full,
            agent_types_logits,
            agent_types_pred,
            lane_states_pred,
            lane_types_logits,
            lane_types_pred,
            lane_conn_logits,
            lane_conn_pred,
        ) = self.decoder(
            agent_latents, lane_latents, a2a_edge_index, l2l_edge_index, l2a_edge_index
        )

        lg_type = data["lg_type"]  # partitioned (1) or non-partitioned (0)
        lane_cond_dis = data["num_lanes_after_origin"]  # gt num lanes after partition

        # ---------------- conditional lane distribution predictor ----------------
        ce_loss = nn.CrossEntropyLoss(reduction="none")
        lane_cond_dis_loss = ce_loss(lane_cond_dis_logits, lane_cond_dis)
        partition_mask = lg_type == NON_PARTITIONED
        assert torch.all(lane_cond_dis[partition_mask] == 0)
        lane_cond_dis_loss[partition_mask] = 0

        # For catch that does not have any partitioned scene, avoid asking on empty tansor means creating nan
        valid_mask = ~partition_mask
        if valid_mask.any():
            lane_cond_dis_loss_val = lane_cond_dis_loss[valid_mask].mean().detach()
            lane_cond_dis_pred_filtered = torch.argmax(
                lane_cond_dis_prob[valid_mask], dim=-1
            )
            lane_cond_dis_acc = (
                (torch.abs(lane_cond_dis_pred_filtered - lane_cond_dis[valid_mask]) <= 3)
                .float()
                .mean()
            )
        else:
            lane_cond_dis_loss_val = lane_cond_dis_loss.new_tensor(0.0).detach()
            lane_cond_dis_acc = lane_cond_dis_loss.new_tensor(0.0)

        # ------agent L1: static & motion separate weight ---
        state_dim = self.cfg.state_dim
        if self.motion_dim > 0 and x_agent_states_full.shape[1] >= state_dim + self.motion_dim:
            gt_static = x_agent_states_full[:, :state_dim]
            gt_motion = x_agent_states_full[:, state_dim: state_dim + self.motion_dim]

            pred_static = agent_states_pred_full[:, :state_dim]
            pred_motion = agent_states_pred_full[:, state_dim: state_dim + self.motion_dim]

            # - New: static 7D internal display pair (x,y) weighted -
            static_abs = torch.abs(pred_static - gt_static)  # (Na, 7)
            # Location error for each agent (x,y)
            pos_err = static_abs[:, 0:2].mean(dim=1)              # (Na,)
            # Average error of the remaining 5 dimensions
            other_err = static_abs[:, 2:].mean(dim=1)             # (Na,)

            static_err_weighted = (
                self.static_xy_weight    * pos_err +
                self.static_other_weight * other_err
            )  # (Na,)

            agent_static_loss = _scene_mean(static_err_weighted, agent_batch)

            # - --motion L1: maintain the original logic (geostationary weighting + scene mean) -
            if self.motion_num_points > 0:
                K = self.motion_num_points
            else:
                K = max(self.motion_dim // 2, 1)
            gt_motion_pts = gt_motion.view(-1, K, 2)
            pred_motion_pts = pred_motion.view(-1, K, 2)

            diffs = gt_motion_pts[:, 1:] - gt_motion_pts[:, :-1]
            seg_len = torch.linalg.norm(diffs, dim=-1)
            path_len_norm = seg_len.sum(dim=-1)

            static_mask = path_len_norm < self.motion_static_eps
            per_agent_l1 = torch.abs(pred_motion_pts - gt_motion_pts).mean(dim=(1, 2))
            weights = torch.ones_like(per_agent_l1)
            if self.motion_static_weight != 1.0:
                weights = torch.where(static_mask,
                                    weights * self.motion_static_weight,
                                    weights)
            motion_loss_per_agent = per_agent_l1 * weights
            agent_motion_loss = _scene_mean(motion_loss_per_agent, agent_batch)

            agent_loss = agent_static_loss + self.lambda_motion * agent_motion_loss
        else:
            # Static 7D only
            gt_static = x_agent_states_full[:, :state_dim]
            pred_static = agent_states_pred_full[:, :state_dim]
            agent_static_loss = self.agent_loss_fn(pred_static, gt_static, agent_batch)
            agent_motion_loss = torch.tensor(
                0.0, device=agent_static_loss.device, dtype=agent_static_loss.dtype
            )
            agent_loss = agent_static_loss

        # ---------------- lane L1 ----------------
        lane_loss = self.lane_loss_fn(lane_states_pred, x_lane_states, lane_batch)

        # -------Lane succ endpoint continuity (geometry) --
        lane_endpoint_loss = torch.tensor(
            0.0,
            device=lane_states_pred.device,
            dtype=lane_states_pred.dtype,
        )
        if self.lane_endpoint_weight > 0.0:
            try:
                # x_lane_conn: (E, C)
                # Note:Waymo/NuPlan_Other Organiser
                #   - 'pred' channel (index=1) corresponds to the edge direction:
                #   - 'succ' channel (index=2) corresponds to the edge direction: accessor predecessor
                #
                # What we really want is for all (ij) and j is i's side of the accessor,
                # Constraint lane_i-1lane_j [0].
                #
                # The **'pred' channel** should therefore be used instead of the original 'succ' channel.
                PRED_IDX = 1
                if x_lane_conn.shape[1] > PRED_IDX:
                    pred_mask = x_lane_conn[:, PRED_IDX] > 0.5  # (E,)
                    if pred_mask.any():
                        pred_edges = l2l_edge_index[:, pred_mask]  # (2, E_pred)
                        src_idx = pred_edges[0]  # predecessor lane i
                        dst_idx = pred_edges[1]  # successor lane j

                        # End: Pred_end_i / succ_start_j (all in the unicoded system)
                        pred_end = lane_states_pred[src_idx, -1, :]   # (E_pred, 2)
                        succ_start = lane_states_pred[dst_idx, 0, :]  # (E_pred, 2)

                        lane_endpoint_loss = (pred_end - succ_start).norm(dim=-1).mean()
            except Exception:
                lane_endpoint_loss = torch.tensor(
                    0.0,
                    device=lane_states_pred.device,
                    dtype=lane_states_pred.dtype,
                )

        # - - - Type & KL - -
        if self.cfg.num_lane_types > 0:
            lane_type_loss = self.lane_type_loss_fn(
                lane_types_logits, x_lane_types, lane_batch
            )
        else:
            lane_type_loss = torch.tensor(
                0.0,
                device=agent_states_pred_full.device,
                dtype=agent_states_pred_full.dtype,
            )

        agent_type_loss = self.agent_type_loss_fn(
            agent_types_logits, x_agent_types, agent_batch
        )
        lane_conn_loss_raw = self.lane_conn_loss_fn(
            lane_conn_logits, x_lane_conn, lane_conn_batch
        )
        lane_conn_loss = lane_conn_loss_raw.mean().detach()

        agent_kl_loss = self.kl_loss_fn(agent_mu, agent_log_var, agent_batch)
        lane_kl_loss = self.kl_loss_fn(lane_mu, lane_log_var, lane_batch)
        kl_loss = agent_kl_loss + lane_kl_loss

        # --------motion smoothing the norm (on a level playing field) -----
        motion_smooth_loss = torch.tensor(
            0.0, device=agent_states_pred_full.device, dtype=agent_states_pred_full.dtype
        )
        if self.motion_dim > 0 and self.lambda_smooth > 0.0:
            try:
                pts = agent_states_pred_full[:, state_dim : state_dim + self.motion_dim]
                pts = pts.view(-1, self.motion_num_points, 2)  # (Na,K,2)
                if self.motion_num_points >= 3:
                    d2 = pts[:, 2:] - 2 * pts[:, 1:-1] + pts[:, :-2]
                    motion_smooth_loss = d2.norm(dim=-1).mean()
            except Exception:
                motion_smooth_loss = torch.tensor(
                    0.0,
                    device=agent_states_pred_full.device,
                    dtype=agent_states_pred_full.dtype,
                )

        # - - - - - Current frame 7D Soft Collation (Physics Space) - -
        motion_collision_loss = torch.tensor(
            0.0, device=agent_states_pred_full.device, dtype=agent_states_pred_full.dtype
        )
        if self.lambda_collision > 0.0:
            try:
                # Inverted state 7D to physical coordinates (x,y,length,width only)
                static_norm = agent_states_pred_full[:, :state_dim]

                # pos_x, pos_y from [-1,1] → [-fov/2, fov/2]
                x_norm = torch.clamp(static_norm[:, 0], -1.0, 1.0)
                y_norm = torch.clamp(static_norm[:, 1], -1.0, 1.0)
                x_phys = (x_norm + 1.0) * 0.5 * self.fov - self.fov / 2.0
                y_phys = (y_norm + 1.0) * 0.5 * self.fov - self.fov / 2.0

                # length / width from [-1,1] → [min,max]
                len_norm = torch.clamp(static_norm[:, 5], -1.0, 1.0)
                wid_norm = torch.clamp(static_norm[:, 6], -1.0, 1.0)
                length_phys = (len_norm + 1.0) * 0.5 * (self.max_length - self.min_length) + self.min_length
                width_phys  = (wid_norm + 1.0) * 0.5 * (self.max_width - self.min_width)   + self.min_width

                centers = torch.stack([x_phys, y_phys], dim=-1)  # (Na,2)
                # The outer circle approximates the radius of the box.
                radii = 0.5 * torch.sqrt(length_phys ** 2 + width_phys ** 2)  # (Na,)

                num_scenes = (
                    int(agent_batch.max().item()) + 1
                    if agent_batch.numel() > 0
                    else 0
                )

                total_violation = agent_states_pred_full.new_tensor(0.0)
                cnt = 0
                margin = float(self.collision_margin)

                for b in range(num_scenes):
                    mask = agent_batch == b
                    idx = torch.where(mask)[0]
                    M = idx.numel()
                    if M <= 1:
                        continue

                    c = centers[idx]  # (M,2)
                    r = radii[idx]    # (M,)

                    diff = c.unsqueeze(1) - c.unsqueeze(0)  # (M,M,2)
                    dist = diff.norm(dim=-1)               # (M,M)

                    # Avoid Self-pairInterference: Greater value of the diagonal line
                    dist = dist + torch.eye(M, device=dist.device) * 1e6

                    rad_sum = r.unsqueeze(1) + r.unsqueeze(0)  # (M,M)
                    # violation = relu(r_i + r_j + margin - d_ij)
                    violation = torch.relu(rad_sum + margin - dist)

                    total_violation = total_violation + violation.mean()
                    cnt += 1

                if cnt > 0:
                    motion_collision_loss = total_violation / float(cnt)
            except Exception:
                motion_collision_loss = torch.tensor(
                    0.0,
                    device=agent_states_pred_full.device,
                    dtype=agent_states_pred_full.dtype,
                )

        # - - - - Total loss assembly - - -
        loss = (
            agent_loss
            + self.cfg.lane_weight * lane_loss
            + agent_type_loss
            + lane_type_loss
            + self.cfg.lane_conn_weight * lane_conn_loss_raw
            + self.cfg.kl_weight * kl_loss
            + self.lambda_smooth * motion_smooth_loss
            + self.lambda_collision * motion_collision_loss
            + self.lane_endpoint_weight * lane_endpoint_loss
        )

        loss = loss + self.cfg.cond_dis_weight * lane_cond_dis_loss

        loss_dict = {
            "loss": loss.mean(),
            "agent_loss": agent_loss.mean().detach(),
            "agent_static_loss": agent_static_loss.mean().detach(),
            "agent_motion_loss": agent_motion_loss.detach(),
            "lane_loss": lane_loss.mean().detach(),
            "agent_type_loss": agent_type_loss.mean().detach(),
            "lane_type_loss": lane_type_loss.mean().detach(),
            "lane_conn_loss": lane_conn_loss,
            "kl_loss": kl_loss.mean().detach(),
            "lane_cond_dis_loss": lane_cond_dis_loss_val,
            "lane_cond_dis_acc": lane_cond_dis_acc,
            "motion_smooth_loss": motion_smooth_loss.detach(),
            "motion_collision_loss": motion_collision_loss.detach(),
            "lane_endpoint_loss": lane_endpoint_loss.detach(),
        }

        return loss_dict

    # ------------------------------------------------------------------ #
    # forward encoder / decoder                                          #
    # ------------------------------------------------------------------ #
    def forward_encoder(
        self, data, return_stats: bool = False, return_lane_embeddings: bool = False
    ):
        """forward pass through the encoder."""
        agent_batch, lane_batch, lane_conn_batch = get_batches(data)
        (
            x_agent,
            x_agent_states_full,
            x_agent_types,
            x_lane,
            x_lane_states,
            x_lane_types,
            x_lane_conn,
        ) = get_features_with_motion(data)
        (
            a2a_edge_index_encoder,
            l2l_edge_index_encoder,
            l2a_edge_index_encoder,
            l2q_edge_index_encoder,
            x_lane_conn_encoder,
        ) = get_encoder_edge_indices(data)

        encoder_output = self.encoder(
            x_agent,
            x_lane,
            x_lane_conn_encoder,
            a2a_edge_index_encoder,
            l2l_edge_index_encoder,
            l2a_edge_index_encoder,
            l2q_edge_index_encoder,
            agent_batch,
            return_lane_embeddings,
        )

        if return_lane_embeddings:
            return encoder_output
        else:
            (
                agent_mu,
                lane_mu,
                agent_log_var,
                lane_log_var,
                lane_cond_dis_logits,
                lane_cond_dis_prob,
            ) = encoder_output

        if return_stats:
            return agent_mu, lane_mu, agent_log_var, lane_log_var

        agent_latents = reparameterize(agent_mu, agent_log_var)
        lane_latents = reparameterize(lane_mu, lane_log_var)

        return agent_latents, lane_latents, lane_cond_dis_prob

    def forward_decoder_with_motion(self, agent_latents, lane_latents, data):
        """Decode latents into **full** agent state (7D static + motion_dim)."""
        a2a_edge_index, l2l_edge_index, l2a_edge_index = get_edge_indices(data)
        (
            agent_states_pred_full,
            agent_types_logits,
            agent_types_pred,
            lane_states_pred,
            lane_types_logits,
            lane_types_pred,
            lane_conn_logits,
            lane_conn_pred,
        ) = self.decoder(
            agent_latents, lane_latents, a2a_edge_index, l2l_edge_index, l2a_edge_index
        )

        return (
            agent_states_pred_full,
            lane_states_pred,
            agent_types_pred,
            lane_types_pred,
            lane_conn_pred,
        )

    def forward_decoder(self, agent_latents, lane_latents, data):
        """Decode latents and **only** return 7D static agent states."""
        (
            agent_states_pred_full,
            lane_states_pred,
            agent_types_pred,
            lane_types_pred,
            lane_conn_pred,
        ) = self.forward_decoder_with_motion(agent_latents, lane_latents, data)

        agent_states_pred_static = agent_states_pred_full[:, : self.cfg.state_dim]
        return (
            agent_states_pred_static,
            lane_states_pred,
            agent_types_pred,
            lane_types_pred,
            lane_conn_pred,
        )

    def forward(self, data, return_latents: bool = False, return_lane_embeddings: bool = False):
        """Convenience forward: encoder + decoder(7D static only)."""
        encoder_output = self.forward_encoder(
            data, return_stats=return_latents, return_lane_embeddings=return_lane_embeddings
        )

        if return_latents or return_lane_embeddings:
            return encoder_output
        else:
            agent_latents, lane_latents, lane_cond_dis_prob = encoder_output

        (
            agent_states_pred,
            lane_states_pred,
            agent_types_pred,
            lane_types_pred,
            lane_conn_pred,
        ) = self.forward_decoder(agent_latents, lane_latents, data)

        return (
            agent_states_pred,
            lane_states_pred,
            agent_types_pred,
            lane_types_pred,
            lane_conn_pred,
            lane_cond_dis_prob,
        )
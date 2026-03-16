import os
import glob
import pickle
import random
import copy
from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import Dataset

from vectorworld.utils.lane_graph_helpers import resample_polyline, get_compact_lane_graph
from vectorworld.utils.data_helpers import add_batch_dim, extract_raw_waymo_data
from vectorworld.utils.torch_helpers import from_numpy
from vectorworld.utils.data_container import CtRLSimData
from vectorworld.utils.geometry import apply_se2_transform, normalize_agents, normalize_angle
from vectorworld.utils.collision_helpers import batched_collision_checker
from vectorworld.utils.k_disks_helpers import (
    transform_box_corners_from_vocab,
    get_local_state_transition,
    transform_box_corners_from_local_state,
    get_global_next_state,
)


class CtRLSimDataset(Dataset):
    # agent_states: [pos_x, pos_y, vel_x, vel_y, heading, length, width, existence]
    POS_X_IDX = 0
    POS_Y_IDX = 1
    VEL_X_IDX = 2
    VEL_Y_IDX = 3
    HEAD_IDX = 4
    LEN_IDX = 5
    WID_IDX = 6
    EXIST_IDX = -1
    # In Waymo, AV is last agent
    AV_IDX = -1

    def __init__(self, cfg, split_name: str = "train"):
        super().__init__()

        self.cfg = cfg
        self.data_root = self.cfg.dataset_path
        self.split_name = split_name
        self.preprocess = bool(self.cfg.preprocess)
        self.delta_t = 1.0 / float(self.cfg.simulation_hz)

        self.preprocessed_dir = os.path.join(self.cfg.preprocess_dir, f"{self.split_name}")
        os.makedirs(self.preprocessed_dir, exist_ok=True)

        if not self.preprocess:
            self.files = glob.glob(os.path.join(self.data_root, f"{self.split_name}") + "/*.pkl")
        else:
            self.files = glob.glob(os.path.join(self.preprocessed_dir) + "/*.pkl")

        self.files = sorted(self.files)
        self.dset_len = len(self.files)

        # Shuffle files for collecting state transitions to build K-disks vocabulary
        if bool(self.cfg.collect_state_transitions):
            random.shuffle(self.files)
        else:
            with open(self.cfg.k_disks_vocab_path, "rb") as f:
                self.V = np.array(pickle.load(f)["V"])
            if self.V.ndim != 2 or self.V.shape[1] != 3:
                raise ValueError(f"Invalid k-disks vocab shape: {self.V.shape}, expected [K,3].")
            if int(self.cfg.vocab_size) != int(self.V.shape[0]):
                raise ValueError(
                    f"cfg.vocab_size={int(self.cfg.vocab_size)} mismatch vocab.K={int(self.V.shape[0])}."
                )
            print(f"Loaded K-disks vocabulary from {self.cfg.k_disks_vocab_path}, V shape: {self.V.shape}")

    def get_upsampled_and_sd_lanes(self, compact_lane_graph):
        """Upsample lane polylines to high resolution for precise offroad checks,
        then downsample to fixed number of points for model input.
        """
        upsampled_lanes = []
        sd_lanes = []
        for lane_id in compact_lane_graph["lanes"]:
            lane = compact_lane_graph["lanes"][lane_id]
            upsampled_lane = resample_polyline(lane, num_points=int(self.cfg.upsample_lane_num_points))
            sd_lane = resample_polyline(upsampled_lane, num_points=int(self.cfg.num_points_per_lane))
            upsampled_lanes.append(upsampled_lane)
            sd_lanes.append(sd_lane)
        return np.asarray(upsampled_lanes), np.asarray(sd_lanes)

    def remove_offroad_agents(self, agent_states, agent_types, lanes):
        """Remove vehicles that are offroad based on distance to nearest lane.

        Waymo: AV is last agent, keep ego always.
        """
        non_ego_agent_states = agent_states[:-1]
        non_ego_agent_types = agent_types[:-1]

        agent_road_dist = np.linalg.norm(
            non_ego_agent_states[:, :1, :2] - lanes.reshape(-1, 2)[np.newaxis, :, :],
            axis=-1,
        ).min(1)

        offroad_mask = agent_road_dist > float(self.cfg.offroad_threshold)
        vehicle_mask = non_ego_agent_types[:, 1].astype(bool)
        offroad_vehicle_mask = offroad_mask & vehicle_mask

        onroad_agents = np.where(~offroad_vehicle_mask)[0]
        new_agent_states = np.concatenate([non_ego_agent_states[onroad_agents], agent_states[-1:]], axis=0)
        new_agent_types = np.concatenate([non_ego_agent_types[onroad_agents], agent_types[-1:]], axis=0)
        return new_agent_states, new_agent_types

    def rollout_k_disks(self, agent_states):
        """Compute discretized actions via K-disks rollout."""
        num_agents = agent_states.shape[0]
        num_steps = agent_states.shape[1] - 1

        states = np.zeros_like(agent_states)
        actions = np.zeros((num_agents, num_steps), dtype=np.int64)

        states[:, 0] = agent_states[:, 0]
        for t in range(num_steps):
            valid_timestep = np.logical_and(
                agent_states[:, t, self.EXIST_IDX],
                agent_states[:, t + 1, self.EXIST_IDX],
            )
            states[:, t, self.EXIST_IDX] = valid_timestep.astype(int)

            corner_0_x = -1.0 * states[:, t, self.LEN_IDX] / 2.0
            corner_0_y = -1.0 * states[:, t, self.WID_IDX] / 2.0
            corner_1_x = -1.0 * states[:, t, self.LEN_IDX] / 2.0
            corner_1_y = states[:, t, self.WID_IDX] / 2.0
            corner_2_x = states[:, t, self.LEN_IDX] / 2.0
            corner_2_y = states[:, t, self.WID_IDX] / 2.0
            corner_3_x = states[:, t, self.LEN_IDX] / 2.0
            corner_3_y = -1.0 * states[:, t, self.WID_IDX] / 2.0

            box_corners = np.array(
                [
                    [corner_0_x, corner_0_y],
                    [corner_1_x, corner_1_y],
                    [corner_2_x, corner_2_y],
                    [corner_3_x, corner_3_y],
                ]
            ).transpose(2, 0, 1)  # [A,4,2]

            box_corners_vocab = transform_box_corners_from_vocab(box_corners, self.V)

            current_state = states[:, t, [self.POS_X_IDX, self.POS_Y_IDX, self.HEAD_IDX]]
            gt_next_state = agent_states[:, t + 1, [self.POS_X_IDX, self.POS_Y_IDX, self.HEAD_IDX]]

            local_state_transitions = get_local_state_transition(current_state=current_state, next_state=gt_next_state)

            box_corners_local_state = transform_box_corners_from_local_state(box_corners, local_state_transitions)

            err = np.linalg.norm(box_corners_vocab - box_corners_local_state[:, None, :, :], axis=-1).mean(2)

            if bool(self.cfg.tokenize_with_nucleus_sampling):
                err_torch = torch.from_numpy(-err)
                action_probs = F.softmax(err_torch / float(self.cfg.tokenization_temperature), dim=1)
                sorted_probs, sorted_indices = torch.sort(action_probs, dim=-1, descending=True)

                cum_probs = torch.cumsum(sorted_probs, dim=-1)
                selected_actions = cum_probs < float(self.cfg.tokenization_nucleus)
                selected_actions[:, 0] = True

                next_action_dis = torch.zeros_like(err_torch)
                next_action_dis.scatter_(1, sorted_indices, selected_actions * sorted_probs)
                next_action_dis = next_action_dis / next_action_dis.sum(dim=-1, keepdim=True).clamp(min=1e-10)

                next_actions = torch.multinomial(next_action_dis, 1)[:, 0].numpy()
            else:
                next_actions = np.argmin(err, axis=1)

            next_actions[~valid_timestep] = 0
            actions[:, t] = next_actions

            next_state_pos_heading = get_global_next_state(current_state, self.V[next_actions])
            next_v = (next_state_pos_heading[:, :2] - current_state[:, :2]) / self.delta_t

            next_exists = np.zeros(num_agents, dtype=int)
            next_state = np.array(
                [
                    next_state_pos_heading[:, 0],
                    next_state_pos_heading[:, 1],
                    next_v[:, 0],
                    next_v[:, 1],
                    next_state_pos_heading[:, 2],
                    states[:, t, self.LEN_IDX],
                    states[:, t, self.WID_IDX],
                    next_exists,
                ]
            ).transpose(1, 0)

            next_state[~valid_timestep] = 0
            states[:, t + 1] = next_state

        return states, actions

    def get_ego_collision_rewards(self, agent_states_all):
        """Compute vehicle-ego collision rewards. Output shape: [A,T] int."""
        ego_state = agent_states_all[self.AV_IDX:, :, :]
        other_states = agent_states_all[: self.AV_IDX, :, :]

        veh_ego_collision_reward = batched_collision_checker(
            ego_state[:, :, [self.POS_X_IDX, self.POS_Y_IDX, self.HEAD_IDX, self.LEN_IDX, self.WID_IDX]],
            other_states[:, :, [self.POS_X_IDX, self.POS_Y_IDX, self.HEAD_IDX, self.LEN_IDX, self.WID_IDX]],
        )

        veh_ego_collision_reward = np.concatenate(
            [veh_ego_collision_reward, np.zeros((1, veh_ego_collision_reward.shape[1]), dtype=int)],
            axis=0,
        )
        veh_ego_collision_reward = veh_ego_collision_reward * agent_states_all[:, :, self.EXIST_IDX].astype(int)
        return veh_ego_collision_reward.astype(int)

    def _compute_ego_proximity_costs(self, agent_states_all: np.ndarray) -> np.ndarray:
        """Soft proximity cost to ego using circle approximation.

        cost = (margin - gap)_+^2
        gap = dist(center_i, center_ego) - (r_i + r_ego)

        Returns: [A,T] float32, ego row = 0.
        """
        margin = float(self.cfg.rtg.reward.proximity_margin)
        min_dist_clip = float(self.cfg.rtg.reward.proximity_min_dist_clip)

        ego = agent_states_all[self.AV_IDX]  # [T,F]
        others = agent_states_all[: self.AV_IDX]  # [A-1,T,F]

        # radii
        r_ego = 0.5 * np.sqrt(ego[:, self.LEN_IDX] ** 2 + ego[:, self.WID_IDX] ** 2)
        r_other = 0.5 * np.sqrt(others[:, :, self.LEN_IDX] ** 2 + others[:, :, self.WID_IDX] ** 2)

        dx = others[:, :, self.POS_X_IDX] - ego[None, :, self.POS_X_IDX]
        dy = others[:, :, self.POS_Y_IDX] - ego[None, :, self.POS_Y_IDX]
        dist = np.sqrt(dx * dx + dy * dy + 1e-6)
        dist = np.maximum(dist, min_dist_clip)

        gap = dist - (r_other + r_ego[None, :])
        violation = np.maximum(0.0, margin - gap)
        cost = violation ** 2  # [A-1,T]

        prox = np.concatenate([cost, np.zeros((1, cost.shape[1]), dtype=cost.dtype)], axis=0)
        prox = prox * agent_states_all[:, :, self.EXIST_IDX].astype(prox.dtype)
        return prox.astype(np.float32)

    def _compute_rewards(
        self,
        veh_ego_collision_rewards: np.ndarray,  # [A,T] int
        veh_ego_proximity_costs: np.ndarray,    # [A,T] float
        agent_states_all: np.ndarray,           # [A,T,F]
    ) -> np.ndarray:
        """Return per-agent per-timestep reward array [A,T] (float32)."""
        mode = str(self.cfg.rtg.reward.mode)
        w_col = float(self.cfg.rtg.reward.collision_weight)
        w_prox = float(self.cfg.rtg.reward.proximity_weight)

        if mode == "ego_collision":
            r = -w_col * veh_ego_collision_rewards.astype(np.float32)
        elif mode == "ego_collision_proximity":
            r = -w_col * veh_ego_collision_rewards.astype(np.float32) - w_prox * veh_ego_proximity_costs.astype(np.float32)
        else:
            raise ValueError(f"Unknown rtg.reward.mode='{mode}'")

        r = r * agent_states_all[:, :, self.EXIST_IDX].astype(np.float32)
        return r.astype(np.float32)

    def get_last_valid_positions(self, states):
        """Last valid positions for all agents (use last exist==1, not first)."""
        num_agents = len(states)
        last_valid_positions = []
        for a in range(num_agents):
            valid = np.where(states[a, :, self.EXIST_IDX] > 0)[0]
            if len(valid) == 0:
                last_valid_positions.append(states[a, 0, :2])
            else:
                last_valid_positions.append(states[a, valid[-1], :2])
        return np.array(last_valid_positions)

    def get_agent_mask(self, agent_states, normalize_dict) -> np.ndarray:
        """Mask agents within FOV. Returns [A,T] bool."""
        fov = float(self.cfg.fov)

        agent_states = normalize_agents(agent_states, normalize_dict)
        agent_states = agent_states[:, :, [self.POS_X_IDX, self.POS_Y_IDX, self.HEAD_IDX]]
        agent_mask = np.logical_and(
            np.logical_and(agent_states[:, :, self.POS_X_IDX] < fov / 2.0, agent_states[:, :, self.POS_Y_IDX] < fov / 2.0),
            np.logical_and(agent_states[:, :, self.POS_X_IDX] > -fov / 2.0, agent_states[:, :, self.POS_Y_IDX] > -fov / 2.0),
        )
        return agent_mask

    def compute_rtgs(self, rewards: np.ndarray) -> np.ndarray:
        """Compute discounted RTG clipped to horizon, normalize to [0,1]."""
        H = int(self.cfg.rtg.horizon_steps)
        gamma = float(self.cfg.rtg.discount)
        clip_min = float(self.cfg.rtg.clip_min)
        clip_max = float(self.cfg.rtg.clip_max)

        if H <= 0:
            raise ValueError(f"rtg.horizon_steps must be > 0, got {H}")
        if not (clip_max > clip_min):
            raise ValueError(f"rtg.clip_max must be > rtg.clip_min, got {clip_min}, {clip_max}")
        if not (0.0 <= gamma <= 1.0):
            raise ValueError(f"rtg.discount must be in [0,1], got {gamma}")

        A, T = rewards.shape
        rtg = np.zeros((A, T), dtype=np.float32)

        weights = (gamma ** np.arange(H)).astype(np.float32)  # [H]

        if T >= H:
            win = np.lib.stride_tricks.sliding_window_view(rewards, window_shape=H, axis=1)  # [A, T-H+1, H]
            rtg[:, : (T - H + 1)] = (win * weights[None, None, :]).sum(-1).astype(np.float32)

            # tail
            for t in range(T - H + 1, T):
                k = T - t
                rtg[:, t] = (rewards[:, t:] * weights[:k][None, :]).sum(-1).astype(np.float32)
        else:
            # horizon longer than sequence: use partial sums
            for t in range(T):
                k = T - t
                rtg[:, t] = (rewards[:, t:] * weights[:k][None, :]).sum(-1).astype(np.float32)

        rtg = np.clip(rtg, a_min=clip_min, a_max=clip_max)
        rtg = (rtg - clip_min) / (clip_max - clip_min)
        return rtg.astype(np.float32)

    def discretize_rtgs(self, rtgs_norm01: np.ndarray) -> np.ndarray:
        """Discretize normalized RTG in [0,1] into integer bins."""
        K = int(self.cfg.rtg.discretization)
        if K <= 1:
            raise ValueError(f"rtg.discretization must be > 1, got {K}")
        bins = np.round(rtgs_norm01 * (K - 1)).astype(np.int64)
        bins = np.clip(bins, 0, K - 1).astype(np.int64)
        return bins

    def select_closest_max_num_agents(
        self,
        agent_states,
        agent_types,
        agent_mask,
        actions,
        rtgs,
        rtg_mask,
        moving_agent_mask,
        origin_agent_idx,
        timestep,
        active_agents=None,
    ):
        """Select closest <= max_num_agents to origin at given timestep.

        IMPORTANT FIX:
        - preserve distance order in both train & simulation paths
        - do NOT use np.intersect1d for order-sensitive selection
        """
        origin_states = agent_states[origin_agent_idx, timestep, :2].reshape(1, -1)
        dist_to_origin = np.linalg.norm(origin_states - agent_states[:, timestep, :2], axis=-1)

        if active_agents is None:
            exists_during_buffer = agent_mask.sum(-1) > 0
            valid_agents = np.where(exists_during_buffer)[0]
        else:
            valid_agents = np.concatenate([np.array([0], dtype=int), active_agents], axis=0)

        sorted_by_dist = np.argsort(dist_to_origin)
        sorted_valid = sorted_by_dist[np.isin(sorted_by_dist, valid_agents)]
        closest_ag_ids = sorted_valid[: int(self.cfg.max_num_agents)].astype(int)

        if not np.isin(origin_agent_idx, closest_ag_ids):
            raise RuntimeError("origin_agent_idx must be included in closest_ag_ids but it is missing.")

        if self.split_name == "train":
            np.random.shuffle(closest_ag_ids)

        final_agent_states = np.zeros((int(self.cfg.max_num_agents), *agent_states[0].shape))
        final_agent_types = -np.ones((int(self.cfg.max_num_agents), *agent_types[0].shape))
        final_agent_mask = np.zeros((int(self.cfg.max_num_agents), *agent_mask[0].shape))
        final_actions = np.zeros((int(self.cfg.max_num_agents), *actions[0].shape))
        final_rtgs = np.zeros((int(self.cfg.max_num_agents), *rtgs[0].shape))
        final_rtg_mask = np.zeros((int(self.cfg.max_num_agents), *rtg_mask[0].shape))
        final_moving_agent_mask = np.zeros(int(self.cfg.max_num_agents))

        n = len(closest_ag_ids)
        final_agent_states[:n] = agent_states[closest_ag_ids]
        final_agent_types[:n] = agent_types[closest_ag_ids]
        final_agent_mask[:n] = agent_mask[closest_ag_ids]
        final_actions[:n] = actions[closest_ag_ids]
        final_rtgs[:n] = rtgs[closest_ag_ids]
        final_rtg_mask[:n] = rtg_mask[closest_ag_ids]
        final_moving_agent_mask[:n] = moving_agent_mask[closest_ag_ids]

        new_origin_agent_idx = int(np.where(closest_ag_ids == origin_agent_idx)[0][0])
        return (
            final_agent_states,
            final_agent_types,
            final_agent_mask,
            final_actions,
            final_rtgs,
            final_rtg_mask,
            final_moving_agent_mask,
            new_origin_agent_idx,
            closest_ag_ids,
        )

    def get_normalized_lanes_in_fov(self, lanes, normalize_dict):
        """Normalize lanes and return lanes within lane_fov up to max_num_lanes."""
        yaw = normalize_dict["yaw"]
        translation = normalize_dict["center"]

        angle_of_rotation = (np.pi / 2.0) + np.sign(-yaw) * np.abs(yaw)
        translation = translation[np.newaxis, np.newaxis, :]

        lanes = apply_se2_transform(coordinates=lanes, translation=translation, yaw=angle_of_rotation)

        lane_point_dists_x = np.abs(lanes[:, :, 0])
        lane_point_dists_y = np.abs(lanes[:, :, 1])
        lanes_within_fov = np.logical_and(
            lane_point_dists_x < float(self.cfg.lane_fov) / 2.0,
            lane_point_dists_y < float(self.cfg.lane_fov) / 2.0,
        )
        valid_lane_mask = lanes_within_fov.sum(1) > 0
        lanes = lanes[valid_lane_mask]
        lane_mask = lanes_within_fov[valid_lane_mask]

        if len(lanes) > int(self.cfg.max_num_lanes):
            min_road_dist_to_orig = np.linalg.norm(lanes[:, :, :2], axis=-1).min(1)
            closest_roads_to_ego = np.argsort(min_road_dist_to_orig)[: int(self.cfg.max_num_lanes)]
            final_lanes = lanes[closest_roads_to_ego]
            final_lane_mask = lane_mask[closest_roads_to_ego]
        else:
            final_lanes = np.zeros((int(self.cfg.max_num_lanes), *lanes.shape[1:]))
            final_lanes[: len(lanes)] = lanes
            final_lane_mask = np.zeros((int(self.cfg.max_num_lanes), *lane_mask.shape[1:]), dtype=bool)
            final_lane_mask[: len(lanes)] = lane_mask

        return final_lanes, final_lane_mask

    def collect_state_transitions(self, data):
        """Collect state transitions for k-disks vocabulary generation."""
        agent_data = data["objects"]
        agent_states_all, _ = extract_raw_waymo_data(agent_data)

        existence_mask = agent_states_all[:, :, self.EXIST_IDX] == 1
        valid_agent_timesteps = np.logical_and(existence_mask[:, :-1], existence_mask[:, 1:])

        agent_states_all = agent_states_all[:, :, [self.POS_X_IDX, self.POS_Y_IDX, self.HEAD_IDX]]
        diff_pos_all = agent_states_all[:, 1:, :2] - agent_states_all[:, :-1, :2]
        diff_head_all = normalize_angle(agent_states_all[:, 1:, 2:] - agent_states_all[:, :-1, 2:])

        diff_pos_all_reshaped = diff_pos_all.reshape(-1, 2)
        rotations_reshaped = -1.0 * agent_states_all[:, :-1, 2].reshape(-1)
        cos_theta = np.cos(rotations_reshaped)
        sin_theta = np.sin(rotations_reshaped)
        rotation_matrices = np.array([[cos_theta, -sin_theta], [sin_theta, cos_theta]])
        rotation_matrices = np.transpose(rotation_matrices, (2, 0, 1))

        rotated_diff_pos_all_reshaped = np.einsum("ijk,ik->ij", rotation_matrices, diff_pos_all_reshaped)
        diff_head_all_reshaped = diff_head_all.reshape(-1, 1)

        state_transitions = np.concatenate([rotated_diff_pos_all_reshaped, diff_head_all_reshaped], axis=-1)
        valid_agent_timesteps = valid_agent_timesteps.reshape(-1)
        state_transitions = state_transitions[valid_agent_timesteps]
        return state_transitions

    def get_data(self, data, idx):
        """Load preprocessed data or preprocess raw data."""
        if self.preprocess:
            num_agents = data["num_agents"]
            lanes = data["lanes"]
            states = data["states"]
            actions = data["actions"]
            agent_types = data["agent_types"]
            last_valid_positions = data["last_valid_positions"]
            veh_ego_collision_reward = data["veh_ego_collision_rewards"]
            veh_ego_proximity_costs = data["veh_ego_proximity_costs"]
        else:
            agent_data = data["objects"]
            states, agent_types = extract_raw_waymo_data(agent_data)

            exists_first_timestep = np.logical_and(states[:, 0, self.EXIST_IDX] == 1, states[:, 1, self.EXIST_IDX] == 1)
            if exists_first_timestep[self.AV_IDX] != 1:
                raise RuntimeError("Ego must exist at first two timesteps.")
            states = states[exists_first_timestep]
            agent_types = agent_types[exists_first_timestep]

            # handle missing data by setting existence to 0 from first missing timestep onward
            num_agents = states.shape[0]
            for a in range(num_agents):
                missing_indices = np.where(states[a, :, self.EXIST_IDX] == 0)[0]
                if len(missing_indices) > 0:
                    first_missing = missing_indices[0]
                    states[a, first_missing:, self.EXIST_IDX] = 0

            compact_lane_graph = get_compact_lane_graph(copy.deepcopy(data))
            lanes_upsampled, lanes = self.get_upsampled_and_sd_lanes(compact_lane_graph)

            states, agent_types = self.remove_offroad_agents(states, agent_types, lanes_upsampled)

            states, actions = self.rollout_k_disks(copy.deepcopy(states))
            num_agents = len(states)

            veh_ego_collision_reward = self.get_ego_collision_rewards(states)
            veh_ego_proximity_costs = self._compute_ego_proximity_costs(states)
            last_valid_positions = self.get_last_valid_positions(states)

            to_pickle = {
                "idx": idx,
                "num_agents": num_agents,
                "lanes": lanes,
                "states": states,
                "actions": actions,
                "agent_types": agent_types,
                "veh_ego_collision_rewards": veh_ego_collision_reward,
                "veh_ego_proximity_costs": veh_ego_proximity_costs,
                "last_valid_positions": last_valid_positions,
            }

            raw_file_name = os.path.splitext(os.path.basename(self.files[idx]))[0]
            with open(os.path.join(self.preprocessed_dir, f"{raw_file_name}.pkl"), "wb") as f:
                pickle.dump(to_pickle, f, protocol=pickle.HIGHEST_PROTOCOL)
            return None, False

        # moving agents
        moving_agents = np.where(
            np.linalg.norm(states[:, 0, :2] - last_valid_positions[:, :2], axis=1) >= float(self.cfg.moving_threshold)
        )[0]

        origin_idx = num_agents - 1
        valid_timesteps = np.where(states[origin_idx, :, self.EXIST_IDX] == 1)[0]
        if len(valid_timesteps) == 0:
            raise RuntimeError("No valid ego timesteps.")

        last_idx_in_ctx = int(self.cfg.train_context_length - 1)
        max_timestep = max(int(np.max(valid_timesteps)) - last_idx_in_ctx, 0)
        start_timestep = random.randint(0, max_timestep)

        if bool(self.cfg.normalize_to_random_timestep):
            normalize_timestep = np.random.randint(
                start_timestep,
                min(start_timestep + last_idx_in_ctx, int(np.max(valid_timesteps))),
            )
            relative_normalize_timestep = normalize_timestep - start_timestep
        else:
            normalize_timestep = min(start_timestep + last_idx_in_ctx, int(np.max(valid_timesteps)))
            relative_normalize_timestep = normalize_timestep - start_timestep

        normalize_dict = {
            "center": states[origin_idx, normalize_timestep, :2].copy(),
            "yaw": states[origin_idx, normalize_timestep, self.HEAD_IDX].copy(),
        }

        timesteps = np.arange(int(self.cfg.train_context_length))
        agent_mask = self.get_agent_mask(copy.deepcopy(states[:, :, : self.HEAD_IDX + 1]), normalize_dict)

        rewards = self._compute_rewards(veh_ego_collision_reward, veh_ego_proximity_costs, states)
        rtgs = self.compute_rtgs(rewards)
        rtgs = self.discretize_rtgs(rtgs)

        rtg_mask = np.ones(rtgs.shape, dtype=bool)

        timestep_buffer = np.repeat(
            timesteps[np.newaxis, :, np.newaxis],
            int(self.cfg.max_num_agents),
            0,
        )
        state_buffer = states[:, start_timestep : start_timestep + int(self.cfg.train_context_length)]
        agent_type_buffer = agent_types
        agent_mask_buffer = agent_mask[:, start_timestep : start_timestep + int(self.cfg.train_context_length)]
        action_buffer = actions[:, start_timestep : start_timestep + int(self.cfg.train_context_length)]
        rtg_buffer = rtgs[:, start_timestep : start_timestep + int(self.cfg.train_context_length)]
        rtg_mask_buffer = rtg_mask[:, start_timestep : start_timestep + int(self.cfg.train_context_length)]
        moving_agent_mask = np.isin(np.arange(num_agents), moving_agents)

        (
            state_buffer,
            agent_type_buffer,
            agent_mask_buffer,
            action_buffer,
            rtg_buffer,
            rtg_mask_buffer,
            moving_agent_mask,
            new_origin_agent_idx,
            _,
        ) = self.select_closest_max_num_agents(
            state_buffer,
            agent_type_buffer,
            agent_mask_buffer,
            action_buffer,
            rtg_buffer,
            rtg_mask_buffer,
            moving_agent_mask,
            origin_agent_idx=origin_idx,
            timestep=relative_normalize_timestep,
        )

        lanes, lanes_mask = self.get_normalized_lanes_in_fov(lanes, normalize_dict)
        state_buffer = normalize_agents(state_buffer, normalize_dict)

        is_ego = np.zeros(len(state_buffer), dtype=int)
        is_ego[new_origin_agent_idx] = 1
        is_ego = np.tile(is_ego[:, None, None], (1, int(self.cfg.train_context_length), 1))

        state_buffer = np.concatenate([state_buffer[:, :, :-1], is_ego, state_buffer[:, :, -1:]], axis=-1)

        state_buffer[~agent_mask_buffer.astype(bool)] = 0
        action_buffer[~agent_mask_buffer.astype(bool)] = 0
        rtg_buffer[~agent_mask_buffer.astype(bool)] = 0
        rtg_mask_buffer[~agent_mask_buffer.astype(bool)] = 0

        lanes[~lanes_mask.astype(bool)] = 0
        lanes = np.concatenate([lanes, lanes_mask[:, :, None]], axis=-1)

        d = {
            "idx": idx,
            "agent": from_numpy(
                {
                    "agent_states": add_batch_dim(state_buffer),
                    "agent_types": add_batch_dim(agent_type_buffer),
                    "actions": add_batch_dim(action_buffer),
                    "rtgs": add_batch_dim(rtg_buffer[:, :, None]),
                    "rtg_mask": add_batch_dim(rtg_mask_buffer[:, :, None]),
                    "timesteps": add_batch_dim(timestep_buffer),
                    "moving_agent_mask": add_batch_dim(moving_agent_mask),
                }
            ),
            "map": from_numpy({"road_points": add_batch_dim(lanes)}),
        }
        return CtRLSimData(d), False

    def get(self, idx):
        if not self.preprocess:
            with open(self.files[idx], "rb") as file:
                data = pickle.load(file)
            if len(data["objects"]) == 1:
                return None
            if bool(self.cfg.collect_state_transitions):
                return self.collect_state_transitions(data)
            d, _ = self.get_data(data, idx)
            return d

        proceed = False
        while not proceed:
            raw_file_name = os.path.splitext(os.path.basename(self.files[idx]))[0]
            raw_path = os.path.join(self.preprocessed_dir, f"{raw_file_name}.pkl")
            if os.path.exists(raw_path):
                with open(raw_path, "rb") as f:
                    data = pickle.load(f)
                proceed = True
            else:
                idx += 1

            if proceed:
                d, no_roadgraph = self.get_data(data, idx)
                if no_roadgraph:
                    proceed = False
                    idx += 1
        return d

    def len(self):
        return self.dset_len
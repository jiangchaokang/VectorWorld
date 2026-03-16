import os
import sys
import glob
import hydra
import torch
import pickle
import random
import copy
from tqdm import tqdm
from typing import Any, Dict, List, Tuple, Union, Sequence

from torch_geometric.data import Dataset
torch.set_printoptions(threshold=100000)
import numpy as np
np.set_printoptions(suppress=True, threshold=sys.maxsize)
from configs.config import CONFIG_PATH, PARTITIONED

from vectorworld.utils.data_container import ScenarioDreamerData
from vectorworld.utils.lane_graph_helpers import resample_polyline, get_compact_lane_graph
from vectorworld.utils.pyg_helpers import get_edge_index_bipartite, get_edge_index_complete_graph
from vectorworld.utils.data_helpers import (
    get_object_type_onehot_waymo, 
    get_lane_connection_type_onehot_waymo, 
    modify_agent_states, 
    normalize_scene_with_motion, 
    randomize_indices_with_motion,
    extract_raw_waymo_data
)
from vectorworld.utils.torch_helpers import from_numpy
from vectorworld.utils.geometry import apply_se2_transform, rotate_and_normalize_angles, normalize_agents


class WaymoDatasetAutoEncoder(Dataset):
    """A Torch-Geometric ``Dataset`` wrapping Waymo scenes for auto-encoding.

    The dataset performs processing of the extracted
    Waymo Open Dataset pickles (obtained from a separate data extraction script), including lane-graph extraction,
    agent-state normalisation, partitioning for in-painting. If preprocess=True, loads directly from preprocessed files
    for efficient autoencoder training. If preprocess=False, saves preprocessed data to disk or COS.
    """

    def __init__(
        self,
        cfg: Any,
        split_name: str = "train",
        mode: str = "train",
        files: Sequence[str] | None = None,
    ) -> None:
        super(WaymoDatasetAutoEncoder, self).__init__()
        self.cfg = cfg
        self.data_root = self.cfg.dataset_path
        self.split_name = split_name 
        self.mode = mode
        self.preprocess = self.cfg.preprocess
        self.preprocessed_dir = os.path.join(self.cfg.preprocess_dir, f"{self.split_name}")
        if not os.path.exists(self.preprocessed_dir):
            os.makedirs(self.preprocessed_dir, exist_ok=True)

        self.is_preprocessing = not self.preprocess
        self.preprocess_seed: int = int(getattr(self.cfg, "preprocess_seed", 1))

        motion_cfg = getattr(self.cfg, "motion", None)
        if motion_cfg is not None:
            self.motion_enabled = bool(getattr(motion_cfg, "enabled", True))

            self.motion_num_points: int = int(getattr(motion_cfg, "num_points", 6))
            self.motion_history_max_m: float = float(
                getattr(motion_cfg, "history_max_m", 12.0)
            )

            if hasattr(motion_cfg, "windows_m") and not hasattr(
                motion_cfg, "history_max_m"
            ):
                ws = list(getattr(motion_cfg, "windows_m"))
                if len(ws) > 0:
                    self.motion_history_max_m = float(max(ws))

            self.motion_d_static: float = float(getattr(motion_cfg, "d_static", 0.5))
            self.motion_v_static: float = float(getattr(motion_cfg, "v_static", 0.2))
            self.motion_t_hist_max: int = int(getattr(motion_cfg, "t_hist_max", 8))

            self.motion_max_displacement: float = float(
                getattr(motion_cfg, "max_displacement", self.motion_history_max_m)
            )
            self.motion_y_max: float = float(
                getattr(motion_cfg, "y_max", self.motion_max_displacement / 2.0)
            )
            self.motion_dim: int = int(
                getattr(motion_cfg, "dim", 2 * self.motion_num_points)
            )

            self.motion_sentinel: float = float(
                getattr(motion_cfg, "sentinel", 0.0)
            )

            if self.motion_num_points <= 0 or self.motion_dim <= 0:
                self.motion_enabled = False
        else:
            self.motion_enabled = False
            self.motion_num_points = 0
            self.motion_history_max_m = 0.0
            self.motion_d_static = 0.5
            self.motion_v_static = 0.2
            self.motion_t_hist_max = 8
            self.motion_max_displacement = 0.0
            self.motion_y_max = 0.0
            self.motion_dim = 0
            self.motion_sentinel = 0.0

        self.save_to_cos = bool(getattr(self.cfg, "save_to_cos", False)) and self.is_preprocessing
        self.cos_bucket = getattr(self.cfg, "cos_bucket", "prod-dl-datasets-1311437600")
        self.cos_prefix = getattr(self.cfg, "cos_prefix", "")
        self.cos_secret_id = getattr(self.cfg, "cos_secret_id", None)
        self.cos_secret_key = getattr(self.cfg, "cos_secret_key", None)
        self.cos_client = None

        if self.save_to_cos:
            try:
                from dap.utils.cos import Cos
                from dap.config import global_config
                global_config.cos_secret_id = self.cos_secret_id
                global_config.cos_secret_key = self.cos_secret_key
                self.cos_client = Cos()
                print(f"☁️  COS enabled: bucket={self.cos_bucket}, prefix={self.cos_prefix}")
            except Exception as e:
                print(f"⚠️  COS initialization failed: {e}")
                self.save_to_cos = False
        if not self.preprocess:
            if files is not None:
                self.files = sorted(list(files))
            else:
                self.files = sorted(
                    glob.glob(os.path.join(self.data_root, f"{self.split_name}") + "/*.pkl")
                )

            if self.split_name == "test":
                self.files_augmented = copy.deepcopy(self.files)
                random.shuffle(self.files)
                self.files_augmented.extend(self.files[:10000])
                self.files = self.files_augmented
        else:
            if files is not None:
                self.files = sorted(list(files))
            else:
                self.files = sorted(glob.glob(self.preprocessed_dir + "/*.pkl"))
        self.dset_len = len(self.files)

    # ─────────────────────────────────────────────
    # COS helper
    # ─────────────────────────────────────────────
    def _save_to_cos_if_enabled(self, data: Any, cos_key: str, max_retries: int = 3) -> bool:
        """Upload a pickle-serialized object to COS if enabled.

        Returns True on success, False otherwise. No-op when COS is disabled."""
        if not self.save_to_cos or self.cos_client is None:
            return False

        import pickle as _pickle
        import os as _os
        import time as _time

        for attempt in range(max_retries):
            try:
                pickle_bytes = _pickle.dumps(data, protocol=_pickle.HIGHEST_PROTOCOL)
                full_key = f"{self.cos_prefix}/{cos_key}" if self.cos_prefix else cos_key
                filesystem_path_to_check = _os.path.join("/datasets/", full_key)
                self.cos_client.put_object(bucket=self.cos_bucket, key=full_key, body=pickle_bytes)
                print(f"  └── Uploaded to COS: cos://{self.cos_bucket}/{full_key}")
                print(f"      Local mount verification: ls -lh {filesystem_path_to_check}")
                return True
            except Exception as e:
                print(f"⚠️  COS upload failed (attempt {attempt+1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    _time.sleep(0.5)
        return False

    def _save_preprocessed_scene(self, payload: Dict[str, Any], raw_file_name: str) -> None:
        filename = f'{raw_file_name}_{payload["lg_type"]}_{payload["scene_timestep"]}.pkl'
        if self.save_to_cos:
            cos_key = f"{self.split_name}/{filename}"
            ok = self._save_to_cos_if_enabled(payload, cos_key)
            if not ok:
                local_path = os.path.join(self.preprocessed_dir, filename)
                with open(local_path, 'wb') as f:
                    pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
        else:
            local_path = os.path.join(self.preprocessed_dir, filename)
            with open(local_path, 'wb') as f:
                pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)

    # -------------------------------------------------------------------------
    # Lane-graph helpers (unchanged)
    # -------------------------------------------------------------------------
    def partition_compact_lane_graph(self, compact_lane_graph: Dict[str, Any]) -> Dict[str, Any]:
        """Split lanes that cross the scene's x-axis (``y = 0``)."""
        max_lane_id = max(list(compact_lane_graph['lanes'].keys()))
        next_lane_id = max_lane_id + 1

        lane_ids = list(compact_lane_graph['lanes'].keys())
        for lane_id in lane_ids:
            lane = compact_lane_graph['lanes'][lane_id]
            
            # Get y-values of the lane and find where it crosses or is near y = 0
            y_values = lane[:, 1]  # Assuming lane is [x, y] points
            sign_diff = np.insert(np.diff(np.signbit(y_values)), 0, 0)
            zero_crossings = np.where(sign_diff)[0]  # Indices where lane crosses y = 0
            
            if len(zero_crossings) == 0:  # If no crossings, skip this lane
                continue
            
            # Add artificial partitions at y = 0 crossings
            new_lanes = {}
            start_index = 0
            for crossing in zero_crossings:
                end_index = crossing + 1  # Create a partition from start to crossing
                new_lanes[next_lane_id] = lane[start_index:end_index]
                start_index = crossing  # Update start index for the next partition
                next_lane_id += 1
            
            # Handle the remaining part of the lane after the last crossing
            if zero_crossings[-1] < len(y_values) - 1:
                new_lanes[next_lane_id] = lane[start_index:]
                next_lane_id += 1
            
            # Update the compact_lane_graph with new lanes
            num_new_lanes = len(new_lanes)
            if num_new_lanes == 1:
                continue
            
            for j, new_lane_id in enumerate(new_lanes.keys()):
                compact_lane_graph['lanes'][new_lane_id] = new_lanes[new_lane_id]
                if j == 0:
                    compact_lane_graph['pre_pairs'][new_lane_id] = compact_lane_graph['pre_pairs'][lane_id]
                    # leveraging bijection between suc/pre
                    # replace successors of other lanes with new lane
                    for other_lane_id in compact_lane_graph['pre_pairs'][lane_id]:
                        if other_lane_id is not None:
                            compact_lane_graph['suc_pairs'][other_lane_id].remove(lane_id)
                            compact_lane_graph['suc_pairs'][other_lane_id].append(new_lane_id)
                    compact_lane_graph['suc_pairs'][new_lane_id] = [new_lane_id + 1] # by way we defined new lane ids
                
                elif j == num_new_lanes - 1:
                    compact_lane_graph['suc_pairs'][new_lane_id] = compact_lane_graph['suc_pairs'][lane_id]
                    # leveraging bijection between suc/pre
                    # replace predecessors of other lanes with new lane
                    for other_lane_id in compact_lane_graph['suc_pairs'][lane_id]:
                        if other_lane_id is not None:
                            compact_lane_graph['pre_pairs'][other_lane_id].remove(lane_id)
                            compact_lane_graph['pre_pairs'][other_lane_id].append(new_lane_id)
                    compact_lane_graph['pre_pairs'][new_lane_id] = [new_lane_id - 1] # by way we define new lane ids
                
                else:
                    compact_lane_graph['pre_pairs'][new_lane_id] = [new_lane_id - 1]
                    compact_lane_graph['suc_pairs'][new_lane_id] = [new_lane_id + 1]

                compact_lane_graph['left_pairs'][new_lane_id] = compact_lane_graph['left_pairs'][lane_id]
                compact_lane_graph['right_pairs'][new_lane_id] = compact_lane_graph['right_pairs'][lane_id]

            for other_lane_id in compact_lane_graph['right_pairs']:
                if lane_id in compact_lane_graph['right_pairs'][other_lane_id]:
                    compact_lane_graph['right_pairs'][other_lane_id].remove(lane_id)
                    for new_lane_id in new_lanes.keys():
                        compact_lane_graph['right_pairs'][other_lane_id].append(new_lane_id)

            for other_lane_id in compact_lane_graph['left_pairs']:
                if lane_id in compact_lane_graph['left_pairs'][other_lane_id]:
                    compact_lane_graph['left_pairs'][other_lane_id].remove(lane_id)
                    for new_lane_id in new_lanes.keys():
                        compact_lane_graph['left_pairs'][other_lane_id].append(new_lane_id)

            # remove old (now partitioned) lane from lane graph
            del compact_lane_graph['lanes'][lane_id]
            del compact_lane_graph['pre_pairs'][lane_id]
            del compact_lane_graph['suc_pairs'][lane_id]
            del compact_lane_graph['left_pairs'][lane_id]
            del compact_lane_graph['right_pairs'][lane_id]

        return compact_lane_graph


    def normalize_compact_lane_graph(self, lane_graph: Dict[str, Any], normalize_dict: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """Translate & rotate lanes so that the AV sits at the origin."""
        lane_ids = lane_graph['lanes'].keys()
        center = normalize_dict['center']
        angle_of_rotation = (np.pi / 2) + np.sign(-normalize_dict['yaw']) * np.abs(normalize_dict['yaw'])
        center = center[np.newaxis, np.newaxis, :]

        # normalize lanes to ego
        for lane_id in lane_ids:
            lane = lane_graph['lanes'][lane_id]
            lane = apply_se2_transform(coordinates=lane[:, np.newaxis, :],
                                       translation=center,
                                       yaw=angle_of_rotation)[:, 0]
            lane_graph['lanes'][lane_id] = lane
        
        return lane_graph


    def get_lane_graph_within_fov(self, lane_graph: Dict[str, Any]) -> Dict[str, Any]:
        """Return only those lanes that intersect the square *field-of-view*."""
        lane_ids = lane_graph['lanes'].keys()
        pre_pairs = lane_graph['pre_pairs']
        suc_pairs = lane_graph['suc_pairs']
        left_pairs = lane_graph['left_pairs']
        right_pairs = lane_graph['right_pairs']
        
        lane_ids_within_fov = []
        valid_pts = {}
        for lane_id in lane_ids:
            lane = lane_graph['lanes'][lane_id]
            points_in_fov_x = np.abs(lane[:, 0]) < (self.cfg.fov / 2)
            points_in_fov_y = np.abs(lane[:, 1]) < (self.cfg.fov / 2)
            points_in_fov = points_in_fov_x * points_in_fov_y
            
            if np.any(points_in_fov):
                lane_ids_within_fov.append(lane_id)
                valid_pts[lane_id] = points_in_fov

        lanes_within_fov = {}
        pre_pairs_within_fov = {}
        suc_pairs_within_fov = {}
        left_pairs_within_fov = {}
        right_pairs_within_fov = {}
        
        for lane_id in lane_ids_within_fov:
            if lane_id in lane_ids:
                lane = lane_graph['lanes'][lane_id][valid_pts[lane_id]]
                resampled_lane = resample_polyline(lane, num_points=self.cfg.upsample_lane_num_points)
                lanes_within_fov[lane_id] = resampled_lane
            
            if lane_id in pre_pairs:
                pre_pairs_within_fov[lane_id] = [l for l in pre_pairs[lane_id] if l in lane_ids_within_fov]
            else:
                pre_pairs_within_fov[lane_id] = []
            
            if lane_id in suc_pairs:
                suc_pairs_within_fov[lane_id] = [l for l in suc_pairs[lane_id] if l in lane_ids_within_fov]
            else:
                suc_pairs_within_fov[lane_id] = [] 

            if lane_id in left_pairs:
                left_pairs_within_fov[lane_id] = [l for l in left_pairs[lane_id] if l in lane_ids_within_fov]
            else:
                left_pairs_within_fov[lane_id] = []
            
            if lane_id in right_pairs:
                right_pairs_within_fov[lane_id] = [l for l in right_pairs[lane_id] if l in lane_ids_within_fov]
            else:
                right_pairs_within_fov[lane_id] = []
        
        lane_graph_within_fov = {
            'lanes': lanes_within_fov,
            'pre_pairs': pre_pairs_within_fov,
            'suc_pairs': suc_pairs_within_fov,
            'left_pairs': left_pairs_within_fov,
            'right_pairs': right_pairs_within_fov
        }
        
        return lane_graph_within_fov

    
    def get_road_points_adj(
        self,
        compact_lane_graph: Dict[str, Any],
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
        """Convert lane graph to adjacency matrices and resample lanes to num_points_per_lane."""
        
        resampled_lanes = []
        idx_to_id = {}
        id_to_idx = {}
        i = 0
        for lane_id in compact_lane_graph['lanes']:
            lane = compact_lane_graph['lanes'][lane_id]
            resampled_lane = resample_polyline(lane, num_points=self.cfg.num_points_per_lane)
            resampled_lanes.append(resampled_lane)
            idx_to_id[i] = lane_id
            id_to_idx[lane_id] = i
            i += 1
        
        resampled_lanes = np.array(resampled_lanes)
        num_lanes = min(len(resampled_lanes), self.cfg.max_num_lanes)
        dist_to_origin = np.linalg.norm(resampled_lanes, axis=-1).min(1)
        closest_lane_ids = np.argsort(dist_to_origin)[:num_lanes]
        resampled_lanes = resampled_lanes[closest_lane_ids]

        idx_to_new_idx = {}
        new_idx_to_idx = {}
        for i, j in enumerate(closest_lane_ids):
            idx_to_new_idx[j] = i 
            new_idx_to_idx[i] = j

        pre_road_adj = np.zeros((num_lanes, num_lanes))
        suc_road_adj = np.zeros((num_lanes, num_lanes))
        left_road_adj = np.zeros((num_lanes, num_lanes))
        right_road_adj = np.zeros((num_lanes, num_lanes))
        
        for new_idx_i in range(num_lanes):
            for id_j in compact_lane_graph['pre_pairs'][idx_to_id[new_idx_to_idx[new_idx_i]]]:
                if id_to_idx[id_j] in closest_lane_ids:
                    pre_road_adj[new_idx_i, idx_to_new_idx[id_to_idx[id_j]]] = 1 

            for id_j in compact_lane_graph['suc_pairs'][idx_to_id[new_idx_to_idx[new_idx_i]]]:
                if id_to_idx[id_j] in closest_lane_ids:
                    suc_road_adj[new_idx_i, idx_to_new_idx[id_to_idx[id_j]]] = 1

            for id_j in compact_lane_graph['left_pairs'][idx_to_id[new_idx_to_idx[new_idx_i]]]:
                if id_to_idx[id_j] in closest_lane_ids:
                    left_road_adj[new_idx_i, idx_to_new_idx[id_to_idx[id_j]]] = 1

            for id_j in compact_lane_graph['right_pairs'][idx_to_id[new_idx_to_idx[new_idx_i]]]:
                if id_to_idx[id_j] in closest_lane_ids:
                    right_road_adj[new_idx_i, idx_to_new_idx[id_to_idx[id_j]]] = 1
        
        return resampled_lanes, pre_road_adj, suc_road_adj, left_road_adj, right_road_adj, num_lanes

    def compute_motion_code(
        self,
        agent_states_all: np.ndarray,
        agent_states_all_ego: np.ndarray,
        global_agent_indices: np.ndarray,
        scene_timestep: int,
        num_points: int,
        history_max_m: float,
        d_static: float,
        v_static: float,
        t_hist_max: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        K = int(num_points)
        assert K >= 2, "num_points (polyline sample count) must be >= 2"

        num_agents = len(global_agent_indices)
        motion_raw = np.zeros((num_agents, 2 * K), dtype=np.float32)
        is_static = np.zeros(num_agents, dtype=bool)

        if K == 0 or history_max_m <= 0.0:
            is_static[:] = True
            return motion_raw, is_static

        for i, ag_idx in enumerate(global_agent_indices):
            states_global = agent_states_all[ag_idx]
            states_ego = agent_states_all_ego[ag_idx]
            exists = states_global[:, -1].astype(bool)

            hist_indices = np.where(exists[: scene_timestep + 1])[0]
            if len(hist_indices) == 0:
                is_static[i] = True
                motion_raw[i] = 0.0
                continue

            if not exists[scene_timestep]:
                is_static[i] = True
                motion_raw[i] = 0.0
                continue

            pos_ego = states_ego[:, :2]
            vel_ego = states_ego[:, 2:4]
            pos_current_ego = pos_ego[scene_timestep]

            recent_hist = hist_indices[-t_hist_max:]
            pos_recent = pos_ego[recent_hist]
            disp_recent = np.linalg.norm(
                pos_recent - pos_current_ego[None, :], axis=-1
            )
            max_disp = float(disp_recent.max()) if len(disp_recent) > 0 else 0.0

            vel_recent = vel_ego[recent_hist]
            speed_recent = np.linalg.norm(vel_recent, axis=-1)
            mean_speed = float(speed_recent.mean()) if len(speed_recent) > 0 else 0.0

            if (max_disp < d_static) or (mean_speed < v_static):
                is_static[i] = True
                motion_raw[i] = 0.0
                continue

            if len(hist_indices) < 2:
                is_static[i] = True
                motion_raw[i] = 0.0
                continue
            states_hist = states_global[hist_indices]          # (H, 8)
            pos_hist_global = states_hist[:, :2]

            p0_global = states_global[scene_timestep, :2].copy()
            yaw0_global = states_global[scene_timestep, 4].copy()
            coords = pos_hist_global[:, np.newaxis, :]         # (H,1,2)
            center = p0_global[np.newaxis, np.newaxis, :]
            pos_hist_body = apply_se2_transform(
                coordinates=coords, translation=center, yaw=-yaw0_global
            )[:, 0]                                            # (H,2)
            if hist_indices[-1] == scene_timestep:
                pos_hist_body[-1] = np.array([0.0, 0.0], dtype=np.float32)
            step_dists = np.linalg.norm(pos_hist_body[1:] - pos_hist_body[:-1], axis=1)
            if step_dists.size == 0 or float(step_dists.sum()) < 1e-3:
                is_static[i] = True
                motion_raw[i] = 0.0
                continue

            cum = np.concatenate([[0.0], np.cumsum(step_dists)])  # (H,)
            total_L = float(cum[-1])

            L = min(total_L, history_max_m)
            if total_L <= history_max_m:
                start_idx = 0
            else:
                target_start = total_L - history_max_m
                start_idx = int(np.searchsorted(cum, target_start, side="left"))

            pos_trunc = pos_hist_body[start_idx:]
            cum_trunc = cum[start_idx:] - cum[start_idx]
            L_trunc = float(cum_trunc[-1])

            if L_trunc < 1e-3:
                is_static[i] = True
                motion_raw[i] = 0.0
                continue

            if K == 1:
                s_samples = np.array([L_trunc], dtype=np.float32)
            else:
                s_samples = np.linspace(0.0, L_trunc, num=K, dtype=np.float32)

            x_interp = np.interp(s_samples, cum_trunc, pos_trunc[:, 0])
            y_interp = np.interp(s_samples, cum_trunc, pos_trunc[:, 1])
            points_body = np.stack([x_interp, y_interp], axis=-1).astype(np.float32)  # (K,2)

            points_body[-1] = np.array([0.0, 0.0], dtype=np.float32)

            motion_raw[i, : 2 * K] = points_body.reshape(-1)
            is_static[i] = False

        return motion_raw.astype(np.float32), is_static.astype(bool)

    # -------------------------------------------------------------------------
    # Agent helpers (modified for index tracking)
    # -------------------------------------------------------------------------
    def get_agents_within_fov(
        self,
        agent_states: np.ndarray,
        agent_types: np.ndarray,
        normalize_dict: Dict[str, np.ndarray],
        return_indices: bool = False,
    ) -> Union[
        Tuple[np.ndarray, np.ndarray],
        Tuple[np.ndarray, np.ndarray, np.ndarray]
    ]:
        """Translate agent states into the AV frame and retain only those in view.

        Parameters
        ----------
        agent_states : (N, D)
            Raw agent states at the selected timestep.
        agent_types : (N, 5)
            One-hot type encodings.
        normalize_dict : Dict[str, np.ndarray]
            Keys: {"center", "yaw"}.
        return_indices : bool, default False
            If True, also return indices into the *input* `agent_states`
            array of the agents that were kept.

        Returns
        -------
        agent_states_fov : (M, D)
        agent_types_fov : (M, 5)
        [closest_ag_ids] : (M,)
            Indices into the original `agent_states` (only if return_indices=True).
        """

        center = normalize_dict['center']
        angle_of_rotation = (np.pi / 2) + np.sign(-normalize_dict['yaw']) * np.abs(normalize_dict['yaw'])
        center = center[np.newaxis, np.newaxis, :]

        agent_states[:, :2] = apply_se2_transform(coordinates=agent_states[:, np.newaxis, :2],
                                    translation=center,
                                    yaw=angle_of_rotation)[:, 0]
        agent_states[:, 2:4] = apply_se2_transform(coordinates=agent_states[:, np.newaxis, 2:4],
                                    translation=np.zeros_like(center),
                                    yaw=angle_of_rotation)[:, 0]
        agent_states[:, 4] = rotate_and_normalize_angles(agent_states[:, 4], angle_of_rotation)

        agents_in_fov_x = np.abs(agent_states[:, 0]) < (self.cfg.fov / 2)
        agents_in_fov_y = np.abs(agent_states[:, 1]) < (self.cfg.fov / 2)
        agents_in_fov_mask = agents_in_fov_x * agents_in_fov_y
        valid_agents = np.where(agents_in_fov_mask > 0)[0]
        
        dist_to_origin = np.linalg.norm(agent_states[:, :2], axis=-1)
        closest_ag_ids = np.argsort(dist_to_origin)[:self.cfg.max_num_agents]
        closest_ag_ids = closest_ag_ids[np.in1d(closest_ag_ids, valid_agents)]

        if return_indices:
            return agent_states[closest_ag_ids], agent_types[closest_ag_ids], closest_ag_ids
        else:
            return agent_states[closest_ag_ids], agent_types[closest_ag_ids]

    
    def remove_offroad_agents(
        self,
        agent_states: np.ndarray,
        agent_types: np.ndarray,
        lane_dict: Dict[int, np.ndarray],
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Drop *vehicle* agents whose centres lie off the centerline map.

        Returns
        -------
        filtered_states : (M, D)
        filtered_types : (M, 5)
        kept_indices : (M,)
            Indices into the original `agent_states` array that are kept.
        """
        
        # keep the ego vehicle always
        non_ego_agent_states = agent_states[1:]
        non_ego_agent_types = agent_types[1:]
        
        road_pts = []
        for lane_id in lane_dict:
            road_pts.append(lane_dict[lane_id])
        road_pts = np.concatenate(road_pts, axis=0)

        agent_road_dist = np.linalg.norm(non_ego_agent_states[:, np.newaxis, :2] - road_pts[np.newaxis, :, :], axis=-1).min(1)
        offroad_mask = agent_road_dist > self.cfg.offroad_threshold
        vehicle_mask = non_ego_agent_types[:, 1].astype(bool)
        offroad_vehicle_mask = offroad_mask * vehicle_mask

        onroad_agents = np.where(~offroad_vehicle_mask)[0]

        filtered_states = np.concatenate([agent_states[:1], non_ego_agent_states[onroad_agents]], axis=0)
        filtered_types = np.concatenate([agent_types[:1], non_ego_agent_types[onroad_agents]], axis=0)

        # map kept indices back to original agent_states indexing
        kept_indices = np.concatenate([[0], 1 + onroad_agents]).astype(int)

        return filtered_states, filtered_types, kept_indices

    def _select_scene_timestep(
        self,
        valid_timesteps: np.ndarray,
        raw_file_name: str,
    ) -> int:
        assert len(valid_timesteps) > 0

        num_valid = len(valid_timesteps)
        start_idx = int(max(0, np.floor(2 * num_valid / 3)))
        candidate_timesteps = valid_timesteps[start_idx:]
        if len(candidate_timesteps) == 0:
            candidate_timesteps = valid_timesteps

        import hashlib

        base_seed = int(getattr(self, "preprocess_seed", 1))
        h = hashlib.sha1(raw_file_name.encode("utf-8")).digest()
        h_int = int.from_bytes(h[:8], "little", signed=False)
        seed = (base_seed + h_int) % (2**32)

        rng = random.Random(seed)
        idx = rng.randrange(len(candidate_timesteps))
        return int(candidate_timesteps[idx])
    
    def get_partitioned_masks(
        self,
        agents: np.ndarray,
        lanes: np.ndarray,
        a2a_edge_index: torch.Tensor,
        l2l_edge_index: torch.Tensor,
        l2a_edge_index: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, np.ndarray]:
        """Create boolean masks that *hide* edges crossing the X-axis partition."""
        a2a_edge_index = a2a_edge_index.numpy()
        l2l_edge_index = l2l_edge_index.numpy()
        l2a_edge_index = l2a_edge_index.numpy()

        agents_y = agents[:, 1]
        lanes_y = lanes[:, 9, 1]
        agents_after_origin = np.where(agents_y > 0)[0]
        lanes_after_origin = np.where(lanes_y > 0)[0]

        a2a_mask = np.isin(a2a_edge_index, agents_after_origin).sum(0) != 1
        l2l_mask = np.isin(l2l_edge_index, lanes_after_origin).sum(0) != 1

        lane_l2a_mask = np.isin(l2a_edge_index[0], lanes_after_origin)[None, :]
        agent_l2a_mask = np.isin(l2a_edge_index[1], agents_after_origin)[None, :]
        l2a_mask = np.concatenate([lane_l2a_mask, agent_l2a_mask], axis=0).sum(0) != 1   

        return torch.from_numpy(a2a_mask), torch.from_numpy(l2l_mask), torch.from_numpy(l2a_mask), lanes_y <= 0
    
    # -------------------------------------------------------------------------
    # Main get_data
    # -------------------------------------------------------------------------
    def get_data(
        self,
        data: Dict[str, Any],
        idx: int,
    ) -> Union[Dict[str, Any], ScenarioDreamerData]:
        """Process **one** Waymo scenario.

        if preprocess=True: read from cached preprocessed pickle and return ScenarioDreamerData object for autoencoder training
        if preprocess=False: cache processed data as pickle file to disk (or COS) to reduce data processing overhead during autoencoder training.
        """

        # ───────────────────────────────────────────────────────────────
        # FAST PATH: already pre-processed tensors on disk
        # ───────────────────────────────────────────────────────────────
        if self.preprocess:
            road_points = data["road_points"]
            agent_states = data["agent_states"]
            edge_index_lane_to_lane = data["edge_index_lane_to_lane"]
            edge_index_lane_to_agent = data["edge_index_lane_to_agent"]
            edge_index_agent_to_agent = data["edge_index_agent_to_agent"]
            road_connection_types = data["road_connection_types"]
            num_lanes = data["num_lanes"]
            num_agents = data["num_agents"]
            agent_types = data["agent_types"]
            lg_type = data["lg_type"]  # 0 = regular, 1 = partitioned

            motion_raw = data.get("agent_motion_raw", None)
            motion_is_static = data.get("agent_motion_is_static", None)

        # ───────────────────────────────────────────────────────────────
        # SLOW PATH: raw Waymo pickle → preprocess and cache to disk/COS
        # ───────────────────────────────────────────────────────────────
        else:
            av_index = data["av_idx"]
            agent_data = data["objects"]
            agent_states_all, agent_types_all = extract_raw_waymo_data(agent_data)

            normalize_statistics: Dict[str, Any] = {}
            raw_file_name = os.path.splitext(os.path.basename(self.files[idx]))[0]

            compact_lane_graph = get_compact_lane_graph(copy.deepcopy(data))

            valid_timesteps = np.where(agent_states_all[av_index, :, -1] == 1)[0]
            if len(valid_timesteps) == 0:
                return {"normalize_statistics": None, "valid_scene": False}

            scene_timestep = self._select_scene_timestep(valid_timesteps, raw_file_name)
            if not agent_states_all[av_index, scene_timestep, -1]:
                return {"normalize_statistics": None, "valid_scene": False}

            # Normalisation dict (ego pose at t0)
            normalize_dict = {
                "center": agent_states_all[av_index, scene_timestep, :2].copy(),
                "yaw": agent_states_all[av_index, scene_timestep, 4].copy(),
            }

            # Full history in ego frame for motion code computation / static detection
            agent_states_all_ego = normalize_agents(agent_states_all, normalize_dict)

            # Lane graph → ego frame → FOV crop
            compact_lane_graph_scene = self.normalize_compact_lane_graph(
                copy.deepcopy(compact_lane_graph), normalize_dict
            )
            compact_lane_graph_scene = self.get_lane_graph_within_fov(
                compact_lane_graph_scene
            )
            if len(compact_lane_graph_scene["lanes"]) == 0:
                return {"normalize_statistics": None, "valid_scene": False}

            # Partitioned variant for in‑painting
            compact_lane_graph_inpainting = self.partition_compact_lane_graph(
                copy.deepcopy(compact_lane_graph_scene)
            )

            # Filter agents: existence mask + class filter + FOV crop
            exists_mask = copy.deepcopy(
                agent_states_all[:, scene_timestep, -1]
            ).astype(bool)
            if self.cfg.generate_only_vehicles:
                agent_mask = copy.deepcopy(agent_types_all[:, 1]).astype(bool)
            else:
                agent_mask = copy.deepcopy(
                    agent_types_all[:, 1]
                    + agent_types_all[:, 2]
                    + agent_types_all[:, 3]
                ).astype(bool)

            exists_mask = exists_mask * agent_mask
            candidate_indices = np.where(exists_mask)[0]

            # initial selection (before FOV)
            agent_states_candidate = copy.deepcopy(
                agent_states_all[candidate_indices, scene_timestep]
            )
            agent_types_candidate = copy.deepcopy(agent_types_all[candidate_indices])

            # FOV filter
            agent_states, agent_types, fov_ids = self.get_agents_within_fov(
                agent_states_candidate,
                agent_types_candidate,
                normalize_dict,
                return_indices=True,
            )
            global_indices_after_fov = candidate_indices[fov_ids]

            # Optional off‑road removal (vehicles only)
            if self.cfg.remove_offroad_agents:
                (
                    agent_states,
                    agent_types,
                    kept_indices,
                ) = self.remove_offroad_agents(
                    agent_states, agent_types, compact_lane_graph_scene["lanes"]
                )
                global_indices_final = global_indices_after_fov[kept_indices]
            else:
                global_indices_final = global_indices_after_fov

            # Replace (vx,vy,yaw) with (speed,cosθ,sinθ)
            agent_states = modify_agent_states(agent_states)
            num_agents = len(agent_states)

            if num_agents == 0 or len(compact_lane_graph_scene["lanes"]) == 0:
                return {"normalize_statistics": None, "valid_scene": False}

            if self.motion_enabled and self.motion_num_points > 0:
                motion_raw, motion_is_static = self.compute_motion_code(
                    agent_states_all=agent_states_all,
                    agent_states_all_ego=agent_states_all_ego,
                    global_agent_indices=global_indices_final,
                    scene_timestep=scene_timestep,
                    num_points=self.motion_num_points,
                    history_max_m=self.motion_history_max_m,
                    d_static=self.motion_d_static,
                    v_static=self.motion_v_static,
                    t_hist_max=self.motion_t_hist_max,
                )
            else:
                motion_raw = None
                motion_is_static = None

            # regular & partitioned lane graphs
            lg_dict = {
                "regular": compact_lane_graph_scene,
                "partitioned": compact_lane_graph_inpainting,
            }

            for lg_type_str in lg_dict.keys():
                lg = lg_dict[lg_type_str]
                (
                    road_points,
                    pre_road_adj,
                    suc_road_adj,
                    left_road_adj,
                    right_road_adj,
                    num_lanes,
                ) = self.get_road_points_adj(lg)

                edge_index_lane_to_lane = get_edge_index_complete_graph(num_lanes)
                edge_index_agent_to_agent = get_edge_index_complete_graph(num_agents)
                edge_index_lane_to_agent = get_edge_index_bipartite(
                    num_lanes, num_agents
                )

                road_connection_types = []
                for i_edge in range(edge_index_lane_to_lane.shape[1]):
                    pre_conn_indicator = pre_road_adj[
                        edge_index_lane_to_lane[1, i_edge],
                        edge_index_lane_to_lane[0, i_edge],
                    ]
                    suc_conn_indicator = suc_road_adj[
                        edge_index_lane_to_lane[1, i_edge],
                        edge_index_lane_to_lane[0, i_edge],
                    ]
                    left_conn_indicator = left_road_adj[
                        edge_index_lane_to_lane[1, i_edge],
                        edge_index_lane_to_lane[0, i_edge],
                    ]
                    right_conn_indicator = right_road_adj[
                        edge_index_lane_to_lane[1, i_edge],
                        edge_index_lane_to_lane[0, i_edge],
                    ]
                    if (
                        edge_index_lane_to_lane[1, i_edge]
                        == edge_index_lane_to_lane[0, i_edge]
                    ):
                        road_connection_types.append(
                            get_lane_connection_type_onehot_waymo("self")
                        )
                    elif pre_conn_indicator:
                        road_connection_types.append(
                            get_lane_connection_type_onehot_waymo("pred")
                        )
                    elif suc_conn_indicator:
                        road_connection_types.append(
                            get_lane_connection_type_onehot_waymo("succ")
                        )
                    elif left_conn_indicator:
                        road_connection_types.append(
                            get_lane_connection_type_onehot_waymo("left")
                        )
                    elif right_conn_indicator:
                        road_connection_types.append(
                            get_lane_connection_type_onehot_waymo("right")
                        )
                    else:
                        road_connection_types.append(
                            get_lane_connection_type_onehot_waymo("none")
                        )
                road_connection_types = np.array(road_connection_types)

                to_pickle: Dict[str, Any] = dict()
                to_pickle["idx"] = idx
                to_pickle["lg_type"] = 0 if lg_type_str == "regular" else 1
                to_pickle["scene_timestep"] = scene_timestep
                to_pickle["num_agents"] = num_agents
                to_pickle["num_lanes"] = num_lanes
                to_pickle["road_points"] = road_points
                to_pickle["agent_states"] = agent_states[:, :-1]
                to_pickle["agent_types"] = agent_types[:, 1:4]  # vehicle, ped, cyclist
                to_pickle["edge_index_lane_to_lane"] = edge_index_lane_to_lane
                to_pickle["edge_index_agent_to_agent"] = edge_index_agent_to_agent
                to_pickle["edge_index_lane_to_agent"] = edge_index_lane_to_agent
                to_pickle["road_connection_types"] = road_connection_types

                to_pickle["agent_motion_raw"] = motion_raw
                to_pickle["agent_motion_is_static"] = motion_is_static

                self._save_preprocessed_scene(to_pickle, raw_file_name)

                if lg_type_str == "regular":
                    normalize_statistics["max_speed"] = agent_states[:, 2].max()
                    normalize_statistics["min_length"] = agent_states[:, 5].min()
                    normalize_statistics["max_length"] = agent_states[:, 5].max()
                    normalize_statistics["min_width"] = agent_states[:, 6].min()
                    normalize_statistics["max_width"] = agent_states[:, 6].max()
                    normalize_statistics["min_lane_x"] = road_points[:, 0].min()
                    normalize_statistics["min_lane_y"] = road_points[:, 1].min()
                    normalize_statistics["max_lane_x"] = road_points[:, 0].max()
                    normalize_statistics["max_lane_y"] = road_points[:, 1].max()

            return {"normalize_statistics": normalize_statistics, "valid_scene": True}

        # ───────────────────────────────────────────────────────────────
        # fast path starts from here (preprocess=True)
        # ───────────────────────────────────────────────────────────────

        agent_states, road_points, motion_code = normalize_scene_with_motion(
            agent_states,
            road_points,
            motion_raw,
            motion_is_static,
            fov=self.cfg.fov,
            min_speed=self.cfg.min_speed,
            max_speed=self.cfg.max_speed,
            min_length=self.cfg.min_length,
            max_length=self.cfg.max_length,
            min_width=self.cfg.min_width,
            max_width=self.cfg.max_width,
            min_lane_x=self.cfg.min_lane_x,
            max_lane_x=self.cfg.max_lane_x,
            min_lane_y=self.cfg.min_lane_y,
            max_lane_y=self.cfg.max_lane_y,
            motion_max_displacement=self.motion_max_displacement,
            sentinel_value=self.motion_sentinel,
        )

        if self.mode == "train":
            (
                agent_states,
                agent_types,
                motion_code,
                road_points,
                edge_index_lane_to_lane,
            ) = randomize_indices_with_motion(
                agent_states,
                agent_types,
                motion_code,
                road_points,
                edge_index_lane_to_lane,
            )
            edge_index_lane_to_lane = torch.from_numpy(edge_index_lane_to_lane)
        else:
            edge_index_lane_to_lane = torch.from_numpy(
                np.asarray(edge_index_lane_to_lane)
            )

        if lg_type == PARTITIONED:
            (
                a2a_mask,
                l2l_mask,
                l2a_mask,
                lane_partition_mask,
            ) = self.get_partitioned_masks(
                agent_states,
                road_points,
                edge_index_agent_to_agent,
                edge_index_lane_to_lane,
                edge_index_lane_to_agent,
            )

            agents_y = agent_states[:, 1]
            lanes_y = road_points[:, 9, 1]
            num_agents_after_origin = len(np.where(agents_y > 0)[0])
            num_lanes_after_origin = len(np.where(lanes_y > 0)[0])

        else:
            a2a_mask = torch.ones(edge_index_agent_to_agent.shape[1]).bool()
            l2l_mask = torch.ones(edge_index_lane_to_lane.shape[1]).bool()
            l2a_mask = torch.ones(edge_index_lane_to_agent.shape[1]).bool()
            lane_partition_mask = np.zeros(num_lanes).astype(bool)
            num_agents_after_origin = 0
            num_lanes_after_origin = 0

        assert a2a_mask.shape[0] == edge_index_agent_to_agent.shape[1]
        assert l2l_mask.shape[0] == edge_index_lane_to_lane.shape[1]
        assert l2a_mask.shape[0] == edge_index_lane_to_agent.shape[1]
        assert lane_partition_mask.shape[0] == num_lanes

        if self.cfg.remove_left_right_connections:
            # remove left and right connections for evaluation
            road_connection_types = road_connection_types[:, [0, 1, 2, 5]]

        # Assemble final PyG heterogeneous graph
        d = ScenarioDreamerData()
        d["idx"] = idx
        d["num_lanes"] = num_lanes
        d["num_agents"] = num_agents
        d["lg_type"] = lg_type
        d["agent"].x = from_numpy(agent_states)
        d["agent"].type = from_numpy(agent_types)
        d["agent"].motion = from_numpy(motion_code) 
        d["lane"].x = from_numpy(road_points)
        d["lane"].partition_mask = from_numpy(lane_partition_mask)
        d["num_agents_after_origin"] = num_agents_after_origin
        d["num_lanes_after_origin"] = num_lanes_after_origin

        d["lane", "to", "lane"].edge_index = edge_index_lane_to_lane
        d["lane", "to", "lane"].type = torch.from_numpy(road_connection_types)
        d["agent", "to", "agent"].edge_index = edge_index_agent_to_agent
        d["lane", "to", "agent"].edge_index = edge_index_lane_to_agent
        d["lane", "to", "lane"].encoder_mask = l2l_mask
        d["lane", "to", "agent"].encoder_mask = l2a_mask
        d["agent", "to", "agent"].encoder_mask = a2a_mask

        return d

    def get(self, idx: int):
        if not self.cfg.preprocess:
            with open(self.files[idx], 'rb') as file:
                data = pickle.load(file)
            d = self.get_data(data, idx)

        else:
            raw_file_name = os.path.splitext(os.path.basename(self.files[idx]))[0]
            raw_path = os.path.join(self.preprocessed_dir, f'{raw_file_name}.pkl')
            with open(raw_path, 'rb') as f:
                data = pickle.load(f)
            d = self.get_data(data, idx)
        
        return d

    
    def len(self):
        return self.dset_len


@hydra.main(version_base=None, config_path=CONFIG_PATH, config_name="config")
def main(cfg):
    cfg.ae.dataset.preprocess = False
    dset = WaymoDatasetAutoEncoder(cfg.ae.dataset, split_name='train')
    print(len(dset))
    np.random.seed(10)
    random.seed(10)
    torch.manual_seed(10)

    for idx in tqdm(range(len(dset))):
        with open(dset.files[idx], 'rb') as file:
            data = pickle.load(file)
        d = dset.get_data(data, idx)



if __name__ == '__main__':
    main()
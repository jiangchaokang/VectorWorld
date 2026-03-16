import os
import pickle
import json
import copy
import math
import numpy as np
import torch
import torch.nn.functional as F

from omegaconf import OmegaConf

from vectorworld.data.waymo.deltasim_dataset import CtRLSimDataset
from vectorworld.utils.gpudrive_helpers import (
    get_action_value_tensor,
    get_ego_state,
    get_partner_obs,
    get_map_obs,
    from_json_Map,
    ForwardKinematics,
)
from vectorworld.simulation.sim_helpers import (
    ego_completed_route,
    ego_collided,
    ego_off_route,
    ego_progress,
    normalize_route,
)
from vectorworld.utils.geometry import normalize_agents
from vectorworld.utils.lane_graph_helpers import resample_lanes_with_mask
from vectorworld.utils.k_disks_helpers import inverse_k_disks, forward_k_disks
from vectorworld.utils.collision_helpers import compute_collision_states_one_scene
from vectorworld.utils.torch_helpers import from_numpy
from vectorworld.utils.data_container import CtRLSimData
from vectorworld.utils.data_helpers import add_batch_dim, modify_agent_states
from vectorworld.utils.viz import render_state
from vectorworld.models.delta_sim import DeltaSim
from vectorworld.models.ldm import VectorWorldLDM as ScenarioDreamerLDM
from vectorworld.utils.sim_env_helpers import (
    generate_simulation_environments_in_memory,
)
from vectorworld.utils.train_helpers import set_latent_stats
from vectorworld.utils.deltasim_ckpt_loader import load_deltasim_model_strict
from vectorworld.utils.deltasim_sampling import sample_from_logits, sample_rtg_tokens


def _get_max_rtg_value(cfg_dataset) -> int:
    """Dynamically get the maximum RTG value from config.
    
    The valid RTG range is [0, K-1] where K = rtg.discretization.
    This replaces the hardcoded MAX_RTG_VAL = 349.
    
    Args:
        cfg_dataset: The dataset config containing rtg.discretization
        
    Returns:
        Maximum valid RTG token value (K - 1)
    """
    if hasattr(cfg_dataset, 'rtg') and hasattr(cfg_dataset.rtg, 'discretization'):
        K = int(cfg_dataset.rtg.discretization)
    elif hasattr(cfg_dataset, 'rtg_discretization'):
        # Backward compatibility with legacy config
        K = int(cfg_dataset.rtg_discretization)
    else:
        # Fallback to original ctrlsim default
        K = 350
    return K - 1


class Simulator:
    """Simple simulator for testing planners with CtRL-Sim behaviour model."""

    def __init__(self, cfg):
        """Initialize simulator."""
        self.cfg = cfg
        self.mode = self.cfg.sim.mode
        self.steps = self.cfg.sim.steps
        self.dt = self.cfg.sim.dt

        if self.mode == "vectorworld":
            self.dataset_path = self.cfg.sim.dataset_path
            self.test_files = sorted(os.listdir(self.dataset_path))
            self.num_test_scenarios = len(self.test_files)

            self.ldm_model = None
            self.ldm_cfg = None
            self.env_pool = None

        elif self.mode == "vectorworld_online":
            self.dataset_path = None
            self.test_files = None
            self.num_test_scenarios = int(self.cfg.sim.num_online_scenarios)

            ldm_cfg = self.cfg.ldm
            OmegaConf.set_struct(ldm_cfg, False)
            ldm_cfg.dataset_name = self.cfg.dataset_name.name
            OmegaConf.set_struct(ldm_cfg, True)

            ae_cfg = self.cfg.ae
            OmegaConf.set_struct(ae_cfg, False)
            ae_cfg.dataset_name = self.cfg.dataset_name.name
            OmegaConf.set_struct(ae_cfg, True)

            self.ldm_cfg = set_latent_stats(ldm_cfg)
            ckpt_path = self.ldm_cfg.eval.ckpt_path
            assert ckpt_path is not None and os.path.exists(ckpt_path), f"[Simulator] Invalid ldm.eval.ckpt_path: {ckpt_path}"

            print(f"[Simulator] Loading LDM checkpoint: {ckpt_path}")
            self.ldm_model = ScenarioDreamerLDM.load_from_checkpoint(
                ckpt_path, cfg=self.ldm_cfg, cfg_ae=ae_cfg, map_location="cpu"
            ).to("cuda")
            self.ldm_model.eval()

            self.env_pool = []
            self.online_batch_size = int(self.cfg.sim.online_batch_size)

        else:
            raise ValueError(f"Unsupported sim.mode='{self.mode}'")

        # ctrl-sim dataset helper
        self.ctrl_sim_dset = CtRLSimDataset(self.cfg.deltasim.dataset, split_name="val")

        # Dynamically get max RTG value from config
        self._max_rtg_val = _get_max_rtg_value(self.cfg.deltasim.dataset)

        # Load behaviour model with smart config handling
        ckpt_path = self.cfg.sim.behaviour_model.model_path
        assert ckpt_path is not None and os.path.exists(ckpt_path), f"[Simulator] behaviour_model.model_path invalid: {ckpt_path}"

        model = self._load_ctrl_sim_model(ckpt_path)

        self.behaviour_model = CtRLSimBehaviourModel(
            mode=self.mode,
            model_path=ckpt_path,
            model=model,
            dset=self.ctrl_sim_dset,
            use_rtg=bool(self.cfg.sim.behaviour_model.use_rtg),
            predict_rtgs=bool(self.cfg.sim.behaviour_model.predict_rtgs),
            action_temperature=float(self.cfg.sim.behaviour_model.action_temperature),
            tilt=float(self.cfg.sim.behaviour_model.tilt),
            steps=self.steps,
            max_rtg_val=self._max_rtg_val,  # Pass dynamic value
        )

        amp_cfg = getattr(self.cfg.sim.behaviour_model, "amp", None)
        if amp_cfg is not None:
            self.behaviour_model.use_autocast = bool(getattr(amp_cfg, "enabled", True)) and torch.cuda.is_available()
            dtype = str(getattr(amp_cfg, "dtype", "bf16")).lower()
            if dtype == "bf16" and torch.cuda.is_bf16_supported():
                self.behaviour_model.autocast_dtype = torch.bfloat16
            elif dtype in ("fp16", "float16"):
                self.behaviour_model.autocast_dtype = torch.float16
            else:
                # fallback: Disable or fp16
                self.behaviour_model.autocast_dtype = torch.float16

        # Motion Code Initialize History
        self.behaviour_model.use_motion_history = bool(self.cfg.sim.behaviour_model.init_history_from_motion)
        self.behaviour_model.history_from_motion_num_points = int(self.cfg.sim.behaviour_model.history_from_motion_num_points)

        motion_hist_cfg = self.cfg.sim.behaviour_model.motion_history
        self.behaviour_model.motion_history_speed_mode = str(motion_hist_cfg.speed_mode)
        self.behaviour_model.motion_history_speed_scale = float(motion_hist_cfg.speed_scale)
        self.behaviour_model.motion_history_min_speed = float(motion_hist_cfg.min_speed)
        self.behaviour_model.motion_history_max_speed = float(motion_hist_cfg.max_speed)
        self.behaviour_model.motion_history_max_scale = float(motion_hist_cfg.max_scale)
        self._apply_behaviour_model_inference_overrides()

        self.action_map = get_action_value_tensor()
        self.data_dict = {}

    def _apply_behaviour_model_inference_overrides(self) -> None:
        """Apply inference-only overrides from Hydra cfg (without touching ckpt)."""
        bm_cfg = getattr(self.cfg.sim, "behaviour_model", None)
        if bm_cfg is None:
            return

        inf = getattr(bm_cfg, "inference", None)
        ctrl_m = getattr(getattr(self.cfg, "deltasim", None), "model", None)

        # ---- RTG inference mode (optional) ----
        rtg_inf = getattr(inf, "rtg_inference", None) if inf is not None else None
        if rtg_inf is not None:
            mode = getattr(rtg_inf, "mode", None)
            if mode is not None:
                self.behaviour_model.rtg_inference_mode = str(mode)

        # ---- collision avoidance (prefer sim.behaviour_model.inference; fallback ctrl_sim.model.*) ----
        ca = getattr(inf, "collision_avoidance", None) if inf is not None else None
        if ca is None and ctrl_m is not None:
            ca = getattr(ctrl_m, "collision_avoidance", None)
        if ca is not None:
            enabled = getattr(ca, "enabled", None)
            beta = getattr(ca, "beta", None)
            margin = getattr(ca, "margin", None)
            min_dist_clip = getattr(ca, "min_dist_clip", None)
            if enabled is not None:
                self.behaviour_model.collision_avoid_enabled = bool(enabled)
            if beta is not None:
                self.behaviour_model.collision_avoid_beta = float(beta)
            if margin is not None:
                self.behaviour_model.collision_avoid_margin = float(margin)
            if min_dist_clip is not None:
                self.behaviour_model.collision_avoid_min_dist_clip = float(min_dist_clip)

        # ---- residual refine (prefer sim.behaviour_model.inference; fallback ctrl_sim.model.residual_refine.*) ----
        rr = getattr(inf, "residual_refine", None) if inf is not None else None
        if rr is None and ctrl_m is not None:
            rr = getattr(ctrl_m, "residual_refine", None)
        if rr is not None:
            enabled = getattr(rr, "enabled", None)
            if enabled is None:
                enabled = getattr(rr, "use_refine_in_eval", None)
            refine_scale = getattr(rr, "refine_scale", None)

            dec = getattr(getattr(self.behaviour_model, "model", None), "decoder", None)
            if dec is not None:
                if enabled is not None:
                    dec.refine_use_in_eval = bool(enabled)
                if refine_scale is not None:
                    dec.refine_scale = float(refine_scale)

        print("[Simulator] Inference overrides applied:")
        print(f"  collision_avoidance.enabled={self.behaviour_model.collision_avoid_enabled}, "
              f"beta={self.behaviour_model.collision_avoid_beta}, margin={self.behaviour_model.collision_avoid_margin}")
        try:
            dec = self.behaviour_model.model.decoder
            print(f"  residual_refine.use_in_eval={getattr(dec,'refine_use_in_eval',None)}, "
                  f"refine_scale={getattr(dec,'refine_scale',None)}")
        except Exception:
            pass
        print(f"  rtg_inference_mode={getattr(self.behaviour_model,'rtg_inference_mode','auto')}")


    def _load_ctrl_sim_model(self, ckpt_path: str) -> DeltaSim:
        """Strictly load CtRL-Sim model from checkpoint (no silent partial load)."""
        ckpt_path = ckpt_path if os.path.isabs(ckpt_path) else os.path.join(self.cfg.project_root, ckpt_path)
        model, _, report = load_deltasim_model_strict(
            ckpt_path,
            device="cuda",
            strict=True,                 # IMPORTANT: no silent missing weights
            allow_legacy_key_map=True,   # allow known safe renames
            override_kdisks_vocab_path=self.cfg.deltasim.dataset.k_disks_vocab_path,
        )

        print("[Simulator] Loaded behaviour model:")
        print(f"  ckpt: {report.ckpt_path}")
        print(f"  legacy_renames: {report.legacy_renames}")
        return model

    def load_initial_scene(self, i):
        if self.mode == "vectorworld":
            filename = self.test_files[i]
            file_path = filename if os.path.isabs(filename) else os.path.join(self.dataset_path, filename)
            with open(file_path, "rb") as f:
                scenario_dict = pickle.load(f)
            return scenario_dict

        elif self.mode == "vectorworld_online":
            if len(self.env_pool) == 0:
                num_to_generate = max(1, min(self.online_batch_size, self.num_test_scenarios))
                max_tries = int(getattr(self.cfg.sim, "max_online_gen_tries", 3))
                last_exception = None

                for attempt in range(max_tries):
                    print(f"[Simulator] Generating {num_to_generate} online scenarios via LDM... (attempt {attempt + 1}/{max_tries})")
                    try:
                        new_envs = generate_simulation_environments_in_memory(
                            model=self.ldm_model,
                            cfg_ldm=self.ldm_cfg,
                            num_envs=num_to_generate,
                            dataset=self.cfg.dataset_name.name,
                            route_length=int(self.ldm_cfg.eval.sim_envs.route_length),
                        )
                    except Exception as e:
                        last_exception = e
                        print(f"[Simulator] LDM generation attempt {attempt + 1} raised an exception: {repr(e)}")
                        continue

                    if new_envs:
                        self.env_pool.extend(new_envs)
                        break
                    else:
                        print(f"[Simulator] LDM generation attempt {attempt + 1} produced 0 valid environments.")

                if len(self.env_pool) == 0:
                    msg = "[Simulator] Failed to generate any valid online scenarios."
                    if last_exception is not None:
                        msg += f" Last exception: {repr(last_exception)}"
                    raise RuntimeError(msg)

            return self.env_pool.pop()

        else:
            raise ValueError(f"Unsupported sim.mode='{self.mode}'")

    def _find_invalid_new_agents(
        self,
        next_states,
        newly_added_agent_mask,
        still_existing_agent_mask,
        dist_gap_s=None,
        heading_threshold=None,
        dist_threshold=None,
    ):
        # Use config values with sensible defaults
        cfg_filter = getattr(self.cfg.sim, 'new_agent_filter', None)
        if dist_gap_s is None:
            dist_gap_s = float(getattr(cfg_filter, 'dist_gap_s', 5.0)) if cfg_filter else 5.0
        if heading_threshold is None:
            heading_threshold = float(getattr(cfg_filter, 'heading_threshold', np.pi / 6)) if cfg_filter else np.pi / 6
        if dist_threshold is None:
            dist_threshold = float(getattr(cfg_filter, 'dist_threshold', 2.0)) if cfg_filter else 2.0

        normalized_next_states = normalize_agents(next_states[:, None], self.local_frame)
        lanes, lanes_mask = self.ctrl_sim_dset.get_normalized_lanes_in_fov(self.data_dict["lanes"], self.local_frame)
        lanes_resampled = resample_lanes_with_mask(lanes, lanes_mask, num_points=100)
        dist_to_lanes = np.linalg.norm(
            normalized_next_states[:, None, :, :2] - lanes_resampled[None],
            axis=-1,
        ).min(2)
        closest_lane_idxs = np.argmin(dist_to_lanes, axis=-1)

        new_agent_idxs_to_remove = []
        newly_added_agent_idxs = np.where(newly_added_agent_mask)[0]
        for new_agent_idx in newly_added_agent_idxs:
            heading = normalized_next_states[new_agent_idx, 0, 4]
            if (
                np.abs(heading - np.pi / 2) < heading_threshold
                and (normalized_next_states[new_agent_idx, 0, 1] - self.cfg.deltasim.dataset.fov) < dist_threshold
            ):
                new_agent_idxs_to_remove.append(new_agent_idx)
                continue

            closest_lane = closest_lane_idxs[new_agent_idx]
            closest_lane_mask = closest_lane_idxs == closest_lane
            agent_in_same_lane_mask = np.logical_and(closest_lane_mask, still_existing_agent_mask)
            if not agent_in_same_lane_mask.sum():
                continue

            dist_to_agent_in_same_lane = np.linalg.norm(
                normalized_next_states[new_agent_idx, :, :2] - normalized_next_states[agent_in_same_lane_mask][:, 0, :2],
                axis=-1,
            )
            closest_agent_idx = np.where(agent_in_same_lane_mask)[0][np.argmin(dist_to_agent_in_same_lane)]
            dist_gap = np.linalg.norm(normalized_next_states[closest_agent_idx, 0, 2:4]) * dist_gap_s
            dist_to_closest_agent = np.linalg.norm(
                normalized_next_states[new_agent_idx, 0, :2] - normalized_next_states[closest_agent_idx, 0, :2]
            )

            if dist_to_closest_agent < dist_gap:
                new_agent_idxs_to_remove.append(new_agent_idx)
        return new_agent_idxs_to_remove

    def step(self, action):
        self.t += 1

        old_ego_state = copy.deepcopy(self.ego_state)
        if action is not None:
            if self.cfg.sim.policy == "rl":
                action = torch.nan_to_num(action, nan=0).long().cpu()
                action = self.action_map[action].numpy()
                self.ego_state = self.gpudrive_kinematics_model.forward_kinematics(action[0])
            else:
                (next_x, next_y, next_theta, next_speed) = (action[0], action[1], action[2], action[3])
                agent_next_state = np.array(
                    [
                        next_x,
                        next_y,
                        next_speed * np.cos(next_theta),
                        next_speed * np.sin(next_theta),
                        next_theta,
                        self.ego_state[5],
                        self.ego_state[6],
                        self.ego_state[7],
                    ]
                )
                self.ego_state = agent_next_state
        else:
            self.ego_state = self.ego_trajectory[self.t]

        self.local_frame = {"center": self.ego_state[:2].copy(), "yaw": self.ego_state[4].copy()}

        inverse_ego_action = inverse_k_disks(old_ego_state, self.ego_state, self.ctrl_sim_dset.V)

        self.data_dict["ego_action"].append(inverse_ego_action)
        self.data_dict["ego_rtg"].append(np.array([self._max_rtg_val])[None, :])

        if self.mode == "waymo_log_replay":
            self.data_dict["agent_next_action"] = self.scenario_dict["actions"][:, self.t - 1]
        else:
            self.data_dict = self.behaviour_model.step(self.data_dict)

        # Check if refined continuous actions are available
        use_refined = (
            "agent_next_action_refined" in self.data_dict 
            and self.data_dict["agent_next_action_refined"] is not None
            and np.any(self.data_dict["agent_next_action_refined"] != 0)
        )

        if use_refined:
            # Use continuous refined actions
            from vectorworld.utils.k_disks_helpers import forward_k_disks_continuous
            next_states = forward_k_disks_continuous(
                states=self.data_dict["agent"][-1],
                actions_continuous=self.data_dict["agent_next_action_refined"],
                delta_t=self.dt,
                exists=self.agent_active,
            )
        else:
            # Use discrete actions (original behavior)
            next_states = forward_k_disks(
                states=self.data_dict["agent"][-1],
                actions=self.data_dict["agent_next_action"],
                vocab=self.ctrl_sim_dset.V,
                delta_t=self.dt,
                exists=self.agent_active,
            )

        # keep inactive agents at last known position
        self.last_active_agent_position[self.agent_active] = next_states[self.agent_active]
        next_states[~self.agent_active] = self.last_active_agent_position[~self.agent_active]

        # compute new FOV mask
        agent_mask = self.ctrl_sim_dset.get_agent_mask(
            copy.deepcopy(next_states[:, None, : self.ctrl_sim_dset.HEAD_IDX + 1]),
            self.local_frame,
        )[:, 0]

        newly_added_agent_mask = np.logical_and(np.logical_and(~self.agent_active, agent_mask), ~self.left_scene)
        still_existing_agent_mask = np.logical_and(self.agent_active, agent_mask)

        if newly_added_agent_mask.sum():
            new_agent_idxs_to_remove = self._find_invalid_new_agents(next_states, newly_added_agent_mask, still_existing_agent_mask)
            for agent_idx in new_agent_idxs_to_remove:
                self.left_scene[agent_idx] = True

        # mark leaving FOV as left_scene
        self.left_scene = np.logical_or(self.left_scene, (self.agent_active.astype(int) - agent_mask.astype(int)) == 1)
        self.agent_active = agent_mask * ~self.left_scene

        # IMPORTANT: sync existence with agent_active
        next_states[:, -1] = self.agent_active.astype(float)

        self.data_dict["agent_active"] = copy.deepcopy(self.agent_active)
        self.data_dict["agent"].append(next_states)
        self.data_dict["agent_action"].append(self.data_dict["agent_next_action"])
        self.data_dict["agent_rtg"].append(self.data_dict["agent_next_rtg"])

        self.data_dict["ego"].append(self.ego_state[None, :])

        terminated = False
        completed_route = ego_completed_route(self.local_frame["center"], self.scenario_dict["route"])
        collided = ego_collided(self.ego_state, self.data_dict["agent"][-1][self.agent_active])
        off_route = ego_off_route(self.local_frame["center"], self.scenario_dict["route"])

        if collided or off_route or completed_route or self.t == self.cfg.sim.steps:
            if completed_route:
                off_route = False
                collided = False
            progress = ego_progress(self.local_frame["center"], self.scenario_dict["route"])
            terminated = True
            info = {"collision": collided, "off_route": off_route, "completed": completed_route, "progress": progress}
        else:
            info = {}

        invalid_agents = self.behaviour_model.update_running_statistics(self.data_dict, self.scenario_dict, terminated)
        invalid_agent_idxs = np.where(invalid_agents)[0]
        if len(invalid_agent_idxs):
            for idx in invalid_agent_idxs:
                self.left_scene[idx] = True
                self.agent_active[idx] = False
            self.data_dict["agent_active"] = copy.deepcopy(self.agent_active)
            self.data_dict["agent"][-1][:, -1] = self.agent_active.astype(float)

        self.current_state = self._get_observation()
        self._update_viz_state()

        return self.current_state, terminated, info

    def _get_observation(self):
        if self.cfg.sim.policy == "rl":
            ego_obs = get_ego_state(self.ego_state)
            partner_obs = get_partner_obs(
                self.data_dict["agent"][-1],
                self.ego_state,
                self.agent_active,
                self.local_frame,
            )
            map_obs = get_map_obs(
                self.data_dict["lanes_compressed"].copy(),
                self.ego_state,
                self.local_frame,
            )
            full_tensor = np.concatenate([ego_obs, partner_obs, map_obs], axis=-1)
            obs = torch.from_numpy(full_tensor).to("cuda:0")
        else:
            current_agent_states = np.concatenate(
                [
                    self.data_dict["agent"][-1],
                    np.expand_dims(copy.deepcopy(self.agent_active), axis=1),
                ],
                axis=1,
            )
            ego_state = np.concatenate([self.ego_state, np.ones(1)])
            obs = np.concatenate([current_agent_states, np.expand_dims(ego_state, axis=0)])
        return obs

    def _update_viz_state(self, num_route_points=30):
        current_agent_states = self.data_dict["agent"][-1]
        current_agent_types = self.data_dict["agent_type"][0]
        agent_active_mask = self.agent_active
        current_agent_states_rel = normalize_agents(current_agent_states[:, None], normalize_dict=self.local_frame)[:, 0]

        lanes, lanes_mask = self.ctrl_sim_dset.get_normalized_lanes_in_fov(self.scenario_dict["lanes"], normalize_dict=self.local_frame)
        lanes[~lanes_mask] = 0.0

        route = normalize_route(self.scenario_dict["route"], normalize_dict=self.local_frame)
        dist_to_route = np.linalg.norm(route, axis=-1)
        route_start = np.argmin(dist_to_route)
        route = route[route_start: route_start + num_route_points]

        self.viz_state = {
            "route": route,
            "agent_states": current_agent_states_rel,
            "agent_types": current_agent_types,
            "agent_active": agent_active_mask,
            "lanes": lanes,
            "lanes_mask": lanes_mask,
        }

    def initialize_data_dict(self):
        data_dict = {}

        ego = self.ego_state[None, :]
        ego_type = np.zeros((1, 5))
        ego_type[0, 1] = 1

        agents = self.scenario_dict["agents"][:, 0].copy()
        agent_types = self.scenario_dict["agent_types"]

        agents[:, -1] = self.agent_active.astype(float)

        data_dict["agent"] = [agents]
        data_dict["agent_type"] = [agent_types]
        data_dict["agent_action"] = []
        data_dict["agent_rtg"] = []
        data_dict["agent_next_action"] = []
        data_dict["agent_next_rtg"] = []

        data_dict["ego"] = [ego]
        data_dict["ego_type"] = [ego_type]
        data_dict["ego_action"] = []
        data_dict["ego_rtg"] = []
        data_dict["ego_next_rtg"] = []

        data_dict["lanes"] = self.scenario_dict["lanes"]
        if self.cfg.sim.policy == "rl":
            data_dict["lanes_compressed"] = self.scenario_dict["lanes_compressed"]
        data_dict["agent_active"] = copy.deepcopy(self.agent_active)

        self.data_dict = data_dict

        invalid_agents = self.behaviour_model.update_running_statistics(self.data_dict, self.scenario_dict)
        invalid_agent_idxs = np.where(invalid_agents)[0]
        if len(invalid_agent_idxs):
            for idx in invalid_agent_idxs:
                self.left_scene[idx] = True
                self.agent_active[idx] = False
            self.data_dict["agent_active"] = copy.deepcopy(self.agent_active)
            self.data_dict["agent"][-1][:, -1] = self.agent_active.astype(float)

    def reset(self, i):
        self.t = 0
        self.scenario_dict = self.load_initial_scene(i)

        self.ego_trajectory = self.scenario_dict["agents"][-1]
        self.ego_state = self.ego_trajectory[0]

        self.rl_kinematics_model = ForwardKinematics(
            self.ego_state[:2],
            self.ego_state[2:4],
            self.ego_state[4],
            self.ego_state[5],
            self.ego_state[6],
        )

        if self.cfg.sim.simulate_vehicles_only:
            vehicle_mask = self.scenario_dict["agent_types"][:-1, 1] == 1
        else:
            vehicle_mask = np.ones(self.scenario_dict["agent_types"][:-1].shape[0], dtype=bool)

        self.scenario_dict["agents"] = self.scenario_dict["agents"][:-1][vehicle_mask]
        self.scenario_dict["agent_types"] = self.scenario_dict["agent_types"][:-1][vehicle_mask]

        if "agent_motion" in self.scenario_dict:
            motion_all = self.scenario_dict["agent_motion"]
            if self.cfg.sim.simulate_vehicles_only:
                motion_env = motion_all[:-1][vehicle_mask]
            else:
                motion_env = motion_all[:-1]
            self.scenario_dict["agent_motion"] = motion_env

        if self.mode == "waymo_log_replay":
            self.scenario_dict["actions"] = self.scenario_dict["actions"][:-1][vehicle_mask]

        self.behaviour_model.reset(len(self.scenario_dict["agents"]) + 1)

        if getattr(self.behaviour_model, "use_motion_history", False) and "agent_motion" in self.scenario_dict:
            try:
                self.behaviour_model.initialize_history_from_motion(self.scenario_dict, dt=self.dt)
            except Exception as e:
                print(f"[Simulator] initialize_history_from_motion failed: {e}")

        self.local_frame = {"center": self.ego_trajectory[0, :2].copy(), "yaw": self.ego_trajectory[0, 4].copy()}

        agent_mask = self.ctrl_sim_dset.get_agent_mask(
            copy.deepcopy(self.scenario_dict["agents"][:, :, : self.ctrl_sim_dset.HEAD_IDX + 1]),
            self.local_frame,
        )
        self.agent_active = agent_mask[:, 0]
        self.left_scene = np.zeros_like(self.agent_active).astype(bool)

        self.last_active_agent_position = self.scenario_dict["agents"][:, 0]

        self.initialize_data_dict()

        self.current_state = self._get_observation()
        self._update_viz_state()

        return self.current_state

    def render_state(self, name, movie_path):
        """Render current simulation state with consistent agent IDs."""
        # Get active agent states
        active_mask = self.viz_state["agent_active"]
        agent_states = self.viz_state["agent_states"][active_mask]
        
        # Get original indices of active agents (for consistent labeling)
        original_indices = np.where(active_mask)[0]
        # Agent IDs: 1-indexed for NPCs, will be converted to labels in render
        agent_ids = list(original_indices + 1)  # 1, 2, 3, ... for NPCs
        
        ego_state = normalize_agents(self.ego_state[None, None, :], normalize_dict=self.local_frame)[:, 0]
        states = np.concatenate([agent_states, ego_state], axis=0)
        
        # Add ego ID (will be displayed as "E")
        agent_ids.append(-1)  # -1 signals ego in render function

        agent_types = self.viz_state["agent_types"][active_mask]
        agent_types = np.concatenate([agent_types, np.array([0, 1, 0, 0, 0], dtype=int)[None, :]], axis=0)

        route = self.viz_state["route"]
        lanes = self.viz_state["lanes"]
        lanes_mask = self.viz_state["lanes_mask"]

        render_state(
            states,
            agent_types,
            route,
            lanes,
            lanes_mask,
            self.t,
            name,
            movie_path,
            lightweight=self.cfg.sim.lightweight,
            agent_ids=agent_ids,
        )

    def save_trajectory(self, episode_idx: int):
        save_flag = bool(getattr(self.cfg.sim, "save_trajectory", False))
        save_dir = getattr(self.cfg.sim, "trajectory_path", None)
        if (not save_flag) or save_dir is None:
            return

        os.makedirs(save_dir, exist_ok=True)
        payload = {
            "episode_idx": int(episode_idx),
            "dt": float(self.dt),
            "steps_executed": int(self.t),
            "sim_mode": self.mode,
            "policy": self.cfg.sim.policy,
            "scenario_dict": self.scenario_dict,
            "data_dict": self.data_dict,
        }
        out_path = os.path.join(save_dir, f"episode_{episode_idx}.pkl")
        with open(out_path, "wb") as f:
            pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)


class CtRLSimBehaviourModel:
    """CtRL-Sim Behaviour Model with proper RTG handling and use_rtg support."""
    
    NUM_AGENT_STATES = 8
    NUM_AGENT_TYPES = 5

    def __init__(
        self,
        mode,
        model_path,
        model,
        dset,
        use_rtg,
        predict_rtgs,
        action_temperature,
        tilt,
        steps,
        max_rtg_val: int = 349,  # Now passed dynamically
    ):
        self.mode = mode
        self.model_path = model_path
        self.model = model
        self.model.eval()
        self.dset = dset
        self.cfg_model = model.cfg.model
        self.cfg_dataset = model.cfg.dataset

        self.steps = steps
        self.use_rtg = use_rtg
        self.predict_rtgs = predict_rtgs
        self.action_temperature = action_temperature
        self.tilt = tilt
        self.t = 0
        self.rtg_inference_mode = "auto"
        
        # Dynamic max RTG value from config
        self.max_rtg_val = max_rtg_val
        
        # Get RTG discretization from model config
        if hasattr(self.cfg_dataset, 'rtg') and hasattr(self.cfg_dataset.rtg, 'discretization'):
            self.rtg_K = int(self.cfg_dataset.rtg.discretization)
        elif hasattr(self.cfg_dataset, 'rtg_discretization'):
            self.rtg_K = int(self.cfg_dataset.rtg_discretization)
        else:
            self.rtg_K = 350  # Fallback

        # Validate max_rtg_val
        if self.max_rtg_val >= self.rtg_K:
            print(f"[CtRLSimBehaviourModel] WARNING: max_rtg_val={self.max_rtg_val} >= rtg_K={self.rtg_K}")
            print(f"  Clamping max_rtg_val to {self.rtg_K - 1}")
            self.max_rtg_val = self.rtg_K - 1

        # Performance knobs
        self.use_autocast = True
        self.autocast_dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16

        # Collision avoidance shaping (inference-only)
        col_cfg = getattr(self.cfg_model, "collision_avoidance", None)
        self.collision_avoid_enabled = bool(getattr(col_cfg, "enabled", False)) if col_cfg is not None else False
        self.collision_avoid_beta = float(getattr(col_cfg, "beta", 2.0)) if col_cfg is not None else 2.0
        self.collision_avoid_margin = float(getattr(col_cfg, "margin", 0.2)) if col_cfg is not None else 0.2
        self.collision_avoid_min_dist_clip = float(getattr(col_cfg, "min_dist_clip", 0.05)) if col_cfg is not None else 0.05

        # Motion history
        self.use_motion_history: bool = False
        self.history_from_motion_num_points: int | None = None
        self.motion_history_env = None

        # Metrics
        self.agent_active_all = []
        self.sim_lin_speeds = []
        self.gt_lin_speeds = []
        self.sim_ang_speeds = []
        self.gt_ang_speeds = []
        self.sim_accels = []
        self.gt_accels = []
        self.sim_dist_near_veh = []
        self.gt_dist_near_veh = []
        self.collision_rate_scenario = []
        self.offroad_rate_scenario = []

        self.has_collided = None
        self.has_offroad = None
        self.has_activated = None
        self.has_activated_vehicle = None

        self._last_token_index = None

    def initialize_history_from_motion(self, scenario_dict, dt: float = 0.1):
        """Initialize history from motion code polyline."""
        if not getattr(self, "use_motion_history", False):
            return
        if "agent_motion" not in scenario_dict:
            return
        if not hasattr(self, "states"):
            return

        motion = scenario_dict["agent_motion"]
        agents_init = scenario_dict["agents"][:, 0, :]
        num_env_agents = agents_init.shape[0]

        num_total_agents = self.states.shape[0]
        if num_total_agents != num_env_agents + 1:
            return

        Dm = motion.shape[1]
        if Dm == 0 or Dm % 2 != 0:
            return

        total_points = Dm // 2
        max_hist_pts = self.history_from_motion_num_points if getattr(self, "history_from_motion_num_points", None) is not None else total_points

        T_ctx = int(self.cfg_dataset.train_context_length)
        K_hist = min(total_points, max_hist_pts, max(1, T_ctx - 2))
        if K_hist <= 0:
            return

        hist_env = np.zeros((num_env_agents, K_hist, self.NUM_AGENT_STATES), dtype=np.float32)

        mode = getattr(self, "motion_history_speed_mode", "clamped")
        speed_scale = float(getattr(self, "motion_history_speed_scale", 1.0))
        v_min = float(getattr(self, "motion_history_min_speed", 0.0))
        v_max = float(getattr(self, "motion_history_max_speed", 40.0))
        max_scale = float(getattr(self, "motion_history_max_scale", 2.0))
        max_scale = max(max_scale, 1.0)

        for idx in range(num_env_agents):
            cur_state = agents_init[idx]
            p0 = cur_state[0:2].astype(np.float32)
            heading_now = float(cur_state[4])
            v_static = float(np.linalg.norm(cur_state[2:4]))

            pts_body_full = motion[idx].reshape(-1, 2).astype(np.float32)
            pts_body = pts_body_full[-K_hist:]

            if np.max(np.linalg.norm(pts_body, axis=1)) < 1e-3:
                for t in range(K_hist):
                    hist_env[idx, t, 0:2] = p0
                    hist_env[idx, t, 2:4] = 0.0
                    hist_env[idx, t, 4] = heading_now
                    hist_env[idx, t, 5:7] = cur_state[5:7]
                    hist_env[idx, t, 7] = 1.0
                continue

            cos_h = math.cos(heading_now)
            sin_h = math.sin(heading_now)
            R = np.array([[cos_h, -sin_h], [sin_h, cos_h]], dtype=np.float32)
            pts_world = pts_body @ R.T + p0[None, :]

            if K_hist > 3:
                try:
                    from scipy.ndimage import gaussian_filter1d
                    pts_world[:, 0] = gaussian_filter1d(pts_world[:, 0], sigma=0.5)
                    pts_world[:, 1] = gaussian_filter1d(pts_world[:, 1], sigma=0.5)
                except Exception:
                    pass

            pts_world[-1] = p0

            dirs = np.zeros_like(pts_world, dtype=np.float32)
            disp = np.zeros(K_hist, dtype=np.float32)

            for t in range(K_hist - 1):
                delta = pts_world[t + 1] - pts_world[t]
                d = float(np.linalg.norm(delta))
                disp[t] = d
                if d > 1e-4:
                    dirs[t] = delta / d
                else:
                    dirs[t] = np.array([math.cos(heading_now), math.sin(heading_now)], dtype=np.float32)

            if K_hist >= 2:
                dirs[-1] = dirs[-2]
                disp[-1] = disp[-2]
            else:
                dirs[-1] = np.array([math.cos(heading_now), math.sin(heading_now)], dtype=np.float32)
                disp[-1] = 0.0

            v_poly = disp / max(dt, 1e-3)

            if mode == "from_static":
                base_v = v_static * speed_scale
                base_v = np.clip(base_v, v_min, v_max)
                v_t = np.full(K_hist, base_v, dtype=np.float32)
            elif mode == "from_polyline":
                v_t = v_poly.astype(np.float32) * speed_scale
                v_t = np.clip(v_t, v_min, v_max)
            else:
                v_poly_mean = float(np.mean(v_poly))
                if v_poly_mean > 1e-3 and v_static > 1e-3:
                    s = (v_static * speed_scale) / v_poly_mean
                    s = np.clip(s, 1.0 / max_scale, max_scale)
                    v_t = (v_poly * s).astype(np.float32)
                else:
                    v_t = (v_poly * speed_scale).astype(np.float32)
                v_t = np.clip(v_t, v_min, v_max)

            for t in range(K_hist):
                pos_t = pts_world[t]
                dir_t = dirs[t]
                speed_t = float(v_t[t])

                vx, vy = dir_t * speed_t
                heading_t = math.atan2(dir_t[1], dir_t[0])

                hist_env[idx, t, 0] = pos_t[0]
                hist_env[idx, t, 1] = pos_t[1]
                hist_env[idx, t, 2] = vx
                hist_env[idx, t, 3] = vy
                hist_env[idx, t, 4] = heading_t
                hist_env[idx, t, 5] = float(cur_state[5])
                hist_env[idx, t, 6] = float(cur_state[6])
                hist_env[idx, t, 7] = 1.0

        self.motion_history_env = hist_env

    def update_running_statistics(self, data_dict, scenario_dict, scene_complete=False, offroad_threshold=3.0):
        """Keep only simulation-side invalid agent filtering; remove evaluation statistics."""
        is_vehicle = data_dict["agent_type"][0][:, 1] == 1
        agent_active = data_dict["agent_active"]
        invalid_agents = np.zeros(agent_active.shape[0], dtype=bool)

        t_idx = min(self.t, len(data_dict["agent"]) - 1)
        sim_agents = np.array(data_dict["agent"])[t_idx, agent_active]

        # Maintains the same logic: Collision/offroad checks are performed only when the number of active agents > 1
        if sim_agents.shape[0] <= 1:
            return invalid_agents

        agents_colliding = compute_collision_states_one_scene(modify_agent_states(sim_agents))
        active_agent_idxs = np.where(agent_active == 1)[0]
        colliding_all = np.zeros(len(agent_active), dtype=bool)
        for active_agent_idx, agent_colliding in zip(active_agent_idxs, agents_colliding):
            colliding_all[active_agent_idx] = agent_colliding

        normalize_dict = {
            "center": data_dict["ego"][min(self.t, len(data_dict["ego"]) - 1)][0, :2].copy(),
            "yaw": data_dict["ego"][min(self.t, len(data_dict["ego"]) - 1)][0, 4].copy(),
        }
        lanes, lanes_mask = self.dset.get_normalized_lanes_in_fov(data_dict["lanes"], normalize_dict)
        lanes_resampled = resample_lanes_with_mask(lanes, lanes_mask, num_points=100)

        agents_normalized = normalize_agents(
            data_dict["agent"][t_idx][:, None],
            normalize_dict,
        )
        min_dist_to_lane = np.linalg.norm(
            lanes_resampled.reshape(-1, 2)[None, :] - agents_normalized[:, :, :2],
            axis=-1,
        ).min(1)

        agents_offroad = min_dist_to_lane > offroad_threshold
        agents_offroad[~agent_active] = False
        agents_offroad[~is_vehicle] = False

        invalid_agents = np.logical_or(colliding_all, agents_offroad)
        return invalid_agents

    def compute_metrics(self):
        """Evaluation metrics have been removed. Keep a no-op interface for compatibility."""
        return {}, []

    def reset(self, num_agents):
        """Reset the behaviour model for a new episode."""
        self.t = 0
        self.states = np.zeros((num_agents, self.steps, self.NUM_AGENT_STATES))
        self.types = np.zeros((num_agents, self.NUM_AGENT_TYPES))
        self.actions = np.zeros((num_agents, self.steps))
        # Use dynamic max_rtg_val instead of hardcoded 349
        self.rtgs = np.ones((num_agents, self.steps, self.cfg_model.num_reward_components)) * self.max_rtg_val
        self.motion_history_env = None
        self._last_token_index = None

    def update_state(self, data_dict):
        """Update internal state from data_dict."""
        t_idx = self.t
        self.states[:1, t_idx, :] = data_dict["ego"][t_idx]
        self.states[1:, t_idx, :] = data_dict["agent"][t_idx]

        if t_idx == 0:
            self.types[:1] = data_dict["ego_type"][0]
            self.types[1:] = data_dict["agent_type"][0]

        self.actions[:1, t_idx] = data_dict["ego_action"][t_idx]
        self.rtgs[:1, t_idx, :] = data_dict["ego_rtg"][t_idx]

        if t_idx > 0:
            self.actions[1:, t_idx - 1] = data_dict["agent_action"][t_idx - 1]
            if self.predict_rtgs:
                self.rtgs[1:, t_idx - 1, 0] = data_dict["agent_rtg"][t_idx - 1]

        self.states[1:][~data_dict["agent_active"]] = 0

    def _collision_avoid_shape_logits(
        self,
        logits_bav: torch.Tensor,
        states_baf: torch.Tensor,
        min_dist_clip: float,
        beta: float,
        margin: float,
    ) -> torch.Tensor:
        """Soft collision avoidance shaping."""
        if (not self.collision_avoid_enabled) or (beta <= 0.0):
            return logits_bav

        device = logits_bav.device
        dtype = logits_bav.dtype
        B, A, V = logits_bav.shape

        exists = states_baf[:, :, -1] > 0
        if exists.sum().item() <= 1:
            return logits_bav

        dx = self.model.token_dx.to(device=device, dtype=dtype).view(1, 1, V)
        dy = self.model.token_dy.to(device=device, dtype=dtype).view(1, 1, V)

        x = states_baf[:, :, 0].to(dtype=dtype)
        y = states_baf[:, :, 1].to(dtype=dtype)
        yaw = states_baf[:, :, 4].to(dtype=dtype)
        length = states_baf[:, :, 5].to(dtype=dtype)
        width = states_baf[:, :, 6].to(dtype=dtype)

        radius = 0.5 * torch.sqrt(length * length + width * width).clamp(min=0.1)

        cos = torch.cos(yaw).unsqueeze(-1)
        sin = torch.sin(yaw).unsqueeze(-1)

        x_next = x.unsqueeze(-1) + cos * dx - sin * dy
        y_next = y.unsqueeze(-1) + sin * dx + cos * dy

        p = torch.softmax(logits_bav.float() / float(self.action_temperature), dim=-1).to(dtype=dtype)
        exp_dx = (p * dx).sum(dim=-1)
        exp_dy = (p * dy).sum(dim=-1)
        x_exp = x + torch.cos(yaw) * exp_dx - torch.sin(yaw) * exp_dy
        y_exp = y + torch.sin(yaw) * exp_dx + torch.cos(yaw) * exp_dy

        x_next_e = x_next.unsqueeze(-1)
        y_next_e = y_next.unsqueeze(-1)

        x_other = x_exp.view(B, 1, 1, A)
        y_other = y_exp.view(B, 1, 1, A)

        ddx = x_next_e - x_other
        ddy = y_next_e - y_other
        dist = torch.sqrt(ddx * ddx + ddy * ddy + 1e-6).clamp(min=float(min_dist_clip))

        r_i = radius.view(B, A, 1, 1)
        r_j = radius.view(B, 1, 1, A)
        thr = r_i + r_j + float(margin)

        exists_j = exists.view(B, 1, 1, A)
        eye = torch.eye(A, device=device, dtype=torch.bool).view(1, A, 1, A).expand(B, A, 1, A)
        valid = exists_j.expand(B, A, V, A) & (~eye.expand(B, A, V, A))

        violation = F.relu(thr - dist) * valid.float()
        cost = (violation ** 2).sum(dim=-1)
        cost = cost * exists.view(B, A, 1).float()

        return logits_bav - float(beta) * cost

    def get_motion_data(self, data_dict):
        """Construct CtRLSim motion_data with proper RTG handling.
        
        Key fix: When use_rtg=False, set rtg_mask to zero so RTG embeddings
        are zeroed out, effectively ablating RTG conditioning.
        """
        T_ctx = int(self.cfg_dataset.train_context_length)
        timesteps = np.arange(T_ctx).astype(int)

        num_agents = self.states.shape[0]

        hist_env = self.motion_history_env if (self.use_motion_history and self.motion_history_env is not None) else None

        if hist_env is not None:
            H = hist_env.shape[1]
            sim_T = self.t + 1
            T_eff = H + sim_T

            all_states = np.zeros((num_agents, T_eff, self.NUM_AGENT_STATES), dtype=self.states.dtype)
            all_actions = np.zeros((num_agents, T_eff), dtype=self.actions.dtype)
            all_rtgs = np.ones((num_agents, T_eff, self.cfg_model.num_reward_components), dtype=self.rtgs.dtype) * self.max_rtg_val

            N_env = hist_env.shape[0]
            assert N_env <= num_agents - 1
            all_states[1:1 + N_env, :H, :] = hist_env

            all_states[:, H:H + sim_T, :] = self.states[:, :sim_T, :]
            all_actions[:, H:H + sim_T] = self.actions[:, :sim_T]
            all_rtgs[:, H:H + sim_T, :] = self.rtgs[:, :sim_T, :]
        else:
            sim_T = self.t + 1
            T_eff = sim_T
            all_states = self.states[:, :sim_T].copy()
            all_actions = self.actions[:, :sim_T].copy()
            all_rtgs = self.rtgs[:, :sim_T].copy()

        if T_eff >= T_ctx:
            state_hist = all_states[:, T_eff - T_ctx: T_eff, :]
            action_hist = all_actions[:, T_eff - T_ctx: T_eff]
            rtg_hist = all_rtgs[:, T_eff - T_ctx: T_eff, :]
            token_index = T_ctx - 1
        else:
            state_hist = np.zeros((num_agents, T_ctx, self.NUM_AGENT_STATES), dtype=self.states.dtype)
            action_hist = np.zeros((num_agents, T_ctx), dtype=self.actions.dtype)
            rtg_hist = np.ones((num_agents, T_ctx, self.cfg_model.num_reward_components), dtype=self.rtgs.dtype) * self.max_rtg_val

            state_hist[:, :T_eff, :] = all_states
            action_hist[:, :T_eff] = all_actions
            rtg_hist[:, :T_eff, :] = all_rtgs
            token_index = T_eff - 1

        self._last_token_index = int(token_index)

        # RTG mask: set to zero if use_rtg=False to ablate RTG conditioning
        if self.use_rtg:
            rtg_mask = state_hist[:, :, -1].astype(bool)
        else:
            # Zero out RTG mask to disable RTG conditioning
            rtg_mask = np.zeros_like(state_hist[:, :, -1], dtype=bool)

        timestep_buffer = np.repeat(
            timesteps[np.newaxis, :, np.newaxis],
            self.cfg_dataset.max_num_agents,
            0,
        )

        normalize_timestep = int(token_index)
        normalize_dict = {
            "center": state_hist[0, normalize_timestep, :2].copy(),
            "yaw": state_hist[0, normalize_timestep, 4].copy(),
        }

        agent_mask = self.dset.get_agent_mask(copy.deepcopy(state_hist[:, :, : self.dset.HEAD_IDX + 1]), normalize_dict)
        moving_agent_mask = np.ones(num_agents).astype(bool)

        lanes_norm, lanes_mask = self.dset.get_normalized_lanes_in_fov(data_dict["lanes"], normalize_dict)
        lanes_feat = np.concatenate([lanes_norm, lanes_mask[:, :, None]], axis=-1)

        motion_datas = {}
        correspondences = {}
        motion_data_id = 0

        unaccounted_veh_ids = np.where(data_dict["agent_active"] == 1)[0]
        while len(unaccounted_veh_ids) > 0:
            (
                state_buffer,
                agent_type_buffer,
                agent_mask_buffer,
                action_buffer,
                rtg_buffer,
                rtg_mask_buffer,
                _,
                new_origin_agent_idx,
                correspondence,
            ) = self.dset.select_closest_max_num_agents(
                state_hist,
                self.types,
                agent_mask,
                action_hist,
                rtg_hist[:, :, 0],
                rtg_mask,
                moving_agent_mask,
                origin_agent_idx=0,
                timestep=normalize_timestep,
                active_agents=unaccounted_veh_ids + 1,
            )
            correspondence = correspondence - 1

            state_buffer = normalize_agents(state_buffer, normalize_dict)

            is_ego = np.zeros(len(state_buffer))
            is_ego[new_origin_agent_idx] = 1
            is_ego = is_ego.astype(int)
            is_ego = np.tile(is_ego[:, None, None], (1, self.cfg_dataset.train_context_length, 1))

            state_buffer = np.concatenate([state_buffer[:, :, :-1], is_ego, state_buffer[:, :, -1:]], axis=-1)

            state_buffer[~agent_mask_buffer.astype(bool)] = 0
            rtg_mask_buffer[~agent_mask_buffer.astype(bool)] = 0

            motion_data = dict()
            motion_data["idx"] = self.t
            motion_data["agent"] = from_numpy(
                {
                    "agent_states": add_batch_dim(state_buffer),
                    "agent_types": add_batch_dim(agent_type_buffer),
                    "actions": add_batch_dim(action_buffer),
                    "rtgs": add_batch_dim(rtg_buffer[:, :, None]),
                    "rtg_mask": add_batch_dim(rtg_mask_buffer[:, :, None]),
                    "timesteps": add_batch_dim(timestep_buffer),
                    "moving_agent_mask": add_batch_dim(moving_agent_mask),
                }
            )
            motion_data["map"] = from_numpy({"road_points": add_batch_dim(lanes_feat)})
            motion_data = CtRLSimData(motion_data)

            unaccounted_veh_ids = np.setdiff1d(unaccounted_veh_ids, correspondence[1:])

            motion_datas[motion_data_id] = motion_data
            correspondences[motion_data_id] = correspondence
            motion_data_id += 1

        return motion_datas, correspondences

    def get_tilt_logits(self, tilt):
        """Get tilt logits for RTG sampling."""
        rtg_bin_values = np.zeros((self.rtg_K, 1))
        rtg_bin_values[:, 0] = tilt * np.linspace(0, 1, self.rtg_K)
        return rtg_bin_values

    def _collate_motion_datas(self, motion_datas: dict):
        """Stack multiple CtRLSimData(batch=1) into CtRLSimData(batch=B)."""
        ids = sorted(list(motion_datas.keys()))
        md_list = [motion_datas[i] for i in ids]

        agent_states = torch.cat([md["agent"].agent_states for md in md_list], dim=0)
        agent_types = torch.cat([md["agent"].agent_types for md in md_list], dim=0)
        actions = torch.cat([md["agent"].actions for md in md_list], dim=0)
        rtgs = torch.cat([md["agent"].rtgs for md in md_list], dim=0)
        rtg_mask = torch.cat([md["agent"].rtg_mask for md in md_list], dim=0)
        timesteps = torch.cat([md["agent"].timesteps for md in md_list], dim=0)
        moving_agent_mask = torch.cat([md["agent"].moving_agent_mask for md in md_list], dim=0)

        road_points = torch.cat([md["map"].road_points for md in md_list], dim=0)

        out = {
            "idx": md_list[0]["idx"],
            "agent": {
                "agent_states": agent_states,
                "agent_types": agent_types,
                "actions": actions,
                "rtgs": rtgs,
                "rtg_mask": rtg_mask,
                "timesteps": timesteps,
                "moving_agent_mask": moving_agent_mask,
            },
            "map": {
                "road_points": road_points
            }
        }
        return CtRLSimData(out), ids

    def predict(self, motion_datas, data_dict, correspondences):
        token_index = int(self._last_token_index) if self._last_token_index is not None else -1

        num_env_agents = int(data_dict["agent"][0].shape[0]) if ("agent" in data_dict and len(data_dict["agent"]) > 0) else 0
        data_dict["agent_next_action"] = np.zeros(num_env_agents, dtype=np.int64)
        data_dict["agent_next_rtg"] = np.zeros(num_env_agents, dtype=np.int64)
        
        data_dict["agent_next_action_refined"] = None

        if not motion_datas:
            return data_dict

        # Check if refine head is available and enabled
        refine_enabled = (
            hasattr(self.model.decoder, "predict_refine") 
            and self.model.decoder.predict_refine is not None
            and getattr(self.model.decoder, "refine_use_in_eval", True)
        )
        refine_scale = float(getattr(self.model.decoder, "refine_scale", 1.0)) if refine_enabled else 0.0

        use_continuous_refine = bool(refine_enabled and (refine_scale > 0.0))
        if use_continuous_refine:
            data_dict["agent_next_action_refined"] = np.zeros((num_env_agents, 3), dtype=np.float32)
        one_pass = bool(getattr(self.model.decoder, "rtg_cond_enabled", False))

        try:
            self.model.runtime_rtg_tilt = float(self.tilt) if self.predict_rtgs else 0.0
        except Exception:
            pass

        motion_data_bat, ids = self._collate_motion_datas(motion_datas)
        motion_data_bat = motion_data_bat.cuda()

        def _forward(md):
            with torch.inference_mode():
                if self.use_autocast:
                    with torch.autocast(device_type="cuda", dtype=self.autocast_dtype):
                        return self.model(md, eval=True)
                return self.model(md, eval=True)

        # Pass 1
        preds1 = _forward(motion_data_bat)

        # Sample RTG tokens
        sampled_rtg_tokens = None
        if self.predict_rtgs and ("rtg_preds" in preds1):
            rtg_logits_full = preds1["rtg_preds"][:, :, token_index, :]
            sampled_rtg_tokens = sample_rtg_tokens(
                rtg_logits_full,
                K=self.rtg_K,
                C=int(self.cfg_model.num_reward_components),
                tilt=float(self.tilt),
            )

            for b, motion_data_id in enumerate(ids):
                corr = correspondences[motion_data_id]
                for tensor_id, veh_id in enumerate(corr):
                    if tensor_id == 0:
                        continue
                    data_dict["agent_next_rtg"][veh_id] = int(sampled_rtg_tokens[b, tensor_id, 0].item())

        # Action logits
        need_two_pass_base = (not one_pass) and self.use_rtg and self.predict_rtgs and (sampled_rtg_tokens is not None)
        mode = str(getattr(self, "rtg_inference_mode", "auto")).lower()
        if mode in ("skip_two_pass", "skip", "one_pass_like"):
            need_two_pass = False
        else:
            need_two_pass = need_two_pass_base
        if need_two_pass:
            motion_data_bat["agent"].rtgs[:, :, token_index, : sampled_rtg_tokens.shape[-1]] = sampled_rtg_tokens.to(
                motion_data_bat["agent"].rtgs.device
            )
            motion_data_bat["agent"].rtg_mask[:, :, token_index, 0] = 1.0
            preds2 = _forward(motion_data_bat)
            action_logits_full = preds2["action_preds"][:, :, token_index, :]
            refine_preds_full = preds2.get("refine_preds", None)
        else:
            action_logits_full = preds1["action_preds"][:, :, token_index, :]
            refine_preds_full = preds1.get("refine_preds", None)

        if use_continuous_refine and (refine_preds_full is not None):
            refine_delta_full = refine_preds_full[:, :, token_index, :]  # [B, A, 3]
            refine_delta_np = refine_delta_full.detach().to(torch.float32).cpu().numpy()
        else:
            refine_delta_full = None
            refine_delta_np = None

        # Collision avoidance shaping
        logits_bav = action_logits_full
        if self.collision_avoid_enabled:
            states_bat = motion_data_bat["agent"].agent_states[:, :, token_index, :]
            logits_bav = self._collision_avoid_shape_logits(
                logits_bav,
                states_bat,
                min_dist_clip=self.collision_avoid_min_dist_clip,
                beta=self.collision_avoid_beta,
                margin=self.collision_avoid_margin,
            )

        # Sample actions and compute refined actions
        vocab = self.dset.V  # [K, 3] numpy array
        
        for b, motion_data_id in enumerate(ids):
            corr = correspondences[motion_data_id]
            for tensor_id, veh_id in enumerate(corr):
                if tensor_id == 0:
                    continue
                
                # Sample discrete token
                next_action_logits = logits_bav[b, tensor_id]
                next_action_token = sample_from_logits(
                    next_action_logits, 
                    temperature=float(self.action_temperature), 
                    nucleus_p=None
                )
                k = int(next_action_token.item())
                data_dict["agent_next_action"][veh_id] = k
                
                # Compute refined continuous action
                anchor_delta = vocab[k]  # [3] numpy: (dx, dy, dtheta)

                if (data_dict["agent_next_action_refined"] is not None) and (refine_delta_np is not None):
                    refine_delta = refine_delta_np[b, tensor_id]  # [3] float32
                    refined_action = anchor_delta + refine_scale * refine_delta
                    data_dict["agent_next_action_refined"][veh_id] = refined_action.astype(np.float32)

        return data_dict

    def step(self, data_dict):
        """Execute one simulation step."""
        self.update_state(data_dict)
        motion_datas, correspondences = self.get_motion_data(data_dict)
        data_dict = self.predict(motion_datas, data_dict, correspondences)
        self.t += 1
        return data_dict
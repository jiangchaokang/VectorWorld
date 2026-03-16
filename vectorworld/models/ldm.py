import os
import pickle
import glob
from tqdm import tqdm

from vectorworld.utils.train_helpers import (
    create_lambda_lr_cosine,
    create_lambda_lr_linear,
    create_lambda_lr_constant,
)

from vectorworld.networks.ldm_net import LDM, FlowLDM, MeanFlowLDM
from vectorworld.models.vae import VectorWorldVAE as ScenarioDreamerAutoEncoder
from vectorworld.utils.data_container import ScenarioDreamerData
from torch_geometric.loader import DataLoader
from torch_geometric.data import Batch
from configs.config import (
    PROPORTION_NOCTURNE_COMPATIBLE,
    NON_PARTITIONED,
    NOCTURNE_COMPATIBLE,
)
from vectorworld.utils.pyg_helpers import get_edge_index_complete_graph, get_edge_index_bipartite
from vectorworld.utils.data_helpers import (
    unnormalize_scene,
    normalize_latents,
    unnormalize_latents,
    convert_batch_to_scenarios,
    reorder_indices,
)
from vectorworld.utils.inpainting_helpers import (
    normalize_and_crop_scene,
    sample_num_lanes_agents_inpainting,
)
from vectorworld.utils.sim_env_helpers import sample_route, get_default_route_center_yaw
from vectorworld.utils.lane_graph_helpers import estimate_heading
from vectorworld.utils.torch_helpers import from_numpy
from vectorworld.utils.viz import visualize_batch
from vectorworld.utils.data_helpers import unnormalize_motion_code

import torch
from torch import nn
import pytorch_lightning as pl
from pytorch_lightning.utilities import grad_norm
from torch_ema import ExponentialMovingAverage
from typing import Any, Dict, Optional, Tuple

torch.set_printoptions(sci_mode=False)


class VectorWorldLDM(pl.LightningModule):
    def __init__(self, cfg, cfg_ae):
        super(VectorWorldLDM, self).__init__()

        self.save_hyperparameters()
        self.cfg = cfg
        self.cfg_model = cfg.model
        self.cfg_dataset = cfg.dataset

        ldm_type = getattr(self.cfg_model, "ldm_type").lower()
        if ldm_type == "diffusion":
            self.gen_model = LDM(self.cfg)
        elif ldm_type == "flow":
            self.gen_model = FlowLDM(self.cfg)
        elif ldm_type in ("meanflow", "mf"):
            self.gen_model = MeanFlowLDM(self.cfg)
        else:
            raise ValueError(
                f"Unsupported ldm_type={ldm_type}. Use 'diffusion', 'flow' or 'meanflow'."
            )

        self.autoencoder = ScenarioDreamerAutoEncoder.load_from_checkpoint(
            self.cfg_model.autoencoder_path,
            cfg=cfg_ae,
            map_location="cpu",
        )

        self.cfg_dataset_ae = self.autoencoder.cfg_dataset

        self.init_prob_matrix = torch.load(self.cfg.eval.init_prob_matrix_path)
        self.ema = ExponentialMovingAverage(
            self.gen_model.parameters(), decay=self.cfg.train.ema_decay
        )

        # Set a reasonable default number of sampling steps in the case of MeanFlowLDM (which can be covered in the event phase)
        if ldm_type in ("meanflow", "mf") and hasattr(
            self.gen_model, "set_num_steps_eval"
        ):
            default_steps = int(getattr(self.cfg_model, "meanflow_num_steps_eval", 1))
            self.gen_model.set_num_steps_eval(default_steps)

    def on_train_start(self):
        self.ema.to(self.device)

    def optimizer_step(self, *args, **kwargs):
        super().optimizer_step(*args, **kwargs)
        self.ema.update()

    def _log_losses(self, loss_dict, split="train", batch_size=None):
        if split == "train":
            on_step = True
            on_epoch = False
            key_lambda = lambda s: s
        elif split == "val":
            on_step = False
            on_epoch = True
            key_lambda = lambda s: f"val_{s}"
        elif split == "test":
            on_step = False
            on_epoch = True
            key_lambda = lambda s: f"test_{s}"

        for k, v in loss_dict.items():
            if k == "loss":
                v = v.item()
            self.log(
                key_lambda(k),
                v,
                prog_bar=True,
                on_step=on_step,
                on_epoch=on_epoch,
                sync_dist=True,
                batch_size=batch_size,
            )

        if split == "train":
            opt = self.trainer.optimizers[0]
            if isinstance(opt, dict):
                cur_lr = opt["param_groups"][0]["lr"]
            else:
                cur_lr = opt.param_groups[0]["lr"]
            self.log(
                "lr",
                cur_lr,
                prog_bar=True,
                on_step=True,
                on_epoch=False,
                sync_dist=True,
            )

    def training_step(self, data, batch_idx):
        loss_dict = self.gen_model.loss(data)
        self._log_losses(loss_dict, split="train")
        return loss_dict["loss"]

    def validation_step(self, data, batch_idx):
        with self.ema.average_parameters():
            loss_dict = self.gen_model.loss(data)
            self._log_losses(loss_dict, split="val", batch_size=data.batch_size)

            visualize = self.cfg.train.num_samples_to_visualize > 0
            viz_dir = self.cfg.train.viz_dir

            num_samples = self.cfg.train.num_samples_to_visualize
            if num_samples is not None and num_samples > 0:
                num_samples = min(num_samples, data.batch_size)
                indices = torch.arange(num_samples)
                subset_data_list = data.index_select(indices)
                subset_data = Batch.from_data_list(subset_data_list)

                _, figures = self.forward(
                    subset_data,
                    "train",
                    batch_idx,
                    viz_dir,
                    visualize=visualize,
                    num_samples_to_visualize=num_samples,
                )

                if (
                    self.cfg.train.track
                    and visualize
                    and self.logger is not None
                    and figures is not None
                ):
                    for tag, fig in figures.items():
                        self.logger.experiment.add_figure(
                            tag, fig, global_step=self.global_step
                        )

    def forward(
        self,
        data,
        mode,
        batch_idx,
        viz_dir=None,
        visualize=False,
        num_samples_to_visualize=None,
    ):
        """Forward for validation / generation."""

        data = data.to(self.device)

        agent_latents, lane_latents = self.gen_model.forward(data, mode=mode)
        agent_latents, lane_latents = unnormalize_latents(
            agent_latents,
            lane_latents,
            self.cfg_dataset.agent_latents_mean,
            self.cfg_dataset.agent_latents_std,
            self.cfg_dataset.lane_latents_mean,
            self.cfg_dataset.lane_latents_std,
        )

        ae_model = self.autoencoder.model
        (
            agent_states_full_norm,
            lane_samples_norm,
            agent_types,
            lane_types,
            lane_conn_samples,
        ) = ae_model.forward_decoder_with_motion(agent_latents, lane_latents, data)

        agent_states_full, lane_samples = unnormalize_scene(
            agent_states_full_norm,
            lane_samples_norm,
            fov=self.cfg_dataset.fov,
            min_speed=self.cfg_dataset.min_speed,
            max_speed=self.cfg_dataset.max_speed,
            min_length=self.cfg_dataset.min_length,
            max_length=self.cfg_dataset.max_length,
            min_width=self.cfg_dataset.min_width,
            max_width=self.cfg_dataset.max_width,
            min_lane_x=self.cfg_dataset.min_lane_x,
            min_lane_y=self.cfg_dataset.min_lane_y,
            max_lane_x=self.cfg_dataset.max_lane_x,
            max_lane_y=self.cfg_dataset.max_lane_y,
        )

        state_dim = ae_model.cfg.state_dim
        motion_dim = getattr(ae_model.cfg, "motion_dim", 0)

        agent_samples = agent_states_full[:, :state_dim]

        motion_cfg_ae = getattr(self.cfg_dataset_ae, "motion", None)
        if motion_cfg_ae is not None:
            motion_max_disp = float(getattr(motion_cfg_ae, "max_displacement", 6.0))
        else:
            motion_max_disp = 6.0

        agent_motion_norm = None
        if motion_dim > 0:
            agent_motion_norm = agent_states_full[
                :, state_dim : state_dim + motion_dim
            ]
            agent_motion_phys = unnormalize_motion_code(
                agent_motion_norm, motion_max_displacement=motion_max_disp
            )
            data["agent"].motion = agent_motion_phys

        figures = None
        if visualize:
            print(f"Visualizing batch {batch_idx}...")
            if num_samples_to_visualize is None:
                num_samples_to_visualize = data.batch_size
            num_samples_to_visualize = min(num_samples_to_visualize, data.batch_size)
            figures = visualize_batch(
                num_samples_to_visualize,
                agent_samples,
                lane_samples,
                agent_types,
                lane_types,
                lane_conn_samples,
                data,
                viz_dir,
                epoch=self.current_epoch,
                batch_idx=batch_idx,
                log_to_tb=False,
                agent_motion_gt=None,
                agent_motion_pred=agent_motion_norm,
                motion_max_displacement=motion_max_disp,
            )

        data["agent"].x = agent_samples
        data["lane"].x = lane_samples
        data["agent"].type = torch.nn.functional.one_hot(
            agent_types, num_classes=self.cfg_dataset.num_agent_types
        )
        if self.cfg.dataset_name == "nuplan":
            data["lane"].type = torch.nn.functional.one_hot(
                lane_types, num_classes=self.cfg_dataset.num_lane_types
            )
        data["lane", "to", "lane"].type = lane_conn_samples

        return data, figures

    def _build_ldm_dset_from_ae_dset_for_inpainting(
        self, ae_dset, batch_size, num_samples
    ):
        """Build a pyg dataset for the LDM from a given autoencoder dataset for inpainting."""
        dataloader = DataLoader(
            ae_dset, batch_size=batch_size, shuffle=False, drop_last=False
        )

        data_list = []
        inpainting_prob_matrix = torch.load(self.cfg.eval.inpainting_prob_matrix_path)
        for batch_idx, data in enumerate(dataloader):
            data = data.to(self.device)
            agent_latents, lane_latents, lane_cond_dis_prob = (
                self.autoencoder.model.forward_encoder(data)
            )

            agent_latents, lane_latents = normalize_latents(
                agent_latents,
                lane_latents,
                self.cfg_dataset.agent_latents_mean,
                self.cfg_dataset.agent_latents_std,
                self.cfg_dataset.lane_latents_mean,
                self.cfg_dataset.lane_latents_std,
            )
            cond_lane_ids = data["lane"].ids
            num_lanes_batch, num_agents_batch = sample_num_lanes_agents_inpainting(
                lane_cond_dis_prob,
                data.map_id,
                data.num_lanes,
                data.num_agents,
                self.cfg_dataset.max_num_lanes,
                inpainting_prob_matrix.to(self.device),
            )

            for i in range(data.batch_size):
                if len(data_list) == num_samples:
                    break

                d = ScenarioDreamerData()
                num_lanes = num_lanes_batch[i].item()
                num_agents = num_agents_batch[i].item()
                d["num_lanes"] = num_lanes
                d["num_agents"] = num_agents
                d["map_id"] = data["map_id"][i].item()
                d["lg_type"] = data["lg_type"][i].item()

                cond_agent_mask = torch.zeros(num_agents).bool()
                cond_agent_mask[: data["num_agents"][i]] = True
                cond_lane_mask = torch.zeros(num_lanes).bool()
                cond_lane_mask[: data["num_lanes"][i]] = True

                d["agent"].mask = cond_agent_mask
                d["lane"].mask = cond_lane_mask

                d["lane"].x = torch.empty(
                    (num_lanes, self.cfg_model.lane_latent_dim)
                )
                d["agent"].x = torch.empty(
                    (num_agents, self.cfg_model.agent_latent_dim)
                )

                agent_latents_i = agent_latents[data["agent"].batch == i]
                agent_states_i = data["agent"].x[data["agent"].batch == i]
                lane_latents_i = lane_latents[data["lane"].batch == i]
                lane_states_i = data["lane"].x[data["lane"].batch == i]
                cond_lane_ids_i = cond_lane_ids[data["lane"].batch == i]

                (
                    agent_latents_i,
                    _,
                    lane_latents_i,
                    cond_lane_ids_i,
                    _,
                    _,
                    _,
                ) = reorder_indices(
                    agent_latents_i.cpu().numpy(),
                    agent_latents_i.cpu().numpy(),
                    lane_latents_i.cpu().numpy(),
                    cond_lane_ids_i.cpu().numpy(),
                    get_edge_index_complete_graph(len(lane_latents_i)).numpy(),
                    agent_states_i.cpu().numpy(),
                    lane_states_i.cpu().numpy(),
                    lg_type=0,
                    dataset=self.cfg.dataset_name,
                )
                agent_latents_i = from_numpy(agent_latents_i)
                lane_latents_i = from_numpy(lane_latents_i)
                cond_lane_ids_i = from_numpy(cond_lane_ids_i)

                agents_latents_i_padded = torch.zeros(
                    (num_agents, self.cfg_model.agent_latent_dim)
                )
                agents_latents_i_padded[: agent_latents_i.shape[0], :] = agent_latents_i
                agent_latents_i = agents_latents_i_padded
                lane_latents_i_padded = torch.zeros(
                    (num_lanes, self.cfg_model.lane_latent_dim)
                )
                lane_latents_i_padded[: lane_latents_i.shape[0], :] = lane_latents_i
                lane_latents_i = lane_latents_i_padded
                cond_lane_ids_i_padded = torch.zeros((num_lanes,))
                cond_lane_ids_i_padded[: cond_lane_ids_i.shape[0]] = cond_lane_ids_i
                cond_lane_ids_i = cond_lane_ids_i_padded

                d["lane"].latents = lane_latents_i
                d["agent"].latents = agent_latents_i
                d["lane"].ids = cond_lane_ids_i

                d["lane", "to", "lane"].edge_index = get_edge_index_complete_graph(
                    num_lanes
                )
                d["agent", "to", "agent"].edge_index = get_edge_index_complete_graph(
                    num_agents
                )
                d["lane", "to", "agent"].edge_index = get_edge_index_bipartite(
                    num_lanes, num_agents
                )

                data_list.append(d)

        return data_list

    def _initialize_pyg_dset(
        self,
        mode,
        num_samples,
        batch_size,
        conditioning_path=None,
        nocturne_compatible_only=False,
        conditioning_scenes=None,
    ):
        """Initialize a PyG dataset for generation.

        - init_prob_matrix;
        - lane_conventioned: read from latient cache;
        - inpainting:
            * If`conditioning_scenes`For None, from`conditioning_path`Read pkl;
            * Otherwise construct cond scenes from the list of digs in memory.
        """
        data_list = []
        map_id_counter = 0

        conditioning_files = None
        cond_scenes_list = None
        cond_scene_ids = None

        if mode == "lane_conditioned":
            assert conditioning_path is not None, (
                "conditioning_path must be provided for lane conditioned agent generation"
            )
            if self.cfg.dataset_name == "waymo":
                conditioning_files = sorted(
                    glob.glob(conditioning_path + "/*-of-*_*_0_*.pkl")
                )
            else:
                conditioning_files = sorted(glob.glob(conditioning_path + "/*_0.pkl"))
            conditioning_files = conditioning_files[:num_samples]

        elif mode == "inpainting":
            if conditioning_scenes is None:
                assert (
                    conditioning_path is not None
                ), "conditioning_path must be provided for inpainting generation"
                conditioning_files = sorted(glob.glob(conditioning_path + "/*.pkl"))
                conditioning_files = conditioning_files[:num_samples]
            else:
                # Convention_screensdict(id->dict)or list[dict]
                if isinstance(conditioning_scenes, dict):
                    cond_scene_ids = list(conditioning_scenes.keys())
                    cond_scenes_list = [conditioning_scenes[_id] for _id in cond_scene_ids]
                else:
                    cond_scenes_list = list(conditioning_scenes)
                    cond_scene_ids = [str(i) for i in range(len(cond_scenes_list))]

                if num_samples > len(cond_scenes_list):
                    num_samples = len(cond_scenes_list)
                cond_scene_ids = cond_scene_ids[:num_samples]
                cond_scenes_list = cond_scenes_list[:num_samples]

        for i in range(num_samples):
            d = ScenarioDreamerData()

            if mode == "initial_scene":
                if self.cfg.dataset_name == "waymo":
                    if nocturne_compatible_only:
                        map_id = torch.tensor(NOCTURNE_COMPATIBLE)
                    else:
                        map_id = torch.multinomial(
                            torch.tensor(
                                [
                                    1 - PROPORTION_NOCTURNE_COMPATIBLE,
                                    PROPORTION_NOCTURNE_COMPATIBLE,
                                ]
                            ),
                            1,
                        )
                else:
                    map_id = map_id_counter
                    map_id_counter = (map_id_counter + 1) % self.cfg_dataset.num_map_ids

                lane_agent_probs = self.init_prob_matrix[map_id].reshape(1, -1)
                folded_num_lanes_agents = torch.multinomial(
                    lane_agent_probs, 1
                ).squeeze(-1)
                num_lanes = (
                    folded_num_lanes_agents
                    // (self.cfg_dataset.max_num_agents + 1)
                ).item()
                num_agents = (
                    folded_num_lanes_agents
                    % (self.cfg_dataset.max_num_agents + 1)
                ).item()
                assert num_lanes > 0 and num_agents > 0

                lg_type = NON_PARTITIONED

                d["map_id"] = int(map_id)
                d["lg_type"] = int(lg_type)
                d["num_lanes"] = int(num_lanes)
                d["num_agents"] = int(num_agents)
                d["lane"].x = torch.empty(
                    (num_lanes, self.cfg_model.lane_latent_dim)
                )
                d["agent"].x = torch.empty(
                    (num_agents, self.cfg_model.agent_latent_dim)
                )
                d["lane", "to", "lane"].edge_index = get_edge_index_complete_graph(
                    num_lanes
                )
                d["agent", "to", "agent"].edge_index = get_edge_index_complete_graph(
                    num_agents
                )
                d["lane", "to", "agent"].edge_index = get_edge_index_bipartite(
                    num_lanes, num_agents
                )
                data_list.append(d)

            elif mode == "inpainting":
                if conditioning_scenes is None:
                    conditioning_file = conditioning_files[i]
                    with open(conditioning_file, "rb") as f:
                        cond_d = pickle.load(f)
                else:
                    cond_d = cond_scenes_list[i]

                required_keys = [
                    "num_agents",
                    "num_lanes",
                    "road_points",
                    "agent_states",
                    "road_connection_types",
                    "map_id",
                ]
                missing = [k for k in required_keys if k not in cond_d]
                if len(missing) > 0:
                    raise KeyError(
                        f"[LDM.inpainting] conditioning scene is missing required fields {missing}."
                    )

                if "route" in cond_d:
                    route = cond_d["route"]
                    center = route[-1]
                    _, yaw = estimate_heading(route)
                else:
                    route, found_route = sample_route(
                        cond_d, dataset=self.cfg.dataset_name
                    )
                    if found_route:
                        center = route[-1]
                        _, yaw = estimate_heading(route)
                    else:
                        center, yaw = get_default_route_center_yaw(
                            dataset=self.cfg.dataset_name
                        )

                normalize_dict = {"center": center, "yaw": yaw}

                motion_dim = getattr(self.autoencoder.model.cfg, "motion_dim", 0)
                d = normalize_and_crop_scene(
                    cond_d,
                    d,
                    normalize_dict,
                    self.cfg_dataset,
                    self.cfg.dataset_name,
                    motion_dim=motion_dim,
                )
                data_list.append(d)

            elif mode == "lane_conditioned":
                conditioning_file = conditioning_files[i]
                # Convention_file is a complete path. Don't start again.
                with open(conditioning_file, "rb") as f:
                    cond_d = pickle.load(f)

                agent_states = cond_d["agent_states"]
                road_points = cond_d["road_points"]
                lane_mu = cond_d["lane_mu"]
                agent_mu = cond_d["agent_mu"]
                lane_log_var = cond_d["lane_log_var"]
                agent_log_var = cond_d["agent_log_var"]
                edge_index_lane_to_lane = cond_d["edge_index_lane_to_lane"]
                edge_index_lane_to_agent = cond_d["edge_index_lane_to_agent"]
                edge_index_agent_to_agent = cond_d["edge_index_agent_to_agent"]
                scene_type = cond_d["scene_type"]
                if self.cfg.dataset_name == "nuplan":
                    map_id = cond_d["map_id"]
                else:
                    map_id = cond_d["nocturne_compatible"]
                num_lanes = lane_mu.shape[0]
                num_agents = agent_mu.shape[0]

                (
                    agent_mu,
                    agent_log_var,
                    lane_mu,
                    lane_log_var,
                    edge_index_lane_to_lane,
                    _,
                    _,
                ) = reorder_indices(
                    agent_mu,
                    agent_log_var,
                    lane_mu,
                    lane_log_var,
                    edge_index_lane_to_lane,
                    agent_states,
                    road_points,
                    scene_type,
                    dataset=self.cfg.dataset_name,
                )
                edge_index_lane_to_lane = torch.from_numpy(edge_index_lane_to_lane)

                d["map_id"] = map_id
                d["lg_type"] = scene_type
                d["num_lanes"] = num_lanes
                d["num_agents"] = num_agents

                _, lane_latents = normalize_latents(
                    torch.empty((num_agents, self.cfg_model.agent_latent_dim)),
                    from_numpy(lane_mu),
                    self.cfg_dataset.agent_latents_mean,
                    self.cfg_dataset.agent_latents_std,
                    self.cfg_dataset.lane_latents_mean,
                    self.cfg_dataset.lane_latents_std,
                )

                d["lane"].x = torch.empty(
                    (num_lanes, self.cfg_model.lane_latent_dim)
                )
                d["agent"].x = torch.empty(
                    (num_agents, self.cfg_model.agent_latent_dim)
                )

                d["lane"].latents = lane_latents
                d["lane", "to", "lane"].edge_index = from_numpy(
                    edge_index_lane_to_lane
                )
                d["agent", "to", "agent"].edge_index = from_numpy(
                    edge_index_agent_to_agent
                )
                d["lane", "to", "agent"].edge_index = from_numpy(
                    edge_index_lane_to_agent
                )
                data_list.append(d)

        # It's infinity.LDMdataset (acting on lotts)
        if mode == "inpainting":
            data_list = self._build_ldm_dset_from_ae_dset_for_inpainting(
                data_list, batch_size, num_samples
            )

        if conditioning_scenes is not None and mode == "inpainting":
            conditioning_filenames = cond_scene_ids
        else:
            conditioning_filenames = (
                [
                    os.path.splitext(os.path.basename(f))[0]
                    for f in conditioning_files
                ]
                if conditioning_files is not None
                else None
            )

        return data_list, conditioning_filenames

    def generate(
        self,
        mode,
        num_samples,
        batch_size,
        cache_samples=False,
        visualize=False,
        conditioning_path=None,
        cache_dir=None,
        viz_dir=None,
        save_wandb=False,
        return_samples=False,
        nocturne_compatible_only=False,
        conditioning_scenes=None,
    ):
        """Generate samples using the diffusion/flow/meanflow model."""
        if conditioning_path is not None and mode != "initial_scene":
            assert (
                len(os.listdir(conditioning_path)) >= num_samples
            ), f"Not enough conditioning samples in {conditioning_path}"

        # MeanFlow: override eval steps
        ldm_type = getattr(self.cfg_model, "ldm_type", "diffusion").lower()
        if ldm_type in ("meanflow", "mf") and hasattr(self.gen_model, "set_num_steps_eval"):
            num_steps_eval = getattr(self.cfg.eval, "meanflow_num_steps", None)
            if num_steps_eval is not None:
                self.gen_model.set_num_steps_eval(int(num_steps_eval))

        print(f"Generating {num_samples} samples with mode={mode}")

        self.eval()
        scenarios = {}

        with torch.no_grad():
            with self.ema.average_parameters():
                # Init dataset
                dset, conditioning_filenames = self._initialize_pyg_dset(
                    mode,
                    num_samples,
                    batch_size,
                    conditioning_path,
                    nocturne_compatible_only,
                    conditioning_scenes=conditioning_scenes,
                )

                dataloader = DataLoader(dset, batch_size=batch_size, shuffle=False, drop_last=False)

                for batch_idx, data in enumerate(tqdm(dataloader)):
                    data, _ = self.forward(
                        data, mode, batch_idx, viz_dir=viz_dir, visualize=visualize
                    )

                    batch_of_scenarios = convert_batch_to_scenarios(
                        data,
                        batch_size=batch_size,
                        batch_idx=batch_idx,
                        cache_dir=cache_dir,
                        conditioning_filenames=conditioning_filenames,
                        cache_samples=cache_samples,
                        cache_lane_types=self.cfg.dataset_name == "nuplan",
                        mode=mode,
                        output_ids=None,
                    )
                    scenarios.update(batch_of_scenarios)

        return scenarios if return_samples else None

    def on_before_optimizer_step(self, optimizer):
        norms_encoder = grad_norm(self.gen_model.model, norm_type=2)
        self.log_dict(norms_encoder)

    def on_save_checkpoint(self, checkpoint):
        checkpoint["ema_state_dict"] = self.ema.state_dict()

    def on_load_checkpoint(self, checkpoint):
        """Robust EMA loading for backward compatibility.

        - If not checkpoint`ema_state_dict`, or the number of parameters does not match:
          Reinitiate with the current gen_model weightEMAInstead of making a mistake.
        """
        ema_state = checkpoint.get("ema_state_dict", None)
        if ema_state is None:
            # Old checkpoint not savedEMA: Re-introduction
            print(
                "[VectorWorldLDM] No EMA state in checkpoint; "
                "re-initializing EMA from current model weights."
            )
            self.ema = ExponentialMovingAverage(
                self.gen_model.parameters(), decay=self.cfg.train.ema_decay
            )
            return

        try:
            self.ema.load_state_dict(ema_state)
        except ValueError as e:
            # Typical scenario: Model structure added/reduced parameters (e.g. increased MeanFlow 2-time channel)
            print(
                "[VectorWorldLDM] Warning: EMA state dict incompatible with current "
                f"model ({e}). Re-initializing EMA from current model weights."
            )
            self.ema = ExponentialMovingAverage(
                self.gen_model.parameters(), decay=self.cfg.train.ema_decay
            )

    def configure_optimizers(self):
        """
        Robust AdamW parameter grouping (decay vs no_decay).

        Root-cause of previous AssertionError:
        - Old logic used `if "bias" in param_name`, which mis-classified weights of modules
        whose *module name* contains the substring "bias" (e.g. `edge_to_bias.weight`).
        - Old logic iterated `module.named_parameters()` with recurse=True, causing the same
        parameter to be visited under different modules and thus potentially placed into
        both sets.

        Fix:
        - Iterate only direct params per module: `named_parameters(recurse=False)`
        - Identify biases by suffix / direct-name patterns, not substring matching.
        - Provide a safe fallback so every trainable param is assigned exactly once.
        """
        decay: set[str] = set()
        no_decay: set[str] = set()

        whitelist_weight_modules = (
            nn.Linear,
            nn.Conv1d,
            nn.Conv2d,
            nn.Conv3d,
            nn.MultiheadAttention,
            nn.LSTM,
            nn.LSTMCell,
            nn.GRU,
            nn.GRUCell,
        )
        blacklist_weight_modules = (
            nn.BatchNorm1d,
            nn.BatchNorm2d,
            nn.BatchNorm3d,
            nn.LayerNorm,
            nn.Embedding,
        )

        # 1) Classify parameters
        for module_name, module in self.gen_model.named_modules():
            for param_name, param in module.named_parameters(recurse=False):
                if not param.requires_grad:
                    continue

                full_param_name = f"{module_name}.{param_name}" if module_name else param_name

                # ---- Bias-like params (NO weight decay) ----
                # direct "bias" for Linear/Conv, plus MHA special bias params like bias_k/bias_v
                if param_name == "bias" or param_name.endswith("_bias") or param_name.endswith("bias") or param_name.startswith("bias_"):
                    no_decay.add(full_param_name)
                    continue

                # ---- Weight-like params ----
                if param_name.endswith("weight") or param_name.endswith("_weight"):
                    if isinstance(module, whitelist_weight_modules):
                        decay.add(full_param_name)
                    elif isinstance(module, blacklist_weight_modules):
                        no_decay.add(full_param_name)
                    else:
                        # Fallback: matrix/tensor weights decay; vector weights no_decay
                        if param.ndim >= 2:
                            decay.add(full_param_name)
                        else:
                            no_decay.add(full_param_name)
                    continue

                # ---- Everything else (typically scales/pos_emb/custom params) -> no_decay ----
                no_decay.add(full_param_name)

        # 2) Build param dict (trainable only)
        param_dict = {n: p for n, p in self.gen_model.named_parameters() if p.requires_grad}

        inter_params = decay & no_decay
        if len(inter_params) != 0:
            # Print a small helpful debug list then fail loudly.
            bad = sorted(list(inter_params))[:50]
            raise RuntimeError(
                "[configure_optimizers] Found params in both decay and no_decay sets "
                f"(count={len(inter_params)}). Example(s):\n  - " + "\n  - ".join(bad)
            )

        union_params = decay | no_decay
        missing = sorted(list(param_dict.keys() - union_params))
        if len(missing) != 0:
            raise RuntimeError(
                "[configure_optimizers] Some trainable parameters were not assigned to any "
                f"optimizer group (count={len(missing)}). Example(s):\n  - "
                + "\n  - ".join(missing[:50])
            )

        # 3) Create optimizer groups
        optim_groups = [
            {
                "params": [param_dict[pn] for pn in sorted(list(decay))],
                "weight_decay": float(self.cfg.train.weight_decay),
            },
            {
                "params": [param_dict[pn] for pn in sorted(list(no_decay))],
                "weight_decay": 0.0,
            },
        ]

        optimizer = torch.optim.AdamW(
            optim_groups,
            lr=float(self.cfg.train.lr),
            betas=(float(self.cfg.train.beta_1), float(self.cfg.train.beta_2)),
            eps=float(self.cfg.train.epsilon),
        )

        # 4) LR scheduler
        if self.cfg.train.lr_schedule == "cosine":
            scheduler = torch.optim.lr_scheduler.LambdaLR(
                optimizer=optimizer, lr_lambda=create_lambda_lr_cosine(self.cfg)
            )
        elif self.cfg.train.lr_schedule == "linear":
            scheduler = torch.optim.lr_scheduler.LambdaLR(
                optimizer=optimizer, lr_lambda=create_lambda_lr_linear(self.cfg)
            )
        elif self.cfg.train.lr_schedule == "constant":
            scheduler = torch.optim.lr_scheduler.LambdaLR(
                optimizer=optimizer, lr_lambda=create_lambda_lr_constant(self.cfg)
            )
        else:
            scheduler = torch.optim.lr_scheduler.LambdaLR(
                optimizer=optimizer, lr_lambda=lambda step: 1.0
            )

        return [optimizer], {
            "scheduler": scheduler,
            "interval": "step",
            "frequency": 1,
        }
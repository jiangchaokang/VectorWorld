import os
import pickle

import torch
import torch.nn.functional as F
from torch import nn
import pytorch_lightning as pl
from pytorch_lightning.utilities import grad_norm
from torch_geometric.data import Batch
from torch_geometric.loader import DataLoader

from vectorworld.networks.vae_net import AutoEncoder
from vectorworld.utils.train_helpers import (
    create_lambda_lr_cosine,
    create_lambda_lr_linear,
    weight_init,
)
from vectorworld.utils.data_helpers import unnormalize_scene, reparameterize
from vectorworld.utils.pyg_helpers import get_edge_index_bipartite, get_edge_index_complete_graph
from vectorworld.utils.viz import visualize_batch


def _worker_init_fn(worker_id):
    os.sched_setaffinity(0, range(os.cpu_count()))


class VectorWorldVAE(pl.LightningModule):
    """Scene VAE: encode vectorized driving scenes into compact latents.

    NOTE: The attribute name `self.model` MUST remain unchanged for checkpoint
    compatibility. Existing checkpoints store parameters under `model.encoder.*`
    and `model.decoder.*`.
    """

    def __init__(self, cfg):
        super().__init__()
        self.save_hyperparameters()
        self.cfg = cfg
        self.cfg_dataset = self.cfg.dataset

        # IMPORTANT: attribute name 'model' is frozen for checkpoint compat
        self.model = AutoEncoder(cfg.model)

        # Latent caching config
        self._setup_latent_caching()

        # Running latent stats buffers (non-persistent, won't enter state_dict)
        for prefix in ("agent", "lane"):
            for suffix in ("sum", "sumsq"):
                self.register_buffer(
                    f"test_{prefix}_latent_{suffix}",
                    torch.zeros(1, dtype=torch.float32),
                    persistent=False,
                )
            self.register_buffer(
                f"test_{prefix}_latent_count",
                torch.zeros(1, dtype=torch.long),
                persistent=False,
            )

    def _setup_latent_caching(self):
        """Initialize latent caching configuration (COS + local)."""
        cache_cfg = getattr(self.cfg.eval, "cache_latents", None)

        self.nocturne_compatible_filenames = None
        if (
            cache_cfg is not None
            and bool(getattr(cache_cfg, "enable_caching", False))
            and self.cfg.dataset_name == "waymo"
        ):
            with open(cache_cfg.nocturne_train_filenames_path, "rb") as f:
                train_fns = pickle.load(f)
            with open(cache_cfg.nocturne_val_filenames_path, "rb") as f:
                val_fns = pickle.load(f)
            self.nocturne_compatible_filenames = train_fns + val_fns

        self.latent_save_to_cos = False
        self.latent_save_local_copy = True
        self.latent_cos_client = None

        if cache_cfg is not None:
            self.latent_save_to_cos = bool(getattr(cache_cfg, "save_to_cos", False))
            self.latent_save_local_copy = bool(getattr(cache_cfg, "save_local_copy", True))

            if self.latent_save_to_cos:
                try:
                    from dap.utils.cos import Cos
                    from dap.config import global_config
                    global_config.cos_secret_id = getattr(cache_cfg, "cos_secret_id", "")
                    global_config.cos_secret_key = getattr(cache_cfg, "cos_secret_key", "")
                    self.latent_cos_client = Cos()
                    self.latent_cos_bucket = getattr(cache_cfg, "cos_bucket", "")
                    self.latent_cos_prefix = getattr(cache_cfg, "cos_prefix", "")
                except Exception as e:
                    print(f"[VectorWorldVAE] COS init failed: {e}")
                    self.latent_save_to_cos = False

            if not self.latent_save_to_cos and not self.latent_save_local_copy:
                self.latent_save_local_copy = True

    def forward(self, data):
        """Forward pass: encode -> decode -> unnormalize.

        Returns:
            agent_samples: (Na, 7) unnormalized static state
            lane_samples: (Nl, P, 2) unnormalized lane points
            agent_types, lane_types, lane_conn_samples: predictions
            lane_cond_dis: lane count distribution (for inpainting)
            agent_motion_pred: (Na, M) normalized motion code or None
        """
        agent_latents, lane_latents, lane_cond_dis = self.model.forward_encoder(data)

        agent_full, lane_pred, agent_types, lane_types, lane_conn = (
            self.model.forward_decoder_with_motion(agent_latents, lane_latents, data)
        )

        agent_full, lane_pred = unnormalize_scene(
            agent_full, lane_pred,
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

        state_dim = self.cfg.model.state_dim
        motion_dim = getattr(self.cfg.model, "motion_dim", 0)
        agent_samples = agent_full[:, :state_dim]
        agent_motion = agent_full[:, state_dim:state_dim + motion_dim] if motion_dim > 0 else None

        return agent_samples, lane_pred, agent_types, lane_types, lane_conn, lane_cond_dis, agent_motion

    def training_step(self, data, batch_idx):
        loss_dict = self.model.loss(data)
        self._log_losses(loss_dict, "train")
        return loss_dict["loss"]

    def validation_step(self, data, batch_idx):
        loss_dict = self.model.loss(data)
        self._log_losses(loss_dict, "val", batch_size=data.batch_size)

    def test_step(self, data, batch_idx):
        if self.cfg.eval.cache_latents.enable_caching:
            self._cache_latents(data)
        else:
            loss_dict = self.model.loss(data)
            self._log_losses(loss_dict, "test", batch_size=data.batch_size)

    def _log_losses(self, loss_dict, split, batch_size=None):
        on_step = split == "train"
        on_epoch = split != "train"
        prefix = "" if split == "train" else f"{split}_"

        for k, v in loss_dict.items():
            val = v.item() if k == "loss" else v
            self.log(
                f"{prefix}{k}", val,
                prog_bar=True, on_step=on_step, on_epoch=on_epoch,
                sync_dist=True, batch_size=batch_size,
            )

        if split == "train":
            self.log("lr", self.trainer.optimizers[0].param_groups[0]["lr"],
                     prog_bar=True, on_step=True)

    def _cache_latents(self, data):
        """Encode scenes and cache latent representations."""
        agent_mu, lane_mu, agent_log_var, lane_log_var = self.model.forward(data, return_latents=True)

        agent_mu_np = agent_mu.detach().cpu().numpy()
        lane_mu_np = lane_mu.detach().cpu().numpy()
        agent_lv_np = agent_log_var.detach().cpu().numpy()
        lane_lv_np = lane_log_var.detach().cpu().numpy()

        agent_batch = data["agent"].batch.cpu().numpy()
        lane_batch = data["lane"].batch.cpu().numpy()
        scene_type = data["lg_type"].cpu().int()
        road_points = data["lane"].x.cpu().numpy()
        agent_states = data["agent"].x.cpu().numpy()

        split_name = self.cfg.eval.cache_latents.split_name
        latent_root = os.path.join(self.cfg.eval.cache_latents.latent_dir, split_name)

        for i in range(data.batch_size):
            idx = data.idx[i].item()
            filename = os.path.basename(self.files[idx])

            am = agent_mu_np[agent_batch == i]
            lm = lane_mu_np[lane_batch == i]
            alv = agent_lv_np[agent_batch == i]
            llv = lane_lv_np[lane_batch == i]

            n_l = lm.shape[0]
            n_a = am.shape[0]

            d = {
                "idx": idx,
                "agent_mu": am, "lane_mu": lm,
                "agent_log_var": alv, "lane_log_var": llv,
                "edge_index_lane_to_lane": get_edge_index_complete_graph(n_l).numpy(),
                "edge_index_agent_to_agent": get_edge_index_complete_graph(n_a).numpy(),
                "edge_index_lane_to_agent": get_edge_index_bipartite(n_l, n_a).numpy(),
                "scene_type": scene_type[i].item(),
                "road_points": road_points[lane_batch == i],
                "agent_states": agent_states[agent_batch == i],
            }

            if self.cfg.dataset_name == "waymo":
                fn_parts = filename.split(".")[1].split("_")[:2]
                train_fn = f"{fn_parts[0]}_{fn_parts[1]}"
                d["nocturne_compatible"] = (
                    1 if self.nocturne_compatible_filenames and train_fn in self.nocturne_compatible_filenames
                    else 0
                )
            else:
                d["map_id"] = int(data["map_id"][i].item())

            if self.latent_save_local_copy:
                local_path = os.path.join(latent_root, filename)
                os.makedirs(os.path.dirname(local_path), exist_ok=True)
                with open(local_path, "wb") as f:
                    pickle.dump(d, f, protocol=pickle.HIGHEST_PROTOCOL)

    def on_before_optimizer_step(self, optimizer):
        self.log_dict(grad_norm(self.model, norm_type=2))

    def configure_optimizers(self):
        """Build optimizer with proper weight-decay grouping.

        Design principle
        ----------------
        ``self.named_modules()`` visits every node in the module tree,
        and ``module.named_parameters()`` (recurse=True by default) yields
        **all** descendant parameters relative to that module.  Thus the
        same parameter is visited multiple times — once per ancestor.

        To avoid putting a parameter into both ``decay`` and ``no_decay``:

        * **bias** → always ``no_decay`` (idempotent across visits).
        * **weight** → only categorised when the **current module** is in
          the whitelist (→ ``decay``) or blacklist (→ ``no_decay``).
          For container modules that belong to neither list, the parameter
          is **skipped** and will be correctly handled when the leaf
          module is eventually visited.
        * **other** (e.g. ``nn.Parameter`` registered directly) →
          always ``no_decay`` (idempotent).

        This three-way branching guarantees ``decay ∩ no_decay == ∅``.
        """
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (
            nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d,
            nn.MultiheadAttention, nn.LSTM, nn.LSTMCell, nn.GRU, nn.GRUCell,
        )
        blacklist_weight_modules = (
            nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d,
            nn.LayerNorm, nn.Embedding,
        )

        for module_name, module in self.named_modules():
            for param_name, _ in module.named_parameters():
                full_param_name = (
                    f"{module_name}.{param_name}" if module_name else param_name
                )
                if "bias" in param_name:
                    no_decay.add(full_param_name)
                elif "weight" in param_name:
                    if isinstance(module, whitelist_weight_modules):
                        decay.add(full_param_name)
                    elif isinstance(module, blacklist_weight_modules):
                        no_decay.add(full_param_name)
                    # else: intentionally skip — will be categorised when
                    #        visited from its owning leaf module
                elif not ("weight" in param_name or "bias" in param_name):
                    no_decay.add(full_param_name)

        # Safety assertions (catch grouping bugs before they reach the optimizer)
        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, (
            f"Parameters in both decay and no_decay: {inter_params}"
        )
        assert len(param_dict.keys() - union_params) == 0, (
            f"Parameters not assigned to any group: {param_dict.keys() - union_params}"
        )

        optim_groups = [
            {
                "params": [param_dict[pn] for pn in sorted(decay)],
                "weight_decay": self.cfg.train.weight_decay,
            },
            {
                "params": [param_dict[pn] for pn in sorted(no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = torch.optim.AdamW(
            optim_groups,
            lr=self.cfg.train.lr,
            weight_decay=self.cfg.train.weight_decay,
            betas=(self.cfg.train.beta_1, self.cfg.train.beta_2),
            eps=self.cfg.train.epsilon,
        )

        if self.cfg.train.lr_schedule == "cosine":
            scheduler = torch.optim.lr_scheduler.LambdaLR(
                optimizer, create_lambda_lr_cosine(self.cfg)
            )
        elif self.cfg.train.lr_schedule == "linear":
            scheduler = torch.optim.lr_scheduler.LambdaLR(
                optimizer, create_lambda_lr_linear(self.cfg)
            )
        else:
            scheduler = torch.optim.lr_scheduler.LambdaLR(
                optimizer, lambda step: 1.0
            )

        return [optimizer], {"scheduler": scheduler, "interval": "step", "frequency": 1}

    def test_dataloader(self):
        """Build test dataloader for latent caching or evaluation."""
        cache_cfg = self.cfg.eval.cache_latents

        if cache_cfg.enable_caching:
            split = cache_cfg.split_name
        else:
            split = "val"

        if self.cfg.dataset_name == "waymo":
            from vectorworld.data.waymo.vae_dataset import WaymoDatasetAutoEncoder as DS
        else:
            from vectorworld.data.nuplan.vae_dataset import NuplanDatasetAutoEncoder as DS

        dataset = DS(self.cfg_dataset, split_name=split, mode="eval")

        if cache_cfg.enable_caching:
            num_shards = int(getattr(cache_cfg, "num_shards", 1))
            shard_id = int(getattr(cache_cfg, "shard_id", 0))
            if num_shards > 1:
                dataset.files = dataset.files[shard_id::num_shards]
                dataset.dset_len = len(dataset.files)
            self.files = dataset.files.copy()
            latent_dir = os.path.join(cache_cfg.latent_dir, split)
            if self.latent_save_local_copy:
                os.makedirs(latent_dir, exist_ok=True)

        return DataLoader(
            dataset,
            batch_size=self.cfg.datamodule.val_batch_size,
            shuffle=False,
            num_workers=self.cfg.datamodule.num_workers,
            pin_memory=self.cfg.datamodule.pin_memory,
            drop_last=False,
            worker_init_fn=_worker_init_fn,
        )
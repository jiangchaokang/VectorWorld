"""Unified training entry point for all VectorWorld models.

Usage:
    python tools/train.py dataset_name=waymo model_name=vae
    python tools/train.py dataset_name=waymo model_name=ldm
    python tools/train.py dataset_name=waymo model_name=deltasim
"""
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import hydra
import torch
from collections import OrderedDict

torch.set_float32_matmul_precision("medium")
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint, ModelSummary
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.loggers import TensorBoardLogger
from hydra.utils import instantiate
from omegaconf import OmegaConf

from configs.config import CONFIG_PATH
from vectorworld.utils.train_helpers import cache_latent_stats, set_latent_stats


def safe_load_state_dict(model, state_dict, ignore_substrings=None,
                         rename_substrings=None, strict_shapes=True, verbose=True):
    """Load state dict with optional key remapping and shape filtering."""
    if ignore_substrings is None:
        ignore_substrings = []
    if rename_substrings is None:
        rename_substrings = {"diff_model.": "gen_model."}

    if "state_dict" in state_dict and isinstance(state_dict["state_dict"], dict):
        state_dict = state_dict["state_dict"]

    # Remap keys
    remapped = OrderedDict()
    for k, v in state_dict.items():
        new_k = k
        for old, new in rename_substrings.items():
            if new_k.startswith(old):
                new_k = new + new_k[len(old):]
        remapped[new_k] = v

    # Filter
    if ignore_substrings:
        remapped = OrderedDict(
            (k, v) for k, v in remapped.items()
            if not any(sub in k for sub in ignore_substrings)
        )

    # Match shapes
    msd = model.state_dict()
    keep = OrderedDict()
    dropped = []
    for k, v in remapped.items():
        if k not in msd:
            dropped.append((k, "not-in-model"))
            continue
        if isinstance(v, torch.Tensor) and v.shape != msd[k].shape:
            if strict_shapes:
                dropped.append((k, f"shape {tuple(v.shape)} vs {tuple(msd[k].shape)}"))
                continue
        keep[k] = v

    ret = model.load_state_dict(keep, strict=False)
    if verbose:
        print(f"[safe-load] loaded={len(keep)} dropped={len(dropped)} "
              f"missing={len(ret.missing_keys)} unexpected={len(ret.unexpected_keys)}")
    return ret


def build_logger(cfg, save_dir):
    if cfg.train.track:
        return TensorBoardLogger(save_dir=save_dir, name="tb", default_hp_metric=False)
    return None


def train_vae(cfg, save_dir):
    """Train the VectorWorld VAE (scene autoencoder)."""
    from vectorworld.models.vae import VectorWorldVAE

    datamodule = instantiate(cfg.datamodule, dataset_cfg=cfg.dataset)
    model = VectorWorldVAE(cfg)

    init_ckpt = getattr(cfg.train, "init_ckpt_path", None)
    resume_ckpt = getattr(cfg.train, "resume_ckpt_path", None)

    if init_ckpt and not resume_ckpt:
        print(f"[train] Initializing VAE from: {init_ckpt}")
        ckpt = torch.load(init_ckpt, map_location="cpu", weights_only=False)
        safe_load_state_dict(model, ckpt.get("state_dict", ckpt), strict_shapes=False)

    trainer = pl.Trainer(
        accelerator=cfg.train.accelerator,
        devices=cfg.train.devices,
        strategy=DDPStrategy(find_unused_parameters=True, gradient_as_bucket_view=True),
        callbacks=[
            ModelSummary(max_depth=-1),
            ModelCheckpoint(filename="model", save_last=True, save_top_k=0, dirpath=save_dir),
            LearningRateMonitor(logging_interval="step"),
        ],
        max_steps=cfg.train.max_steps,
        check_val_every_n_epoch=cfg.train.check_val_every_n_epoch,
        precision=cfg.train.precision,
        limit_train_batches=cfg.train.limit_train_batches,
        limit_val_batches=cfg.train.limit_val_batches,
        gradient_clip_val=cfg.train.gradient_clip_val,
        logger=build_logger(cfg, save_dir),
    )
    trainer.fit(model, datamodule, ckpt_path=resume_ckpt)


def train_ldm(cfg, cfg_ae, save_dir):
    """Train the VectorWorld LDM (EGR-DiT latent generative model)."""
    from vectorworld.models.ldm import VectorWorldLDM

    if not os.path.exists(cfg.dataset.latent_stats_path):
        cache_latent_stats(cfg)
    cfg = set_latent_stats(cfg)

    datamodule = instantiate(cfg.datamodule, dataset_cfg=cfg.dataset)
    resume_ckpt = getattr(cfg.train, "resume_ckpt_path", None)
    init_ckpt = getattr(cfg.train, "init_ckpt_path", None)

    if resume_ckpt:
        model = VectorWorldLDM.load_from_checkpoint(resume_ckpt, cfg=cfg, cfg_ae=cfg_ae, map_location="cpu")
    else:
        model = VectorWorldLDM(cfg=cfg, cfg_ae=cfg_ae)
        if init_ckpt:
            print(f"[train] Initializing LDM from: {init_ckpt}")
            ckpt = torch.load(init_ckpt, map_location="cpu", weights_only=False)
            safe_load_state_dict(
                model, ckpt.get("state_dict", ckpt),
                ignore_substrings=["autoencoder."],
                rename_substrings={"diff_model.": "gen_model."},
                strict_shapes=False,
            )

    save_top_k = getattr(cfg.train, "save_top_k", 0)
    ckpt_cb = ModelCheckpoint(
        monitor="val_loss" if save_top_k > 0 else None,
        save_last=True, save_top_k=save_top_k, dirpath=save_dir,
    )

    trainer = pl.Trainer(
        accelerator=cfg.train.accelerator,
        devices=cfg.train.devices,
        strategy=DDPStrategy(find_unused_parameters=True, gradient_as_bucket_view=True),
        callbacks=[ckpt_cb, LearningRateMonitor(logging_interval="step")],
        max_steps=cfg.train.max_steps,
        check_val_every_n_epoch=cfg.train.check_val_every_n_epoch,
        precision=cfg.train.precision,
        limit_train_batches=cfg.train.limit_train_batches,
        limit_val_batches=cfg.train.limit_val_batches,
        gradient_clip_val=cfg.train.gradient_clip_val,
        logger=build_logger(cfg, save_dir),
    )
    trainer.fit(model, datamodule, ckpt_path=resume_ckpt)


def train_deltasim(cfg, save_dir):
    """Train the DeltaSim behavior model."""
    from vectorworld.models.delta_sim import DeltaSim

    datamodule = instantiate(cfg.datamodule, dataset_cfg=cfg.dataset)
    model = DeltaSim(cfg)

    init_ckpt = getattr(cfg.train, "init_ckpt_path", None)
    resume_ckpt = getattr(cfg.train, "resume_ckpt_path", None)

    if init_ckpt and not resume_ckpt:
        print(f"[train] Initializing DeltaSim from: {init_ckpt}")
        ckpt = torch.load(init_ckpt, map_location="cpu", weights_only=False)
        safe_load_state_dict(model, ckpt.get("state_dict", ckpt), strict_shapes=False)

    trainer = pl.Trainer(
        accelerator=cfg.train.accelerator,
        devices=cfg.train.devices,
        strategy=DDPStrategy(find_unused_parameters=True, gradient_as_bucket_view=True),
        callbacks=[
            ModelSummary(max_depth=-1),
            ModelCheckpoint(monitor="val_loss", save_last=True, every_n_epochs=1,
                           save_top_k=15, dirpath=save_dir),
            LearningRateMonitor(logging_interval="step"),
        ],
        max_steps=cfg.train.max_steps,
        check_val_every_n_epoch=cfg.train.check_val_every_n_epoch,
        precision=cfg.train.precision,
        limit_train_batches=cfg.train.limit_train_batches,
        limit_val_batches=cfg.train.limit_val_batches,
        gradient_clip_val=cfg.train.gradient_clip_val,
        logger=build_logger(cfg, save_dir),
    )
    trainer.fit(model, datamodule, ckpt_path=resume_ckpt)


@hydra.main(version_base=None, config_path=os.path.join("..", "configs"), config_name="config")
def main(cfg):
    dataset_name = cfg.dataset_name.name

    if cfg.model_name == "vae":
        model_cfg = cfg.ae
        OmegaConf.set_struct(model_cfg, False)
        model_cfg.dataset_name = dataset_name
        OmegaConf.set_struct(model_cfg, True)

        pl.seed_everything(model_cfg.train.seed, workers=True)
        save_dir = os.path.join(model_cfg.train.save_dir, model_cfg.train.run_name)
        os.makedirs(save_dir, exist_ok=True)
        print(f"Training VectorWorld VAE on {dataset_name}")
        train_vae(model_cfg, save_dir)

    elif cfg.model_name == "ldm":
        cfg_ae = cfg.ae
        model_cfg = cfg.ldm
        for c in (model_cfg, cfg_ae):
            OmegaConf.set_struct(c, False)
            c.dataset_name = dataset_name
            OmegaConf.set_struct(c, True)

        pl.seed_everything(model_cfg.train.seed, workers=True)
        save_dir = os.path.join(model_cfg.train.save_dir, model_cfg.train.run_name)
        os.makedirs(save_dir, exist_ok=True)
        print(f"Training VectorWorld EGR-DiT LDM on {dataset_name}")
        train_ldm(model_cfg, cfg_ae, save_dir)

    elif cfg.model_name == "deltasim":
        model_cfg = cfg.deltasim
        OmegaConf.set_struct(model_cfg, False)
        model_cfg.dataset_name = dataset_name
        OmegaConf.set_struct(model_cfg, True)

        pl.seed_everything(model_cfg.train.seed, workers=True)
        save_dir = os.path.join(model_cfg.train.save_dir, model_cfg.train.run_name)
        os.makedirs(save_dir, exist_ok=True)
        print(f"Training DeltaSim on {dataset_name}")
        train_deltasim(model_cfg, save_dir)

    else:
        raise ValueError(f"Unknown model_name='{cfg.model_name}'. Use: vae | ldm | deltasim")


if __name__ == "__main__":
    main()
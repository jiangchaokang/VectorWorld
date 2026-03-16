import os
import pickle

import hydra
from omegaconf import OmegaConf

import torch
torch.set_float32_matmul_precision('medium')
import torch.nn.functional as F

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelSummary
from pytorch_lightning.strategies import DDPStrategy

from torch_geometric.loader import DataLoader
from tqdm import tqdm

from vectorworld.models.vae import VectorWorldVAE as ScenarioDreamerAutoEncoder
from vectorworld.models.ldm import VectorWorldLDM as ScenarioDreamerLDM


from vectorworld.data.waymo.vae_dataset import WaymoDatasetAutoEncoder
from vectorworld.data.nuplan.vae_dataset import NuplanDatasetAutoEncoder

from configs.config import CONFIG_PATH
from vectorworld.utils.train_helpers import set_latent_stats
import vectorworld.utils.sim_env_helpers as _sim_env_helpers
from vectorworld.utils.data_helpers import convert_batch_to_scenarios, unnormalize_motion_code


def _require_ckpt_path(path, model_name):
    assert path is not None and os.path.exists(path), (
        f"[ERROR] Please provide a valid checkpoint path via `{model_name}.eval.ckpt_path=/abs/path/to.ckpt`"
    )


def generate_simulation_environments(cfg, cfg_ae, save_dir=None):
    """ Generate simulation environments using the Scenario Dreamer Latent Diffusion Model.
    
    This involves 1 step of initial scene generation followed by multiple steps of
    inpainting to extend the scenario until the desired route length is reached.
    Additional rule-based heuristics are applied to ensure scenario validity.
    """
    cfg = set_latent_stats(cfg)
    ckpt_path = cfg.eval.ckpt_path
    _require_ckpt_path(ckpt_path, "ldm")

    print(f"Loading LDM ckpt: {ckpt_path}")
    model = ScenarioDreamerLDM.load_from_checkpoint(ckpt_path, cfg=cfg, cfg_ae=cfg_ae).to('cuda')
    _sim_env_helpers.generate_simulation_environments(model, cfg, save_dir)


def eval_ldm(cfg, cfg_ae, save_dir=None):
    
    # Other modes: intical_scene /lane_conventioned / incorporating / representation_environments
    cfg = set_latent_stats(cfg)
    ckpt_path = cfg.eval.ckpt_path
    _require_ckpt_path(ckpt_path, "ldm")
    
    # generate samples
    model = ScenarioDreamerLDM.load_from_checkpoint(ckpt_path, cfg=cfg, cfg_ae=cfg_ae).to('cuda')
    model.generate(
        mode = cfg.eval.mode,  # initial_scene, lane_conditioned, inpainting
        num_samples = cfg.eval.num_samples,
        batch_size = cfg.eval.batch_size,
        cache_samples = cfg.eval.cache_samples,
        visualize = cfg.eval.visualize,
        conditioning_path = cfg.eval.conditioning_path,
        cache_dir = os.path.join(save_dir, f'{cfg.eval.mode}_samples'),
        viz_dir = cfg.eval.viz_dir,
        save_wandb = False,
        return_samples=False,
    )


def generate_ae_reconstruction_samples(cfg, save_dir=None):
    
    ckpt_path = getattr(cfg.eval, "ckpt_path", None)
    _require_ckpt_path(ckpt_path, "autoencoder")

    device = "cuda" if torch.cuda.is_available() and getattr(cfg.eval, "accelerator", "gpu") != "cpu" else "cpu"

    print(f"[AE-Recon] Loading AutoEncoder ckpt: {ckpt_path}")
    model = ScenarioDreamerAutoEncoder.load_from_checkpoint(
        ckpt_path, cfg=cfg, map_location="cpu"
    ).to(device)
    model.eval()

    dataset_name = cfg.dataset_name
    split_name = getattr(cfg.eval, "split_name", "val")

    # ------------------------------------------------------------------ #
    # 1) Build Foundation
    # ------------------------------------------------------------------ #
    if dataset_name == "waymo":
        dataset = WaymoDatasetAutoEncoder(cfg.dataset, split_name=split_name, mode="eval")
    else:
        dataset = NuplanDatasetAutoEncoder(cfg.dataset, split_name=split_name, mode="eval")

    # ------------------------------------------------------------------ #
    # 1.1) Optional: list of files aligned according to the event_set#
    # ------------------------------------------------------------------ #
    eval_set_path = getattr(cfg.eval, "eval_set", None)
    use_eval_set = eval_set_path is not None and os.path.exists(eval_set_path)

    if use_eval_set:
        with open(eval_set_path, "rb") as f:
            meta = pickle.load(f)
        gt_files = meta.get("files", [])

        if not gt_files:
            print(
                f"[AE-Recon] WARNING: eval_set='{eval_set_path}' has no 'files' field or is empty, "
                f"falling back to reconstructing the entire split='{split_name}'."
            )
            use_eval_set = False
        else:
            # Create with Basenamedataset.filesIndexes
            ds_map = {os.path.basename(p): p for p in dataset.files}
            selected_files = []
            missing = []
            for rel in gt_files:
                bn = os.path.basename(rel)
                path = ds_map.get(bn, None)
                if path is None:
                    missing.append(bn)
                else:
                    selected_files.append(path)

            if missing:
                print(
                    f"[AE-Recon] WARNING: {len(missing)} files in eval_set "
                    f"do not exist in the AE dataset for split='{split_name}', examples:\n"
                    + "\n".join(f"  - {m}" for m in missing[:10])
                )

            if selected_files:
                dataset.files = selected_files
                if hasattr(dataset, "dset_len"):
                    dataset.dset_len = len(dataset.files)
                print(
                    f"[AE-Recon] Aligned with eval_set, reconstructing {len(dataset.files)} scenes in total "
                    f"(split='{split_name}', eval_set='{eval_set_path}')."
                )
            else:
                print(
                    f"[AE-Recon] WARNING: eval_set has no filename overlap with current split='{split_name}', "
                    "will fall back to reconstructing the entire split."
                )
                use_eval_set = False

    # ------------------------------------------------------------------ #
    # 1.2) Optional: Random subsampling is allowed only if ** has not been used eval_set**
    # ------------------------------------------------------------------ #
    max_samples = getattr(cfg.eval, "num_samples", None)
    if (
        (max_samples is not None)
        and (max_samples > 0)
        and (max_samples < len(dataset))
    ):
        if use_eval_set:
            print(
                """[AE-Recon] NOTE: detectedcfg.eval.eval_setThe AE reconstruction phase will be ignored to ensure full alignment with the event_setcfg.eval.num_samplesI don't know.
          If you want to evaluate the subset, pass when generating the eval_set`waymo_eval_set.num_samples`Control."""
            )
        else:
            import random

            rng = random.Random(cfg.eval.seed)
            idxs = list(range(len(dataset)))
            rng.shuffle(idxs)
            idxs = sorted(idxs[:max_samples])
            dataset.files = [dataset.files[i] for i in idxs]
            if hasattr(dataset, "dset_len"):
                dataset.dset_len = len(dataset.files)
            print(
                f"[AE-Recon] Sub-sampling dataset to {len(dataset.files)} scenes "
                f"(from original {len(idxs)}) for AE reconstruction."
            )

    batch_size = getattr(cfg.eval, "batch_size", cfg.datamodule.val_batch_size)
    num_workers = cfg.datamodule.num_workers
    pin_memory = cfg.datamodule.pin_memory

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )

    # ------------------------------------------------------------------ #
    # 2) Output directory                                                #
    # ------------------------------------------------------------------ #
    if save_dir is None:
        save_dir = os.path.join(cfg.eval.save_dir, cfg.eval.run_name)
    samples_dir = os.path.join(save_dir, "initial_scene_samples")
    os.makedirs(samples_dir, exist_ok=True)

    # Motion Inverted Scale (if position code is enabled)
    motion_cfg = getattr(cfg.dataset, "motion", None)
    motion_max_disp = None
    if motion_cfg is not None and getattr(motion_cfg, "enabled", False):
        motion_max_disp = float(getattr(motion_cfg, "max_displacement", 12.0))

    print(
        f"[AE-Recon] Reconstructing {len(dataset)} scenes on split='{split_name}' "
        f"→ saving to: {samples_dir}"
    )

    total_saved = 0
    for batch_idx, data in enumerate(tqdm(dataloader)):
        data = data.to(device)

        with torch.no_grad():
            (
                agent_samples,
                lane_samples,
                agent_types,
                lane_types,
                lane_conn_samples,
                _lane_cond_dis,
                agent_motion_pred,
            ) = model(data)

        # 2.1 Rewrite AE reconstruction results back to data (andLDM.forwardand in a consistent format)
        data["agent"].x = agent_samples
        data["lane"].x = lane_samples
        data["agent"].type = F.one_hot(
            agent_types, num_classes=cfg.dataset.num_agent_types
        )

        if dataset_name == "nuplan" and (lane_types is not None):
            data["lane"].type = F.one_hot(
                lane_types, num_classes=cfg.dataset.num_lane_types
            )

        data["lane", "to", "lane"].type = lane_conn_samples

        if motion_max_disp is not None and agent_motion_pred is not None:
            # AE.forwardReturn is the normalization [.]-1Movement code
            motion_phys = unnormalize_motion_code(
                agent_motion_pred, motion_max_displacement=motion_max_disp
            )
            if isinstance(motion_phys, torch.Tensor):
                data["agent"].motion = motion_phys
            else:
                data["agent"].motion = torch.from_numpy(motion_phys).to(
                    agent_samples.device
                )

        # 2.2 Call convert_batch_to_scenarios for standardized scene dict (not falling on board)
        batch_scenarios = convert_batch_to_scenarios(
            data,
            batch_size=data.batch_size,
            batch_idx=batch_idx,
            cache_dir=None,
            conditioning_filenames=None,
            cache_samples=False,
            cache_lane_types=(dataset_name == "nuplan"),
            mode="initial_scene",
        )

        # data.idx:♪ Every scene ♪dataset.filesIndexes in
        idx_in_dataset = data.idx.detach().cpu().numpy().astype(int)
        assert len(idx_in_dataset) == len(
            batch_scenarios
        ), "Mismatch between batch scenes and converted scenarios."

        # Python3.7+ Dictionary stays inserted in order consistent with the chronology of convert_batch_to_scenarios
        scenario_keys = list(batch_scenarios.keys())

        for local_i, key in enumerate(scenario_keys):
            ds_idx = int(idx_in_dataset[local_i])
            assert 0 <= ds_idx < len(dataset.files)
            original_file_path = dataset.files[ds_idx]
            filename = os.path.basename(original_file_path)

            out_path = os.path.join(samples_dir, filename)
            with open(out_path, "wb") as f:
                pickle.dump(batch_scenarios[key], f)

            total_saved += 1

    print(f"[AE-Recon] Saved {total_saved} reconstructed scenes to '{samples_dir}'")


def eval_autoencoder(cfg, save_dir=None):
    model = ScenarioDreamerAutoEncoder(cfg)
    model_summary = ModelSummary(max_depth=-1)
    
    # load checkpoint
    ckpt_path = getattr(cfg.eval, "ckpt_path", None)
    assert ckpt_path is not None and os.path.exists(ckpt_path), \
        f"No valid checkpoint found. Please set `ae.eval.ckpt_path` explicitly or ensure last*.ckpt exists in {save_dir}"

    # Equipment andDDPPolicy Configuration
    accelerator = getattr(cfg.eval, "accelerator", "gpu")
    devices = int(getattr(cfg.eval, "devices", 1))

    if devices > 1:
        # Find_unused_parameters not required under pure reasoning/cache scene, reducing costs
        strategy = DDPStrategy(find_unused_parameters=False, gradient_as_bucket_view=True)
    else:
        strategy = "auto"

    precision = getattr(cfg.eval, "precision", "32-true")

    tester = pl.Trainer(
        accelerator=accelerator,
        devices=devices,
        strategy=strategy,
        callbacks=[model_summary],
        precision=precision,
        logger=False,                # Default close logger for evaluation phase, less expense
        enable_checkpointing=False,  # Checkpoint is not stored in the event phase
        enable_progress_bar=True,
    )
    
    tester.test(model, ckpt_path=ckpt_path)


@hydra.main(version_base=None, config_path=CONFIG_PATH, config_name="config")
def main(cfg):
    # need to track whether we are evaluating a nuplan or waymo model as 
    # nuplan predicts lane types (lane/green light/red light) and waymo does not
    dataset_name = cfg.dataset_name.name
    if cfg.model_name == "vae":
        # AutoEncoder Configuration
        model_name = cfg.model_name
        cfg = cfg.ae
        # not the cleanest solution, but need to track dataset name
        OmegaConf.set_struct(cfg, False)   # unlock to allow setting dataset name
        cfg.dataset_name = dataset_name
        OmegaConf.set_struct(cfg, True)    # relock

    else:
        # LDM / CtRL-Sim
        model_name = cfg.model_name
        cfg_ae = cfg.ae
        cfg = cfg.ldm
        OmegaConf.set_struct(cfg, False)   # unlock to allow setting dataset name
        OmegaConf.set_struct(cfg_ae, False)
        cfg.dataset_name = dataset_name
        cfg_ae.dataset_name = dataset_name
        OmegaConf.set_struct(cfg, True)    # relock
        OmegaConf.set_struct(cfg_ae, True)
    
    pl.seed_everything(cfg.eval.seed, workers=True)

    # checkpoints loaded from here
    save_dir = os.path.join(cfg.eval.save_dir, cfg.eval.run_name)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)

    print(f"Evaluating Scenario Dreamer {model_name} trained on {dataset_name} dataset.")
    if model_name == "vae":
        eval_mode = getattr(cfg.eval, "mode", None)

        if eval_mode in ("initial_scene", "reconstruct", "reconstruction", "ae_recon"):
            # Rebuild and drop the scene as pkl with AE v eval split
            generate_ae_reconstruction_samples(cfg, save_dir)

        else:
            # Back to Lightingtest()(loss / visualization / item caching)
            eval_autoencoder(cfg, save_dir)

    elif model_name == 'ldm':
        if cfg.eval.mode == 'simulation_environments':
            # Generate offline simulation environment (fall)
            generate_simulation_environments(cfg, cfg_ae, save_dir)
        else:
            # initial_scene / lane_conditioned / inpainting / metrics
            eval_ldm(cfg, cfg_ae, save_dir) 

    else:
        raise ValueError(f"Unsupported model_name='{model_name}'. Use 'vae' or 'ldm'.")


if __name__ == '__main__':
    main()
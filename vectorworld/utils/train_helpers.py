import torch.nn as nn
import torch
import math
import os
from torch_geometric.loader import DataLoader
from tqdm import tqdm
import pickle
import numpy as np

from vectorworld.data.waymo.ldm_dataset import WaymoDatasetLDM
from vectorworld.data.nuplan.ldm_dataset import NuplanDatasetLDM
from vectorworld.utils.data_helpers import sample_latents
from torch.nn import Transformer


def create_lambda_lr_cosine(cfg):
    """cosine learning rate schedule with warmup"""
    return lambda current_step: (
        current_step / cfg.train['warmup_steps'] if current_step < cfg.train['warmup_steps']
        else max(
            0.0,
            0.5 * (1 + math.cos(math.pi * (current_step - cfg.train['warmup_steps']) / (cfg.train['max_steps'] - cfg.train['warmup_steps'])))
        )
    )


def create_lambda_lr_linear(cfg):
    """linear learning rate schedule with warmup"""
    return lambda current_step: (
        current_step / cfg.train['warmup_steps'] if current_step < cfg.train['warmup_steps']
        else max(
            0.0,
            (cfg.train['max_steps'] - current_step) / (cfg.train['max_steps'] - cfg.train['warmup_steps'])
        )
    )


def create_lambda_lr_constant(cfg):
    """constant learning rate schedule with warmup"""
    return lambda current_step: (
        current_step / cfg.train['warmup_steps'] if current_step < cfg.train['warmup_steps']
        else 1
    )


def weight_init(m):
    """Initialize weights of PyTorch modules. Inspired by QCNET: https://github.com/ZikangZhou/QCNet"""
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
        fan_in = m.in_channels / m.groups
        fan_out = m.out_channels / m.groups
        bound = (6.0 / (fan_in + fan_out)) ** 0.5
        nn.init.uniform_(m.weight, -bound, bound)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Embedding):
        nn.init.normal_(m.weight, mean=0.0, std=0.02)
    elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
        nn.init.ones_(m.weight)
        nn.init.zeros_(m.bias)
    elif isinstance(m, nn.LayerNorm):
        nn.init.ones_(m.weight)
        nn.init.zeros_(m.bias)
    elif isinstance(m, nn.MultiheadAttention):
        if m.in_proj_weight is not None:
            fan_in = m.embed_dim
            fan_out = m.embed_dim
            bound = (6.0 / (fan_in + fan_out)) ** 0.5
            nn.init.uniform_(m.in_proj_weight, -bound, bound)
        else:
            nn.init.xavier_uniform_(m.q_proj_weight)
            nn.init.xavier_uniform_(m.k_proj_weight)
            nn.init.xavier_uniform_(m.v_proj_weight)
        if m.in_proj_bias is not None:
            nn.init.zeros_(m.in_proj_bias)
        nn.init.xavier_uniform_(m.out_proj.weight)
        if m.out_proj.bias is not None:
            nn.init.zeros_(m.out_proj.bias)
        if m.bias_k is not None:
            nn.init.normal_(m.bias_k, mean=0.0, std=0.02)
        if m.bias_v is not None:
            nn.init.normal_(m.bias_v, mean=0.0, std=0.02)
    elif isinstance(m, (nn.LSTM, nn.LSTMCell)):
        for name, param in m.named_parameters():
            if 'weight_ih' in name:
                for ih in param.chunk(4, 0):
                    nn.init.xavier_uniform_(ih)
            elif 'weight_hh' in name:
                for hh in param.chunk(4, 0):
                    nn.init.orthogonal_(hh)
            elif 'weight_hr' in name:
                nn.init.xavier_uniform_(param)
            elif 'bias_ih' in name:
                nn.init.zeros_(param)
            elif 'bias_hh' in name:
                nn.init.zeros_(param)
                nn.init.ones_(param.chunk(4, 0)[1])
    elif isinstance(m, (nn.GRU, nn.GRUCell)):
        for name, param in m.named_parameters():
            if 'weight_ih' in name:
                for ih in param.chunk(3, 0):
                    nn.init.xavier_uniform_(ih)
            elif 'weight_hh' in name:
                for hh in param.chunk(3, 0):
                    nn.init.orthogonal_(hh)
            elif 'bias_ih' in name:
                nn.init.zeros_(param)
            elif 'bias_hh' in name:
                nn.init.zeros_(param)


def cache_latent_stats(cfg):
    """Cache the latent statistics (mean/std) for agent and lane latents computed from the training dataset.

    New edition achieved:
    ----------
    - Use **Dividmean/std** (shape = [latent_dim]) rather than a single mark;
    - Here's the thing.`normalize_latents` / `unnormalize_latents`You can separate each dimension.
      A better distribution of directional opposites following the introduction of motion + gating;
    - Calculating a "simmary" (recalculating dimensions) for log printing.
    """
    if cfg.dataset_name == 'waymo':
        dset = WaymoDatasetLDM(cfg.dataset, split_name='train')
    else:
        dset = NuplanDatasetLDM(cfg.dataset, split_name='train')
    print("Caching latent stats (mean/std of agent/lane latents). Size of dataset: ", len(dset))

    os.makedirs(cfg.dataset.latent_stats_dir, exist_ok=True)
    # Place: The real value at the back will be writtenlatent_stats.pklLee.
    cfg.dataset.agent_latents_mean = 0.0
    cfg.dataset.agent_latents_std = 1.0
    cfg.dataset.lane_latents_mean = 0.0
    cfg.dataset.lane_latents_std = 1.0

    dloader = DataLoader(
        dset,
        batch_size=cfg.train.batch_size_compute_stats,
        shuffle=True,
        num_workers=cfg.datamodule.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    agent_latents_all = []
    lane_latents_all = []
    for i, d in enumerate(tqdm(dloader)):
        if i == cfg.train.num_batches_compute_stats:
            break

        # Here.sample_latents(return normalize=False)Directly frommu/log_varAfter sampling,
        # Don't do it while, withLDMThe training phase is consistent.
        agent_latents, lane_latents = sample_latents(
            d,
            cfg.dataset.agent_latents_mean,
            cfg.dataset.agent_latents_std,
            cfg.dataset.lane_latents_mean,
            cfg.dataset.lane_latents_std,
            normalize=False,
        )
        agent_latents_all.append(agent_latents)
        lane_latents_all.append(lane_latents)

    agent_latents_all = torch.cat(agent_latents_all, dim=0)  # (Na_total, Da)
    lane_latents_all = torch.cat(lane_latents_all, dim=0)    # (Nl_total, Dl)

    # ---------------------------------------------------------------------------------
    # shape: (Da,), (Dl,)
    agent_mean_vec = agent_latents_all.mean(dim=0)
    agent_std_vec = agent_latents_all.std(dim=0)
    lane_mean_vec = lane_latents_all.mean(dim=0)
    lane_std_vec = lane_latents_all.std(dim=0)

    # ---- metric submary (comparable with previous logs) --
    agent_mean_scalar = agent_mean_vec.mean().item()
    agent_std_scalar = agent_std_vec.mean().item()
    lane_mean_scalar = lane_mean_vec.mean().item()
    lane_std_scalar = lane_std_vec.mean().item()

    print(
        f"Agent Latents Mean (scalar avg over dims): {agent_mean_scalar:.4f}, "
        f"Std: {agent_std_scalar:.4f}"
    )
    print(
        f"Lane  Latents Mean (scalar avg over dims): {lane_mean_scalar:.4f}, "
        f"Std: {lane_std_scalar:.4f}"
    )

    # -----------------------------------------------------------------------------------------
    # In order to be compatible, this is a Python list.
    # The old version can be read by the same standard floatnp.asarrayDeal.
    stats = {
        'agent_latents_mean': agent_mean_vec.cpu().numpy().tolist(),
        'agent_latents_std': agent_std_vec.cpu().numpy().tolist(),
        'lane_latents_mean': lane_mean_vec.cpu().numpy().tolist(),
        'lane_latents_std': lane_std_vec.cpu().numpy().tolist(),
        # Attached metric sumary, to facilitate follow-up analysis (optional, without prejudice to training)
        'agent_latents_mean_scalar': agent_mean_scalar,
        'agent_latents_std_scalar': agent_std_scalar,
        'lane_latents_mean_scalar': lane_mean_scalar,
        'lane_latents_std_scalar': lane_std_scalar,
    }
    with open(cfg.dataset.latent_stats_path, 'wb') as f:
        pickle.dump(stats, f)


def set_latent_stats(cfg):
    """Set the latent statistics (mean/std) for agent and lane latents from a cached file.

    New edition of the act:
    ----------
    - Compatibility of old metric statistics and new dimensional statistics:
        * If pkl is float,np.asarraythen obtain a set of metrics;
        * If it's a list,np.asarray(D,) arrays;
    - cfg.dataset.* Retain original Python objects (float or list),
      In order to...`normalize_latents`It was broadcast automatically by PyTorch to the correct address;
    - When printing a log, do one more for each vector, meaning, get a submary mark.
    """
    # Load the latent stats from the cached file
    with open(cfg.dataset.latent_stats_path, 'rb') as f:
        stats = pickle.load(f)

    agent_mean = stats['agent_latents_mean']
    agent_std = stats['agent_latents_std']
    lane_mean = stats['lane_latents_mean']
    lane_std = stats['lane_latents_std']

    # Just hang the original object on cfg (float or list)
    cfg.dataset.agent_latents_mean = agent_mean
    cfg.dataset.agent_latents_std = agent_std
    cfg.dataset.lane_latents_mean = lane_mean
    cfg.dataset.lane_latents_std = lane_std

    # Make a massary with numbery easy to print
    agent_mean_arr = np.asarray(agent_mean, dtype=np.float32)
    agent_std_arr = np.asarray(agent_std, dtype=np.float32)
    lane_mean_arr = np.asarray(lane_mean, dtype=np.float32)
    lane_std_arr = np.asarray(lane_std, dtype=np.float32)

    agent_mean_scalar = float(agent_mean_arr.mean())
    agent_std_scalar = float(agent_std_arr.mean())
    lane_mean_scalar = float(lane_mean_arr.mean())
    lane_std_scalar = float(lane_std_arr.mean())

    print("Set latent stats from cached file:")
    print(
        f"Agent Latents Mean (scalar avg over dims): {agent_mean_scalar:.4f}, "
        f"Std: {agent_std_scalar:.4f}"
    )
    print(
        f"Lane  Latents Mean (scalar avg over dims): {lane_mean_scalar:.4f}, "
        f"Std: {lane_std_scalar:.4f}"
    )

    return cfg


def get_causal_mask(cfg, num_timesteps, num_types):
    """ Get the causal mask for the transformer decoder in CtRL-Sim model."""
    num_agents = cfg.dataset.max_num_agents
    num_steps = num_timesteps
    state_index = 0
    num_tokens = num_agents * num_steps * num_types

    mask = Transformer.generate_square_subsequent_mask(num_tokens)
    multi_agent_mask = torch.Tensor(mask.shape).fill_(0)
    offset = 0
    index = 0
    for index in range(len(multi_agent_mask)):
        mask_out = torch.Tensor(num_agents * num_types).fill_(float('-inf'))
        agent_id = (index // num_types) % num_agents
        mask_out[agent_id * num_types:(agent_id + 1) * (num_types)] = 0
        multi_agent_mask[index, offset:offset + (num_agents * num_types)] = mask_out

        if (index + 1) % (num_agents * num_types) == 0:
            offset += num_agents * num_types

    mask = torch.minimum(mask, multi_agent_mask)

    # current state of all agents is visible
    for index_i in range(len(mask)):
        timestep_idx = index_i // (num_types * num_agents)
        for index_j in range(len(mask)):
            if (index_j < (timestep_idx + 1) * (num_agents * num_types)
                    and index_j % num_types == state_index):
                mask[index_i, index_j] = 0.

    return mask
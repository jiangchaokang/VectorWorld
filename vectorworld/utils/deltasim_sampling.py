from __future__ import annotations

from typing import Optional, Sequence, Union, List

import torch


def _as_tilt_list(tilt: Union[float, Sequence[float]], C: int) -> List[float]:
    if isinstance(tilt, (list, tuple)):
        if len(tilt) != C:
            raise ValueError(f"tilt list length must be C={C}, got {len(tilt)}")
        return [float(x) for x in tilt]
    return [float(tilt) for _ in range(C)]


@torch.no_grad()
def sample_from_logits(
    logits: torch.Tensor,
    *,
    temperature: float = 1.0,
    nucleus_p: Optional[float] = None,
) -> torch.Tensor:
    """
    logits: [..., V]
    returns: [...] int64 sampled ids
    """
    if temperature <= 0:
        raise ValueError(f"temperature must be > 0, got {temperature}")

    orig_shape = logits.shape[:-1]
    V = logits.shape[-1]
    x = logits.float().reshape(-1, V) / float(temperature)
    probs = torch.softmax(x, dim=-1)

    if nucleus_p is not None and nucleus_p < 1.0:
        # nucleus filtering
        sorted_probs, sorted_idx = torch.sort(probs, dim=-1, descending=True)
        cum = torch.cumsum(sorted_probs, dim=-1)
        keep = cum <= float(nucleus_p)
        keep[:, 0] = True  # always keep top-1

        filtered = torch.zeros_like(probs)
        filtered.scatter_(dim=-1, index=sorted_idx, src=sorted_probs * keep)

        denom = filtered.sum(dim=-1, keepdim=True).clamp(min=1e-12)
        probs = filtered / denom

    tokens = torch.multinomial(probs, num_samples=1).squeeze(-1)
    return tokens.reshape(orig_shape).long()


@torch.no_grad()
def sample_rtg_tokens(
    rtg_logits: torch.Tensor,
    *,
    K: int,
    C: int,
    tilt: Union[float, Sequence[float]] = 0.0,
) -> torch.Tensor:
    """
    rtg_logits can be:
      - [..., K*C]
      - [..., K, C]
    returns: [..., C] int64
    """
    tilt_list = _as_tilt_list(tilt, C)

    if rtg_logits.dim() >= 2 and rtg_logits.shape[-1] == K * C:
        x = rtg_logits.view(*rtg_logits.shape[:-1], K, C)
    elif rtg_logits.dim() >= 3 and rtg_logits.shape[-2] == K and rtg_logits.shape[-1] == C:
        x = rtg_logits
    else:
        raise ValueError(f"Invalid rtg_logits shape {tuple(rtg_logits.shape)} for K={K}, C={C}")

    device = x.device
    dtype = x.dtype
    tilt_axis = torch.linspace(0.0, 1.0, steps=K, device=device, dtype=dtype)  # [K]

    outs = []
    for c in range(C):
        lc = x[..., :, c] + tilt_axis * float(tilt_list[c])
        tok = sample_from_logits(lc, temperature=1.0, nucleus_p=None)
        outs.append(tok)

    return torch.stack(outs, dim=-1).long()
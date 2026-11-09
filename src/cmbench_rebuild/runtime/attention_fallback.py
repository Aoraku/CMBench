"""PyTorch SDPA stand-in for ``wan.modules.attention.flash_attention``.

flash-attn 2.x ships no kernels for Blackwell (sm_100/sm_103) GPUs such as the
B300.  When neither FlashAttention 2 nor 3 can be imported, the vendored LingBot
modules would assert inside cross-attention.  This installs an exact-semantics
replacement (variable key lengths become a boolean mask) without editing the
pinned vendored source, so its subtree hash stays valid.
"""
from __future__ import annotations

from typing import Any


def sdpa_flash_attention(q: Any, k: Any, v: Any, q_lens: Any = None, k_lens: Any = None, dropout_p: float = 0.0,
                         softmax_scale: float | None = None, q_scale: float | None = None, causal: bool = False,
                         window_size: tuple[int, int] = (-1, -1), deterministic: bool = False, dtype: Any = None,
                         version: Any = None) -> Any:
    import torch

    if tuple(window_size) != (-1, -1):
        raise NotImplementedError("SDPA fallback does not implement sliding-window attention")
    out_dtype = q.dtype
    # Like flash_attention, cast non-half inputs to the requested half dtype (bf16 by default).
    if dtype is None or dtype in (torch.float16, torch.bfloat16):
        half = dtype or torch.bfloat16
        q, k, v = (x if x.dtype in (torch.float16, torch.bfloat16) else x.to(half) for x in (q, k, v))
    if q_scale is not None:
        q = q * q_scale
    batch, lq, lk = int(q.shape[0]), int(q.shape[1]), int(k.shape[1])
    mask = None
    if k_lens is not None:
        keys = torch.arange(lk, device=q.device)[None, :] < k_lens.to(q.device)[:, None]
        mask = keys[:, None, None, :].expand(batch, 1, lq, lk)
    if q.shape[2] != k.shape[2]:
        repeat = int(q.shape[2]) // int(k.shape[2])
        k, v = k.repeat_interleave(repeat, dim=2), v.repeat_interleave(repeat, dim=2)
    out = torch.nn.functional.scaled_dot_product_attention(
        q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), attn_mask=mask,
        dropout_p=float(dropout_p), is_causal=bool(causal) and mask is None, scale=softmax_scale,
    ).transpose(1, 2)
    if q_lens is not None:
        valid = torch.arange(lq, device=q.device)[None, :] < q_lens.to(q.device)[:, None]
        out = out * valid[:, :, None, None].to(out.dtype)
    return out.to(out_dtype)


def install_if_needed() -> bool:
    """Patch every vendored module that imported ``flash_attention`` by name."""
    import importlib

    attention = importlib.import_module("wan.modules.attention")
    if attention.FLASH_ATTN_2_AVAILABLE or attention.FLASH_ATTN_3_AVAILABLE:
        return False
    for name in ("wan.modules.attention", "wan.modules", "wan.modules.model", "wan.modules.model_fast",
                 "wan.modules.model_causal", "wan.distributed.ulysses", "wan.distributed.sequence_parallel"):
        try:
            module = importlib.import_module(name)
        except Exception:
            continue
        if hasattr(module, "flash_attention"):
            module.flash_attention = sdpa_flash_attention
    return True

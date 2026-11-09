"""Torch kernels shared by the rebuttal baselines.

Every function here is pure tensor math with no cache bookkeeping, so it can be
unit-tested on CPU.  Shapes follow the runtime convention: keys and queries are
``[tokens, heads, head_dim]`` unless stated otherwise.
"""
from __future__ import annotations

import math
from typing import Any


def sampled_query_rows(sequence_tokens: int, samples: int, *, device: Any) -> Any:
    """Deterministic, evenly spaced query rows (SnapKV-style observation set)."""
    import torch

    count = max(1, min(int(samples), int(sequence_tokens)))
    return torch.linspace(0, int(sequence_tokens) - 1, count, device=device).round().long().unique()


def attention_mass_over_keys(
    queries: Any,
    keys: Any,
    *,
    target_stop: int,
    block_tokens: int = 8192,
) -> Any:
    """Softmax attention mass that ``queries`` place on ``keys[:target_stop]``.

    ``queries`` is ``[groups, heads, dim]`` (one softmax per group and head) and
    ``keys`` is ``[tokens, heads, dim]``.  The softmax denominator runs over all
    keys, as in the real attention call; the returned vector is the mass summed
    over groups and heads for the first ``target_stop`` keys.  Two passes with a
    running log-sum-exp keep peak memory at one key block.
    """
    import torch

    groups, heads, dim = (int(value) for value in queries.shape)
    total = int(keys.shape[0])
    stop = max(0, min(int(target_stop), total))
    if stop == 0 or total == 0:
        return torch.zeros(stop, dtype=torch.float32, device=queries.device)
    q = queries.float().permute(1, 0, 2).contiguous()  # [H, G, D]
    scale = 1.0 / math.sqrt(dim)
    running_max = torch.full((heads, groups), -math.inf, device=queries.device)
    running_sum = torch.zeros((heads, groups), device=queries.device)
    for start in range(0, total, int(block_tokens)):
        block = keys[start:min(total, start + int(block_tokens))].to(queries.device).float().permute(1, 2, 0)
        logits = torch.bmm(q, block) * scale  # [H, G, B]
        block_max = logits.amax(dim=-1)
        new_max = torch.maximum(running_max, block_max)
        running_sum = running_sum * torch.exp(running_max - new_max) + torch.exp(logits - new_max[..., None]).sum(dim=-1)
        running_max = new_max
    log_normalizer = running_max + torch.log(running_sum)
    mass = torch.zeros(stop, dtype=torch.float32, device=queries.device)
    for start in range(0, stop, int(block_tokens)):
        end = min(stop, start + int(block_tokens))
        block = keys[start:end].to(queries.device).float().permute(1, 2, 0)
        probabilities = torch.exp(torch.bmm(q, block) * scale - log_normalizer[..., None])
        mass[start:end] = probabilities.sum(dim=(0, 1))
    return mass


def spatial_pool_chunk_scores(scores: Any, *, frames: int, height: int, width: int, kernel: int) -> Any:
    """SnapKV's score pooling, applied on each latent frame's 2-D token grid."""
    import torch.nn.functional as functional

    if int(kernel) <= 1:
        return scores
    grid = scores.float().reshape(int(frames), 1, int(height), int(width))
    pooled = functional.avg_pool2d(grid, kernel_size=int(kernel), stride=1, padding=int(kernel) // 2, count_include_pad=False)
    return pooled.reshape(-1)


def top_fraction_mask(scores: Any, *, keep_ratio: float) -> Any:
    """Keep the ``keep_ratio`` highest scores (ties broken by earlier index)."""
    import torch

    count = int(scores.numel())
    keep = max(0, min(count, int(round(float(keep_ratio) * count))))
    mask = torch.zeros(count, dtype=torch.bool, device=scores.device)
    if keep:
        order = torch.argsort(scores.float(), descending=True, stable=True)
        mask[order[:keep]] = True
    return mask


def worldkv_anchor_scores(chunk_keys: Any, *, frame_tokens: int) -> Any:
    """WorldKV Eq. 5: mean cosine of each non-anchor key to all anchor keys.

    ``chunk_keys`` is ``[frames * frame_tokens, heads, dim]``; the first frame is
    the anchor.  Cosines are averaged over heads, so the result is one score per
    non-anchor token (higher means more redundant with the anchor).
    """
    import torch

    keys = torch.nn.functional.normalize(chunk_keys.float(), dim=-1)
    anchor = keys[: int(frame_tokens)]
    rest = keys[int(frame_tokens):]
    # mean_i cos(k_j, a_i) = k_j . mean_i(a_i) for unit vectors.
    anchor_mean = anchor.mean(dim=0)  # [H, D]
    return torch.einsum("nhd,hd->nh", rest, anchor_mean).mean(dim=1)


def riesz_contribution(squared_distance: Any, *, scale: float, power: float, eps: float) -> Any:
    return 1.0 / (float(eps) + squared_distance.float() / (float(scale) ** 2)).pow(float(power))


def head_mean_riesz(
    left: Any,
    right: Any,
    *,
    scale: float,
    power: float,
    eps: float,
) -> Any:
    """Head-averaged Riesz kernel between ``[A, H, D]`` and ``[B, H, D]`` keys -> ``[A, B]``."""
    import torch

    a = left.float().permute(1, 0, 2)  # [H, A, D]
    b = right.float().permute(1, 0, 2)  # [H, B, D]
    squared = (a.square().sum(-1)[..., :, None] + b.square().sum(-1)[..., None, :] - 2.0 * torch.bmm(a, b.transpose(1, 2))).clamp_min_(0.0)
    return riesz_contribution(squared, scale=scale, power=power, eps=eps).mean(dim=0)


def _blocked_head_mean_riesz(left: Any, right: Any, *, block: int, **kernel: Any):
    for start in range(0, int(right.shape[0]), int(block)):
        stop = min(int(right.shape[0]), start + int(block))
        yield start, stop, head_mean_riesz(left, right[start:stop], **kernel)


def densitykv_admit(
    candidate_keys: Any,
    bank_keys: Any,
    bank_density: Any,
    bank_baseline: Any,
    *,
    capacity: int,
    scale: float = 8.0,
    power: float = 2.0,
    eps: float = 1.0,
    growth_limit: float = 2.0,
    baseline_floor: float = 1.0e-6,
    work_chunk: int = 512,
) -> dict[str, Any]:
    """One DensityKV bank update with head-averaged density (single shared group).

    Mirrors ``DensityLimitedKVBank.update`` from the public DensityKV release:
    candidates are ordered by how sharply they raise an existing entry's density
    relative to its baseline, the longest prefix that keeps every surviving entry
    under ``growth_limit`` is admitted, and capacity overflow evicts growth
    violators first and then the densest entries.

    Returns indices into ``candidate_keys`` that are admitted, indices into the
    bank that are evicted, and the updated density/baseline for the kept bank
    entries followed by the admitted candidates (in that order).
    """
    import torch

    device = candidate_keys.device
    kernel = {"scale": scale, "power": power, "eps": eps}
    candidates = int(candidate_keys.shape[0])
    active = int(bank_keys.shape[0])
    bank_density = bank_density.to(device=device, dtype=torch.float32)
    baseline = bank_baseline.to(device=device, dtype=torch.float32).clamp_min(float(baseline_floor))
    max_accepted = min(candidates, int(capacity))
    empty_long = torch.zeros(0, dtype=torch.long, device=device)
    if candidates == 0 or max_accepted == 0:
        return {"admitted": empty_long, "evicted": empty_long, "kept_density": bank_density, "kept_baseline": baseline, "admitted_density": torch.zeros(0, device=device), "admitted_baseline": torch.zeros(0, device=device)}

    external = torch.zeros(candidates, device=device)
    score = torch.zeros(candidates, device=device)
    if active:
        for start in range(0, candidates, int(work_chunk)):
            stop = min(candidates, start + int(work_chunk))
            block_external = torch.zeros(stop - start, device=device)
            block_score = torch.zeros(stop - start, device=device)
            for b0, b1, contribution in _blocked_head_mean_riesz(candidate_keys[start:stop], bank_keys, block=16384, **kernel):
                block_external += contribution.sum(dim=1)
                block_score = torch.maximum(block_score, (contribution / baseline[b0:b1][None, :]).amax(dim=1))
            external[start:stop] = block_external
            score[start:stop] = block_score
    order = torch.argsort(score, descending=False, stable=True)
    accepted = max_accepted
    increment = torch.zeros(active, device=device)
    if active:
        accepted = 0
        running = torch.zeros(active, device=device)
        best = running.clone()
        ordered = candidate_keys.index_select(0, order[:max_accepted])
        for start in range(0, max_accepted, int(work_chunk)):
            stop = min(max_accepted, start + int(work_chunk))
            contribution = torch.cat([c for _, _, c in _blocked_head_mean_riesz(ordered[start:stop], bank_keys, block=16384, **kernel)], dim=1)
            cumulative = running[None, :] + contribution.cumsum(dim=0)  # [chunk, active]
            violating = ((bank_density[None, :] + cumulative) / baseline[None, :] >= float(growth_limit)).sum(dim=1)
            prefix = torch.arange(start + 1, stop + 1, device=device)
            budget = (active + prefix - int(capacity)).clamp_min(0)
            feasible = (violating <= budget).nonzero(as_tuple=False)
            if feasible.numel():
                last = int(feasible[-1, 0].item())
                accepted = start + last + 1
                best = cumulative[last].clone()
            running = cumulative[-1].clone()
        increment = best
    admitted = order[:accepted]
    selected = candidate_keys.index_select(0, admitted)
    selected_internal = torch.zeros(accepted, device=device)
    for start in range(0, accepted, int(work_chunk)):
        stop = min(accepted, start + int(work_chunk))
        block = head_mean_riesz(selected[start:stop], selected, **kernel)
        rows = torch.arange(stop - start, device=device)
        block[rows, rows + start] = 0.0
        selected_internal[start:stop] = block.sum(dim=1)
    selected_density = external.index_select(0, admitted) + selected_internal
    proposal_old = bank_density + increment
    free = max(int(capacity) - active, 0)
    evict_count = max(accepted - free, 0)
    evicted = empty_long
    if evict_count:
        must = proposal_old / baseline >= float(growth_limit)
        priority = proposal_old + must.float() * 1.0e30
        evicted = torch.topk(priority, k=evict_count).indices.sort().values
    keep_old = torch.ones(active, dtype=torch.bool, device=device)
    keep_old[evicted] = False
    kept_density = proposal_old[keep_old]
    if evict_count:
        evicted_keys = bank_keys.index_select(0, evicted)
        kept_keys = bank_keys[keep_old]
        for start in range(0, int(kept_keys.shape[0]), int(work_chunk)):
            stop = min(int(kept_keys.shape[0]), start + int(work_chunk))
            kept_density[start:stop] -= head_mean_riesz(kept_keys[start:stop], evicted_keys, **kernel).sum(dim=1)
        for start in range(0, accepted, int(work_chunk)):
            stop = min(accepted, start + int(work_chunk))
            selected_density[start:stop] -= head_mean_riesz(selected[start:stop], evicted_keys, **kernel).sum(dim=1)
    kept_density.clamp_min_(0.0)
    selected_density.clamp_min_(0.0)
    return {
        "admitted": admitted.sort().values,
        "admitted_order": admitted,
        "evicted": evicted,
        "kept_density": kept_density,
        "kept_baseline": baseline[keep_old],
        "admitted_density": selected_density,
        "admitted_baseline": selected_density.clamp_min(float(baseline_floor)),
    }


def future_merge_plan(
    chunk_keys: Any,
    keep_mask: Any,
    future_query_vectors: Any,
    *,
    threshold: float,
) -> Any:
    """Future Forcing's merge routing inside one chunk.

    Each key is projected onto the future-query proxies (``[horizon, heads, dim]``)
    giving ``K_hat`` in R^{heads*horizon}; every evicted token is routed to its most
    cosine-similar kept token and merged only when that cosine >= ``threshold``.
    Returns a long tensor with, for every token, the index of the kept token it
    merges into, or -1.
    """
    import torch

    projected = torch.einsum("nhd,ehd->nhe", chunk_keys.float(), future_query_vectors.float()).reshape(int(chunk_keys.shape[0]), -1)
    projected = torch.nn.functional.normalize(projected, dim=-1)
    kept = keep_mask.nonzero(as_tuple=False).reshape(-1)
    evicted = (~keep_mask).nonzero(as_tuple=False).reshape(-1)
    target = torch.full((int(chunk_keys.shape[0]),), -1, dtype=torch.long, device=chunk_keys.device)
    if not kept.numel() or not evicted.numel():
        return target
    similarity = projected.index_select(0, evicted) @ projected.index_select(0, kept).T
    best, where = similarity.max(dim=1)
    admitted = best >= float(threshold)
    target[evicted[admitted]] = kept[where[admitted]]
    return target


def merge_values(values: Any, merge_target: Any, weights: Any) -> Any:
    """Weighted value merge: V_i <- (w_i V_i + sum_j w_j V_j) / (w_i + sum_j w_j)."""
    import torch

    merged = values.float().clone()
    sources = (merge_target >= 0).nonzero(as_tuple=False).reshape(-1)
    if not sources.numel():
        return values
    destinations = merge_target.index_select(0, sources)
    w = weights.float().clamp_min(1.0e-12)
    numerator = merged * w[:, None, None]
    denominator = w.clone()
    numerator.index_add_(0, destinations, merged.index_select(0, sources) * w.index_select(0, sources)[:, None, None])
    denominator.index_add_(0, destinations, w.index_select(0, sources))
    touched = torch.unique(destinations)
    merged[touched] = numerator[touched] / denominator[touched][:, None, None]
    return merged.to(values.dtype)

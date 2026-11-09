"""Visible-subset attention for WorldKV retrieval and Focused Forcing.

Both methods keep more history in storage than any single attention call reads.
These calls bypass the generic per-layer staging (which would copy the whole
layer to the GPU) and transfer only the selected rows.
"""
from __future__ import annotations

import math
from typing import Any

from .cache_attention import _write_source_ids, _write_tpos


def _int(value: Any) -> int:
    return int(value.item()) if hasattr(value, "item") else int(value)


def _zscore(values: Any, dim: int) -> Any:
    mean = values.mean(dim=dim, keepdim=True)
    std = values.std(dim=dim, keepdim=True, unbiased=False).clamp_min(1.0e-6)
    return (values - mean) / std


def _rotation_angle(left: Any, right: Any) -> Any:
    """Geodesic angle between rotation matrices ``[..., 3, 3]``."""
    import torch

    relative = torch.matmul(left.transpose(-1, -2), right)
    trace = relative.diagonal(dim1=-2, dim2=-1).sum(-1)
    return torch.arccos(((trace - 1.0) / 2.0).clamp(-1.0, 1.0))


def estimate_context_rotations(video_cthw: Any, *, latent_frames: int, stride: int = 4) -> dict[str, Any]:
    """Absolute camera rotation of every context latent frame, relative to the LAST one.

    Uses the same SIFT + homography rotation recovery as the runtime's tail
    camera estimator, chained over consecutive latent-aligned pixel frames.
    """
    import cv2
    import numpy as np
    import torch

    cv2.setRNGSeed(0)
    pixels = int(video_cthw.shape[1])
    indices = [min(pixels - 1, frame * int(stride)) for frame in range(int(latent_frames))]
    rgb = (((video_cthw[:, indices].clamp(-1, 1) + 1.0) * 127.5).to(torch.uint8).permute(1, 2, 3, 0).cpu().numpy())
    gray = [cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY) for frame in rgb]
    height, width = gray[0].shape
    intrinsic = np.array([[502.9116 / 832.0 * width, 0.0, 415.77786 / 832.0 * width], [0.0, 503.10812 / 480.0 * height, 239.77779 / 480.0 * height], [0.0, 0.0, 1.0]])
    detector = cv2.SIFT_create(nfeatures=2000)
    matcher = cv2.BFMatcher(cv2.NORM_L2)
    features = [detector.detectAndCompute(image, None) for image in gray]
    absolute = [np.eye(3)]
    robust = 0
    for index in range(1, len(gray)):
        step = np.eye(3)
        (kp_a, des_a), (kp_b, des_b) = features[index - 1], features[index]
        if des_a is not None and des_b is not None and len(des_a) >= 2 and len(des_b) >= 2:
            good = [m for m, n in matcher.knnMatch(des_a, des_b, k=2) if m.distance < 0.7 * n.distance]
            if len(good) >= 12:
                pa = np.float32([kp_a[m.queryIdx].pt for m in good])
                pb = np.float32([kp_b[m.trainIdx].pt for m in good])
                homography, mask = cv2.findHomography(pa, pb, cv2.RANSAC, ransacReprojThreshold=2.0)
                if homography is not None and mask is not None and int(mask.sum()) >= 10:
                    raw = np.linalg.inv(intrinsic) @ homography @ intrinsic
                    u, _, vt = np.linalg.svd(raw)
                    rotation = u @ vt
                    if np.linalg.det(rotation) < 0:
                        u[:, -1] *= -1
                        rotation = u @ vt
                    step = rotation.T
                    robust += 1
        absolute.append(absolute[-1] @ step)
    stack = np.stack(absolute)
    last_inverse = stack[-1].T
    relative_to_last = np.einsum("ij,fjk->fik", last_inverse, stack)
    return {
        "rotations": torch.tensor(relative_to_last, dtype=torch.float32),
        "robust_pairs": robust,
        "pairs": max(0, len(gray) - 1),
        "robust_fraction": robust / max(1, len(gray) - 1),
    }


def plan_absolute_rotations(camera: Any, *, context_frames: int) -> Any:
    """Absolute rotation of every timeline frame relative to the last context frame."""
    import torch

    steps = camera.relative_poses[:, :3, :3].float()
    rotations = torch.eye(3).repeat(int(steps.shape[0]), 1, 1)
    current = torch.eye(3)
    for frame in range(int(context_frames), int(steps.shape[0])):
        current = current @ steps[frame]
        rotations[frame] = current
    return rotations


class VisibleSubsetController:
    """Holds per-layer summaries and decides what each attention call may see."""

    def __init__(self, policy: Any) -> None:
        self.policy = policy
        self.name = policy.name
        self.parameters = policy.parameters
        self.cache = policy.cache
        self.frame_tokens = policy.frame_tokens
        self.tokens_per_chunk = policy.tokens_per_chunk
        self.chunk_size = policy.chunk_size
        self.context_frames = int(policy.geometry.context_frames)
        self.current_chunk = -1
        self.dense_chunks: set[int] = set()
        self.stored_chunks: list[int] = []
        self.chunk_mean_keys: dict[int, dict[int, Any]] = {index: {} for index in range(len(self.cache))}
        self.frame_means: dict[int, Any] = {}
        self.retrieval_mode = str(self.parameters.get("retrieval", "query"))
        self.context_rotations = None
        self.timeline_rotations = None
        self.camera_status: dict[str, Any] = {"mode": self.retrieval_mode}
        self.selected_log: list[dict[str, Any]] = []
        self.visible_tokens_log: list[int] = []
        for index, layer in enumerate(self.cache):
            layer["rb_visible"] = {"controller": self, "layer": index}

    # --------------------------------------------------------------- case inputs
    def attach_case_inputs(self, *, context_payload: Any, camera: Any) -> None:
        if self.name != "worldkv" or self.retrieval_mode != "camera":
            return
        import torch.distributed as dist

        rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
        estimate = estimate_context_rotations(context_payload, latent_frames=self.context_frames) if rank == 0 and context_payload is not None else None
        if dist.is_available() and dist.is_initialized():
            box = [estimate]
            dist.broadcast_object_list(box, src=0)
            estimate = box[0]
        if estimate is None or float(estimate["robust_fraction"]) < 0.5:
            self.retrieval_mode = "query"
            self.camera_status = {"mode": "query", "fallback_from": "camera", "robust_fraction": None if estimate is None else estimate["robust_fraction"]}
            return
        rotations = plan_absolute_rotations(camera, context_frames=self.context_frames)
        rotations[: self.context_frames] = estimate["rotations"]
        self.timeline_rotations = rotations
        self.camera_status = {"mode": "camera", "robust_fraction": estimate["robust_fraction"], "robust_pairs": estimate["robust_pairs"]}

    # --------------------------------------------------------------- lifecycle
    def prepare_chunk(self, chunk_idx: int) -> None:
        self.current_chunk = int(chunk_idx)
        units = []
        for layer in self.cache:
            heads = int(layer["k"].shape[2])
            units.append(self._expected_visible_tokens(int(chunk_idx)) * heads)
        for layer, value in zip(self.cache, units):
            layer["rb_visible_history_units"] = int(value)
        self.visible_tokens_log.append(int(sum(units)))

    def after_compaction(self, chunk_idx: int) -> None:
        return None

    def after_clean(self, chunk_idx: int) -> None:
        import torch

        self.dense_chunks.add(int(chunk_idx))
        if self.name == "focused_forcing":
            start = int(chunk_idx) * self.tokens_per_chunk
            for index, layer in enumerate(self.cache):
                device = torch.device(str(layer.get("compute_device", layer["k"].device)))
                keys = layer["k"][0, start:start + self.tokens_per_chunk].to(device).float()
                means = keys.reshape(self.chunk_size, self.frame_tokens, *keys.shape[1:]).mean(dim=1)
                store = self.frame_means.get(index)
                total_frames = int(layer["k"].shape[1]) // self.frame_tokens
                if store is None:
                    store = torch.zeros((total_frames, *means.shape[1:]), dtype=torch.float32, device=device)
                    self.frame_means[index] = store
                first = int(chunk_idx) * self.chunk_size
                store[first:first + self.chunk_size] = means

    def register_stored_chunk(self, layer_idx: int, chunk_idx: int, layer: dict[str, Any]) -> None:
        import torch

        if int(chunk_idx) not in self.stored_chunks and layer_idx == 0:
            self.stored_chunks.append(int(chunk_idx))
            self.dense_chunks.discard(int(chunk_idx))
        live = _int(layer["local_end_index"])
        ids = layer["source_ids"][0, :live]
        positions = (ids // self.tokens_per_chunk == int(chunk_idx)).nonzero(as_tuple=False).reshape(-1)
        device = torch.device(str(layer.get("compute_device", layer["k"].device)))
        keys = layer["k"][0].index_select(0, positions.to(layer["k"].device)).to(device).float()
        self.chunk_mean_keys[layer_idx][int(chunk_idx)] = keys.mean(dim=0)

    def on_rope_reindex(self, *, plan: Any, freqs: Any) -> None:
        """Keep summaries consistent with RoPE-retargeted context keys."""
        import torch

        if plan.mode == "off":
            return
        if self.name == "worldkv":
            for layer_idx, layer in enumerate(self.cache):
                live = _int(layer["local_end_index"])
                ids = layer["source_ids"][0, :live].cpu()
                chunk_of = ids // self.tokens_per_chunk
                for chunk in list(self.chunk_mean_keys[layer_idx]):
                    positions = (chunk_of == chunk).nonzero(as_tuple=False).reshape(-1)
                    device = self.chunk_mean_keys[layer_idx][chunk].device
                    self.chunk_mean_keys[layer_idx][chunk] = layer["k"][0].index_select(0, positions.to(layer["k"].device)).to(device).float().mean(dim=0)
            return
        virtual = torch.tensor(plan.virtual_positions(self.context_frames), dtype=torch.float64)
        theta = torch.angle(freqs[1].to(torch.complex128)).cpu()
        pairs = plan.temporal_pair_count(int(freqs.shape[1]))
        theta = theta[:pairs]
        delta = virtual - torch.arange(self.context_frames, dtype=torch.float64)
        rotation = torch.polar(torch.ones(self.context_frames, pairs, dtype=torch.float64), delta[:, None] * theta[None, :]).to(torch.complex64)
        for index, store in self.frame_means.items():
            block = store[: self.context_frames].clone()
            complex_block = torch.view_as_complex(block.reshape(self.context_frames, block.shape[1], -1, 2).contiguous())
            complex_block[..., :pairs] *= rotation.to(store.device)[:, None, :]
            store[: self.context_frames] = torch.view_as_real(complex_block).reshape(block.shape)

    # --------------------------------------------------------------- selection
    def _sink_chunks(self) -> int:
        return int(self.parameters.get("sink_chunks", 1))

    def _expected_visible_tokens(self, chunk_idx: int) -> int:
        history_tokens = int(chunk_idx) * self.tokens_per_chunk
        if self.name == "focused_forcing":
            frames = history_tokens // self.frame_tokens
            return min(frames, int(self.parameters["frame_budget"])) * self.frame_tokens
        stored = len(self.stored_chunks)
        retrieved = min(stored, int(self.parameters["retrieved_chunks"]))
        per_stored = self.frame_tokens + int(round(float(self.parameters["nonanchor_keep_ratio"]) * (self.tokens_per_chunk - self.frame_tokens)))
        dense = history_tokens - stored * self.tokens_per_chunk
        return dense + retrieved * per_stored

    def _worldkv_selected(self, layer_idx: int, query: Any) -> list[int]:
        import torch

        stored = list(self.stored_chunks)
        budget = int(self.parameters["retrieved_chunks"])
        if len(stored) <= budget:
            return stored
        if self.retrieval_mode == "camera" and self.timeline_rotations is not None:
            frame = min(int(self.timeline_rotations.shape[0]) - 1, self.current_chunk * self.chunk_size + self.chunk_size // 2)
            target = self.timeline_rotations[frame]
            centers = [min(self.context_frames - 1, chunk * self.chunk_size + self.chunk_size // 2) if chunk * self.chunk_size < self.context_frames else chunk * self.chunk_size + self.chunk_size // 2 for chunk in stored]
            poses = self.timeline_rotations[torch.tensor(centers)]
            distance = _rotation_angle(target[None].expand_as(poses), poses)
            order = torch.argsort(distance, stable=True)[:budget].tolist()
            return sorted(stored[index] for index in order)
        from ..methods.rebuttal_kernels import sampled_query_rows

        means = torch.stack([self.chunk_mean_keys[layer_idx][chunk] for chunk in stored]).to(query.device)  # [C,H,D]
        rows = sampled_query_rows(int(query.shape[1]), int(self.parameters.get("query_samples", 256)), device=query.device)
        q = query[0].index_select(0, rows).float()  # [S,H,D]
        logits = torch.einsum("shd,chd->shc", q, means) / math.sqrt(int(q.shape[-1]))
        score = torch.softmax(logits, dim=-1).mean(dim=(0, 1))
        order = torch.argsort(score, descending=True, stable=True)[:budget].tolist()
        return sorted(stored[index] for index in order)

    # --------------------------------------------------------------- attention
    def attend(self, layer_idx: int, query: Any, key: Any, value: Any, cache: dict[str, Any], *, history_len: int, attender: Any) -> Any:
        import torch

        if history_len <= 0:
            return attender(query, key, value)
        storage_device = cache["k"].device
        if self.name == "worldkv":
            selected = self._worldkv_selected(layer_idx, query)
            if layer_idx == 0:
                self.selected_log.append({"chunk": self.current_chunk, "selected": selected, "mode": self.retrieval_mode})
            ids = cache["source_ids"][0, :history_len]
            chunk_of = ids // self.tokens_per_chunk
            stored = torch.tensor(self.stored_chunks, dtype=torch.long, device=ids.device)
            allowed = ~torch.isin(chunk_of, stored) if stored.numel() else torch.ones_like(chunk_of, dtype=torch.bool)
            if selected:
                allowed |= torch.isin(chunk_of, torch.tensor(selected, dtype=torch.long, device=ids.device))
            positions = allowed.nonzero(as_tuple=False).reshape(-1)
            history_k = cache["k"][0].index_select(0, positions.to(storage_device)).to(query.device, non_blocking=True)
            history_v = cache["v"][0].index_select(0, positions.to(storage_device)).to(query.device, non_blocking=True)
            return attender(query, torch.cat([history_k, key[0]])[None], torch.cat([history_v, value[0]])[None])
        return self._focused_attend(layer_idx, query, key, value, cache, history_len=history_len, attender=attender)

    def _focused_attend(self, layer_idx: int, query: Any, key: Any, value: Any, cache: dict[str, Any], *, history_len: int, attender: Any) -> Any:
        import torch

        frame_tokens = self.frame_tokens
        history_frames = history_len // frame_tokens
        budget = int(self.parameters["frame_budget"])
        storage_device = cache["k"].device
        if history_frames <= budget:
            history_k = cache["k"][0, :history_len].to(query.device, non_blocking=True)
            history_v = cache["v"][0, :history_len].to(query.device, non_blocking=True)
            return attender(query, torch.cat([history_k, key[0]])[None], torch.cat([history_v, value[0]])[None])
        heads, dim = int(query.shape[2]), int(query.shape[3])
        query_frames = max(1, int(query.shape[1]) // frame_tokens)
        sink = min(int(self.parameters["sink_frames"]), history_frames)
        recent_start = max(sink, history_frames - int(self.parameters["recent_frames"]))
        reserved = list(range(sink)) + list(range(recent_start, history_frames))
        candidates = torch.arange(sink, recent_start, device=query.device)
        scored = max(0, budget - len(reserved))
        means = self.frame_means[layer_idx][:history_frames].to(query.device)  # [F,H,D]
        q_bar = query[0, : query_frames * frame_tokens].float().reshape(query_frames, frame_tokens, heads, dim).mean(dim=1)
        attention = torch.einsum("qhd,fhd->qhf", q_bar, means.index_select(0, candidates)) / math.sqrt(dim)
        global_mean = means.mean(dim=0)  # [H,D]
        redundancy = torch.nn.functional.cosine_similarity(means.index_select(0, candidates), global_mean[None].expand(int(candidates.numel()), -1, -1), dim=-1)  # [C,H]
        weight = float(self.parameters["attention_weight"])
        score = weight * _zscore(attention, dim=-1) + (1.0 - weight) * _zscore(-redundancy.T, dim=-1)[None]
        top = candidates[torch.topk(score, k=min(scored, int(candidates.numel())), dim=-1).indices]  # [Q,H,m]
        reserved_tensor = torch.tensor(reserved, dtype=torch.long, device=query.device)
        frames = torch.cat([reserved_tensor[None, None].expand(query_frames, heads, -1), top], dim=-1).sort(dim=-1).values  # [Q,H,b]
        union = torch.unique(frames)
        local = torch.full((history_frames,), -1, dtype=torch.long, device=query.device)
        local[union] = torch.arange(int(union.numel()), device=query.device)
        rows = (union[:, None] * frame_tokens + torch.arange(frame_tokens, device=query.device)[None]).reshape(-1)
        union_k = cache["k"][0].index_select(0, rows.to(storage_device)).to(query.device, non_blocking=True)  # [U*T,H,D]
        union_v = cache["v"][0].index_select(0, rows.to(storage_device)).to(query.device, non_blocking=True)
        try:
            from flash_attn import flash_attn_func
        except Exception:  # pragma: no cover - CPU test environment
            flash_attn_func = None
        output = torch.empty_like(query)
        token_offsets = torch.arange(frame_tokens, device=query.device)
        group = max(1, int(self.parameters.get("head_group_size", 8)))
        for h0 in range(0, heads, group):
            h1 = min(heads, h0 + group)
            selected = frames[:, h0:h1]  # [Q,G,b]
            gather_rows = (local[selected][..., None] * frame_tokens + token_offsets).reshape(query_frames, h1 - h0, -1)  # [Q,G,bT]
            head_index = torch.arange(h0, h1, device=query.device)[None, :, None].expand_as(gather_rows)
            k_hist = union_k[gather_rows, head_index]  # [Q,G,bT,D]
            v_hist = union_v[gather_rows, head_index]
            k_cur = key[0, :, h0:h1].permute(1, 0, 2)[None].expand(query_frames, -1, -1, -1)  # [Q,G,4T,D]
            v_cur = value[0, :, h0:h1].permute(1, 0, 2)[None].expand(query_frames, -1, -1, -1)
            k_all = torch.cat([k_hist, k_cur], dim=2).reshape(query_frames * (h1 - h0), -1, 1, dim)
            v_all = torch.cat([v_hist, v_cur], dim=2).reshape(query_frames * (h1 - h0), -1, 1, dim)
            q_all = query[0, : query_frames * frame_tokens, h0:h1].reshape(query_frames, frame_tokens, h1 - h0, dim).permute(0, 2, 1, 3).reshape(query_frames * (h1 - h0), frame_tokens, 1, dim)
            if flash_attn_func is not None and q_all.is_cuda:
                attended = flash_attn_func(q_all.contiguous(), k_all.contiguous(), v_all.contiguous())
            else:
                attended = torch.nn.functional.scaled_dot_product_attention(q_all.transpose(1, 2).float(), k_all.transpose(1, 2).float(), v_all.transpose(1, 2).float()).transpose(1, 2).to(query.dtype)
            attended = attended.reshape(query_frames, h1 - h0, frame_tokens, dim).permute(0, 2, 1, 3).reshape(query_frames * frame_tokens, h1 - h0, dim)
            output[0, : query_frames * frame_tokens, h0:h1] = attended
        return output

    def diagnostics(self) -> dict[str, Any]:
        return {
            "retrieval": self.camera_status if self.name == "worldkv" else None,
            "stored_chunks": len(self.stored_chunks),
            "selected_log": self.selected_log[-64:],
            "visible_history_units_by_chunk": self.visible_tokens_log,
        }


def visible_subset_attention(query: Any, key: Any, value: Any, *, cache: Any, current_start: int, frame_tokens: int, attender: Any) -> Any:
    """Write the current chunk to storage, then attend to the controller's visible subset."""
    spec = cache["rb_visible"]
    controller, layer_idx = spec["controller"], int(spec["layer"])
    sequence = int(query.shape[1])
    current_end = int(current_start) + sequence
    previous_global = _int(cache["global_end_index"])
    if current_end < previous_global:
        raise RuntimeError(f"non-monotonic absolute KV write: {current_end} < {previous_global}")
    is_new = current_end > previous_global
    previous_local = _int(cache["local_end_index"])
    local_start = previous_local if is_new else previous_local - sequence
    local_end = local_start + sequence
    if local_start < 0 or local_end > int(cache["k"].shape[1]):
        raise RuntimeError(f"visible-subset KV overflow: [{local_start},{local_end}) of {int(cache['k'].shape[1])}")
    storage = cache["k"].device
    cache["k"][:, local_start:local_end] = key.to(storage)
    cache["v"][:, local_start:local_end] = value.to(storage)
    start_frame = int(current_start) // int(frame_tokens)
    _write_tpos(cache, "tpos", local_start, local_end, frame_tokens=frame_tokens, start_frame=start_frame)
    _write_source_ids(cache, "source_ids", local_start, local_end, current_start=int(current_start))
    cache["local_end_index"].fill_(local_end)
    cache["global_end_index"].fill_(current_end)
    return controller.attend(layer_idx, query, key, value, cache, history_len=local_start, attender=attender)

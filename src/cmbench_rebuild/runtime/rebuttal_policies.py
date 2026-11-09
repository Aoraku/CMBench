"""Online lifecycles for the rebuttal baselines.

All token-eviction baselines share DeCoPrune's lifecycle so that only the
selection rule differs: chunk 0 is a dense sink, the most recent
``local_window_chunks`` stay dense, and a chunk is finalized (scored, selected and
physically compacted) when it leaves that window.  Selection is per layer.

``focused_forcing`` and ``worldkv`` additionally restrict which stored tokens each
attention call may see; see :mod:`.rebuttal_attention`.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping

from ..methods.rebuttal_kernels import (
    densitykv_admit,
    future_merge_plan,
    merge_values,
    spatial_pool_chunk_scores,
    top_fraction_mask,
    worldkv_anchor_scores,
)
from .compaction import drop_layer_source_ids

TOKEN_EVICTION_METHODS = frozenset({"snapkv", "future_forcing", "densitykv", "worldkv"})
VISIBLE_SUBSET_METHODS = frozenset({"focused_forcing", "worldkv"})
REBUTTAL_METHODS = TOKEN_EVICTION_METHODS | VISIBLE_SUBSET_METHODS


def _int(value: Any) -> int:
    return int(value.item()) if hasattr(value, "item") else int(value)


def _dist():
    import torch.distributed as dist

    return dist if dist.is_available() and dist.is_initialized() else None


def _all_reduce_sum(tensor: Any) -> Any:
    dist = _dist()
    if dist is not None:
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return tensor


def _all_gather_heads(tensor: Any) -> Any:
    """Concatenate a ``[tokens, local_heads, dim]`` tensor over sequence-parallel ranks."""
    import torch

    dist = _dist()
    if dist is None:
        return tensor
    parts = [torch.empty_like(tensor) for _ in range(dist.get_world_size())]
    dist.all_gather(parts, tensor.contiguous())
    return torch.cat(parts, dim=1)


def token_grid(frame_tokens: int) -> tuple[int, int]:
    if int(frame_tokens) == 1560:
        return 30, 52
    height = int(round(int(frame_tokens) ** 0.5))
    while height > 1 and int(frame_tokens) % height:
        height -= 1
    return height, int(frame_tokens) // height


@dataclass
class ChunkRecord:
    chunk_idx: int
    phase: str
    finalized: bool = False
    stats: dict[str, Any] = field(default_factory=dict)


class RebuttalPolicy:
    """Per-case state machine driven by :class:`OnlinePolicyRuntime`."""

    def __init__(self, *, cache: list[dict[str, Any]], geometry: Any, name: str, parameters: Mapping[str, Any],
                 request: Any, pipeline: Any, emit: Callable[[Mapping[str, Any]], None]) -> None:
        import torch

        self.cache = cache
        self.geometry = geometry
        self.name = str(name)
        self.parameters = dict(parameters)
        self.request = request
        self.pipeline = pipeline
        self.emit = emit
        self.events: list[dict[str, Any]] = []
        self.records: list[ChunkRecord] = []
        self.window = int(self.parameters.get("local_window_chunks", 1))
        self.tokens_per_chunk = int(geometry.tokens_per_chunk)
        self.frame_tokens = int(geometry.frame_tokens)
        self.chunk_size = int(geometry.chunk_size)
        self.total_tokens = (int(geometry.context_frames) + int(geometry.output_frames)) * self.frame_tokens
        self.height, self.width = token_grid(self.frame_tokens)
        self.total_heads = int(geometry.num_heads)
        self.samples = int(self.parameters.get("query_samples", 256))
        self.finalize_seconds = 0.0
        needs_attention = self.name in {"snapkv", "future_forcing"}
        for layer in cache:
            device = layer["k"].device
            pin = device.type == "cpu" and torch.cuda.is_available()
            if needs_attention:
                layer["rb_attn_by_source"] = torch.zeros(self.total_tokens, dtype=torch.float32, device=device, pin_memory=pin)
            if self.name == "future_forcing":
                horizon = int(self.parameters["horizon"])
                layer["rb_future_by_source"] = torch.zeros(self.total_tokens, dtype=torch.float32, device=device, pin_memory=pin)
                layer["rb_future_qvec"] = torch.zeros((horizon, int(layer["k"].shape[2]), int(layer["k"].shape[3])), dtype=torch.float32, device=device, pin_memory=pin)
            if self.name == "densitykv":
                layer["rb_density_by_source"] = torch.zeros(self.total_tokens, dtype=torch.float32, device=device, pin_memory=pin)
                layer["rb_baseline_by_source"] = torch.zeros(self.total_tokens, dtype=torch.float32, device=device, pin_memory=pin)
        if self.name in VISIBLE_SUBSET_METHODS:
            from .rebuttal_attention import VisibleSubsetController

            self.visible = VisibleSubsetController(self)
        else:
            self.visible = None

    # ------------------------------------------------------------------ hooks
    def prepare_chunk(self, chunk_idx: int) -> None:
        if self.visible is not None:
            self.visible.prepare_chunk(int(chunk_idx))

    def before_clean(self, chunk_idx: int, *, phase: str) -> None:
        if self.name in TOKEN_EVICTION_METHODS:
            self._finalize_due(next_chunk_idx=int(chunk_idx) + 1)
        if self.visible is not None:
            self.visible.after_compaction(int(chunk_idx))
        capture = self._capture_spec()
        if capture is not None:
            for layer in self.cache:
                layer["rb_capture"] = capture

    def after_clean(self, chunk_idx: int, *, phase: str) -> None:
        for layer in self.cache:
            layer.pop("rb_capture", None)
        if int(chunk_idx) > 0 and self.name in TOKEN_EVICTION_METHODS:
            self.records.append(ChunkRecord(int(chunk_idx), str(phase)))
        if self.visible is not None:
            self.visible.after_clean(int(chunk_idx))

    def _capture_spec(self) -> dict[str, Any] | None:
        if self.name == "snapkv":
            return {"samples": self.samples}
        if self.name == "future_forcing":
            return {"samples": self.samples, "future_horizon": int(self.parameters["horizon"]), "frame_tokens": self.frame_tokens}
        return None

    # ------------------------------------------------------------ finalization
    def _finalize_due(self, *, next_chunk_idx: int) -> None:
        import time
        import torch

        cutoff = int(next_chunk_idx) - 2 * self.window
        for record in self.records:
            if record.finalized or record.chunk_idx >= cutoff:
                continue
            started = time.perf_counter()
            ids = torch.arange(record.chunk_idx * self.tokens_per_chunk, (record.chunk_idx + 1) * self.tokens_per_chunk, dtype=torch.long)
            passes = max(1, int(next_chunk_idx) - 1 - record.chunk_idx)
            kept_per_layer = []
            for layer_idx, layer in enumerate(self.cache):
                kept_per_layer.append(self._finalize_layer(layer, layer_idx, ids, record, passes))
            record.finalized = True
            self.finalize_seconds += time.perf_counter() - started
            event = {
                "event": "rebuttal_chunk_finalized",
                "method": self.name,
                "phase": record.phase,
                "chunk_idx": record.chunk_idx,
                "next_chunk_idx": int(next_chunk_idx),
                "kept_tokens_mean": float(sum(kept_per_layer) / max(1, len(kept_per_layer))),
                "kept_tokens_min": int(min(kept_per_layer)),
                "kept_tokens_max": int(max(kept_per_layer)),
                "total_tokens": self.tokens_per_chunk,
                **record.stats,
            }
            self.events.append(event)
            self.emit(event)

    def _positions(self, layer: dict[str, Any], ids: Any) -> Any:
        import torch

        live = _int(layer["local_end_index"])
        physical = layer["source_ids"][0, :live]
        wanted = ids.to(physical.device)
        positions = torch.searchsorted(physical, wanted)
        if positions.numel() and (int(positions.max().item()) >= live or not torch.equal(physical.index_select(0, positions), wanted)):
            raise RuntimeError(f"{self.name}: chunk tokens are not all physically present at finalization")
        return positions

    def _compute_device(self, layer: dict[str, Any]) -> Any:
        import torch

        return torch.device(str(layer.get("compute_device", layer["k"].device)))

    def _finalize_layer(self, layer: dict[str, Any], layer_idx: int, ids: Any, record: ChunkRecord, passes: int) -> int:
        import torch

        device = self._compute_device(layer)
        if self.name == "snapkv":
            scores = layer["rb_attn_by_source"].index_select(0, ids.to(layer["rb_attn_by_source"].device)).to(device)
            _all_reduce_sum(scores)
            scores = spatial_pool_chunk_scores(scores, frames=self.chunk_size, height=self.height, width=self.width, kernel=int(self.parameters["pool_kernel"]))
            keep = top_fraction_mask(scores, keep_ratio=float(self.parameters["keep_ratio"]))
            drop_layer_source_ids(layer, ids[~keep.cpu()])
            return int(keep.sum().item())
        if self.name == "future_forcing":
            history = layer["rb_attn_by_source"].index_select(0, ids.to(layer["rb_attn_by_source"].device)).to(device)
            future = layer["rb_future_by_source"].index_select(0, ids.to(layer["rb_future_by_source"].device)).to(device)
            _all_reduce_sum(history)
            _all_reduce_sum(future)
            horizon = int(self.parameters["horizon"])
            history = history / float(self.samples * self.total_heads * passes)
            future = future / float(self.samples * self.total_heads * horizon)
            weight = float(self.parameters["future_weight"])
            scores = weight * future + (1.0 - weight) * history
            keep = top_fraction_mask(scores, keep_ratio=float(self.parameters["keep_ratio"]))
            merged = 0
            if bool(self.parameters.get("merge", True)):
                positions = self._positions(layer, ids)
                keys = layer["k"][0].index_select(0, positions.to(layer["k"].device)).to(device)
                qvec = layer["rb_future_qvec"].to(device)
                target = future_merge_plan(keys, keep, qvec, threshold=float(self.parameters["merge_threshold"]))
                merged = int((target >= 0).sum().item())
                if merged:
                    values = layer["v"][0].index_select(0, positions.to(layer["v"].device)).to(device)
                    updated = merge_values(values, target, future.clamp_min(0.0))
                    layer["v"][0].index_copy_(0, positions.to(layer["v"].device), updated.to(layer["v"].device, layer["v"].dtype))
            record.stats.setdefault("merged_tokens_by_layer", []).append(merged)
            drop_layer_source_ids(layer, ids[~keep.cpu()])
            return int(keep.sum().item())
        if self.name == "densitykv":
            return self._densitykv_layer(layer, ids, record, device)
        if self.name == "worldkv":
            positions = self._positions(layer, ids)
            keys = _all_gather_heads(layer["k"][0].index_select(0, positions.to(layer["k"].device)).to(device))
            scores = worldkv_anchor_scores(keys, frame_tokens=self.frame_tokens)
            keep_rest = top_fraction_mask(-scores, keep_ratio=float(self.parameters["nonanchor_keep_ratio"]))
            keep = torch.cat([torch.ones(self.frame_tokens, dtype=torch.bool, device=device), keep_rest])
            drop_layer_source_ids(layer, ids[~keep.cpu()])
            self.visible.register_stored_chunk(layer_idx, record.chunk_idx, layer)
            return int(keep.sum().item())
        raise RuntimeError(f"no finalization rule for {self.name}")

    def _densitykv_layer(self, layer: dict[str, Any], ids: Any, record: ChunkRecord, device: Any) -> int:
        import torch

        live = _int(layer["local_end_index"])
        physical = layer["source_ids"][0, :live].cpu()
        chunk_lo = int(ids[0].item())
        bank_mask = (physical >= self.tokens_per_chunk) & (physical < chunk_lo)
        bank_ids = physical[bank_mask]
        bank_positions = bank_mask.nonzero(as_tuple=False).reshape(-1)
        candidate_positions = self._positions(layer, ids).cpu()
        storage = layer["k"].device
        bank_keys = _all_gather_heads(layer["k"][0].index_select(0, bank_positions.to(storage)).to(device))
        candidate_keys = _all_gather_heads(layer["k"][0].index_select(0, candidate_positions.to(storage)).to(device))
        density_store, baseline_store = layer["rb_density_by_source"], layer["rb_baseline_by_source"]
        result = densitykv_admit(
            candidate_keys,
            bank_keys,
            density_store.index_select(0, bank_ids.to(density_store.device)).to(device),
            baseline_store.index_select(0, bank_ids.to(baseline_store.device)).to(device),
            capacity=int(self.parameters["max_entries"]),
            scale=float(self.parameters["density_scale"]),
            power=float(self.parameters["riesz_power"]),
            eps=float(self.parameters["riesz_eps"]),
            growth_limit=float(self.parameters["growth_limit"]),
            work_chunk=int(self.parameters["work_chunk_size"]),
        )
        admitted = result["admitted_order"].cpu()
        keep_candidate = torch.zeros(int(ids.numel()), dtype=torch.bool)
        keep_candidate[admitted] = True
        evicted_ids = bank_ids[result["evicted"].cpu()]
        keep_bank = torch.ones(int(bank_ids.numel()), dtype=torch.bool)
        keep_bank[result["evicted"].cpu()] = False
        kept_bank_ids = bank_ids[keep_bank]
        density_store.index_copy_(0, kept_bank_ids.to(density_store.device), result["kept_density"].to(density_store.device))
        baseline_store.index_copy_(0, kept_bank_ids.to(baseline_store.device), result["kept_baseline"].to(baseline_store.device))
        admitted_ids = ids[admitted]
        density_store.index_copy_(0, admitted_ids.to(density_store.device), result["admitted_density"].to(density_store.device))
        baseline_store.index_copy_(0, admitted_ids.to(baseline_store.device), result["admitted_baseline"].to(baseline_store.device))
        rejected = torch.cat([ids[~keep_candidate], evicted_ids])
        drop_layer_source_ids(layer, rejected.sort().values)
        record.stats.setdefault("bank_size_by_layer", []).append(int(kept_bank_ids.numel() + admitted_ids.numel()))
        record.stats.setdefault("evicted_by_layer", []).append(int(evicted_ids.numel()))
        return int(keep_candidate.sum().item())

    def diagnostics(self) -> dict[str, Any]:
        data = {
            "rebuttal_method": self.name,
            "finalized_chunks": sum(record.finalized for record in self.records),
            "unfinalized_chunks": sum(not record.finalized for record in self.records),
            "finalize_seconds": self.finalize_seconds,
            "events": list(self.events),
        }
        if self.visible is not None:
            data["visible_subset"] = self.visible.diagnostics()
        return data


def capture_attention_statistics(cache: dict[str, Any], query: Any, *, left: int, local_start: int, local_end: int, aux: Mapping[str, Any] | None) -> None:
    """Attention-side evidence for SnapKV / Future Forcing (runs inside ``cached_attention``)."""
    from ..methods.rebuttal_kernels import attention_mass_over_keys, sampled_query_rows

    capture = cache.get("rb_capture")
    if not capture:
        return
    history = int(local_start) - int(left)
    if history <= 0:
        return
    keys = cache["k"][0, int(left):int(local_end)]
    ids = cache["source_ids"][0, int(left):int(local_start)]
    rows = sampled_query_rows(int(query.shape[1]), int(capture["samples"]), device=query.device)
    mass = attention_mass_over_keys(query[0].index_select(0, rows), keys, target_stop=history)
    buffer = cache["rb_attn_by_source"]
    buffer.index_add_(0, ids.to(buffer.device), mass.to(buffer.device))
    future = (aux or {}).get("future_queries")
    if future is not None and "rb_future_by_source" in cache:
        total = None
        for index in range(int(future.shape[0])):
            part = attention_mass_over_keys(future[index], keys, target_stop=history)
            total = part if total is None else total + part
        target = cache["rb_future_by_source"]
        target.index_copy_(0, ids.to(target.device), total.to(target.device))
        cache["rb_future_qvec"].copy_(future.float().mean(dim=1).to(cache["rb_future_qvec"].device))


def future_query_proxy(pre_rope_query: Any, *, grid_sizes: Any, freqs: Any, start_frame: int, frame_tokens: int, horizon: int, samples: int) -> Any:
    """Future Forcing proxy: mean pre-RoPE query of the chunk, RoPE-rotated to t+1..t+horizon.

    Returns ``[horizon, samples, heads, dim]`` sampled proxy rows.
    """
    import torch
    from wan.modules.model_fast import causal_rope_apply

    from ..methods.rebuttal_kernels import sampled_query_rows

    tokens = int(pre_rope_query.shape[1])
    frames = max(1, tokens // int(frame_tokens))
    mean_query = pre_rope_query[0, :frames * int(frame_tokens)].float().reshape(frames, int(frame_tokens), *pre_rope_query.shape[2:]).mean(dim=0)
    height, width = int(grid_sizes[0][1]), int(grid_sizes[0][2])
    grid = torch.tensor([[1, height, width]], dtype=torch.long)
    rows = sampled_query_rows(int(frame_tokens), int(samples), device=pre_rope_query.device)
    last_frame = int(start_frame) + frames - 1
    rotated = []
    for offset in range(1, int(horizon) + 1):
        value = causal_rope_apply(mean_query[None], grid, freqs, start_frame=last_frame + offset)[0]
        rotated.append(value.index_select(0, rows))
    return torch.stack(rotated).to(pre_rope_query.dtype)

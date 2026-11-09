from __future__ import annotations

import math
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from cmbench_rebuild.config import RunConfig
from cmbench_rebuild.core.metrics import active_history_token_head_units_before_current
from cmbench_rebuild.methods import rebuttal_kernels as kernels
from cmbench_rebuild.methods.registry import resolve
from cmbench_rebuild.runtime.cache_attention import cached_attention
from cmbench_rebuild.runtime.cache_factory import CacheGeometry, allocate_case_cache
from cmbench_rebuild.runtime.online_policies import OnlinePolicyRuntime

NEW_METHODS = ("contextfree", "snapkv", "focused_forcing", "worldkv", "future_forcing", "densitykv")


def sdpa(q, k, v):
    out = torch.nn.functional.scaled_dot_product_attention(q.transpose(1, 2).float(), k.transpose(1, 2).float(), v.transpose(1, 2).float())
    return out.transpose(1, 2).to(q.dtype)


# --------------------------------------------------------------------- kernels
def test_attention_mass_matches_naive_softmax():
    g = torch.Generator().manual_seed(0)
    q = torch.randn(5, 3, 8, generator=g)
    k = torch.randn(37, 3, 8, generator=g)
    logits = torch.einsum("shd,nhd->shn", q, k) / math.sqrt(8)
    naive = torch.softmax(logits, dim=-1).sum(dim=(0, 1))[:20]
    blocked = kernels.attention_mass_over_keys(q, k, target_stop=20, block_tokens=7)
    assert torch.allclose(blocked, naive, atol=1e-5)


def test_top_fraction_and_pooling_shapes():
    scores = torch.tensor([0.1, 0.9, 0.5, 0.7, 0.2, 0.3])
    mask = kernels.top_fraction_mask(scores, keep_ratio=0.5)
    assert mask.tolist() == [False, True, True, True, False, False]
    pooled = kernels.spatial_pool_chunk_scores(torch.arange(12.0), frames=2, height=2, width=3, kernel=3)
    assert pooled.shape == (12,)


def test_worldkv_anchor_score_is_mean_cosine():
    g = torch.Generator().manual_seed(1)
    keys = torch.randn(4 * 6, 2, 4, generator=g)
    scores = kernels.worldkv_anchor_scores(keys, frame_tokens=6)
    unit = torch.nn.functional.normalize(keys, dim=-1)
    naive = torch.einsum("nhd,ahd->nah", unit[6:], unit[:6]).mean(dim=1).mean(dim=1)
    assert torch.allclose(scores, naive, atol=1e-6)


def test_densitykv_respects_capacity_and_growth_limit():
    g = torch.Generator().manual_seed(2)
    bank = torch.randn(10, 2, 4, generator=g)
    density = kernels.head_mean_riesz(bank, bank, scale=1.0, power=2.0, eps=1.0).sum(1) - 1.0
    candidates = torch.cat([bank[:4] + 0.01 * torch.randn(4, 2, 4, generator=g), 5 * torch.randn(8, 2, 4, generator=g)])
    result = kernels.densitykv_admit(candidates, bank, density, density.clamp_min(1e-6), capacity=14, scale=1.0)
    kept_bank = 10 - int(result["evicted"].numel())
    assert kept_bank + int(result["admitted"].numel()) <= 14
    # near-duplicates of existing entries are the most redundant, so they are admitted last
    assert set(result["admitted"].tolist()) >= {4, 5, 6, 7}


def test_future_merge_routes_only_similar_tokens():
    keys = torch.zeros(4, 1, 2)
    keys[0, 0] = torch.tensor([1.0, 0.0])
    keys[1, 0] = torch.tensor([0.99, 0.05])
    keys[2, 0] = torch.tensor([0.0, 1.0])
    keys[3, 0] = torch.tensor([-1.0, 0.0])
    keep = torch.tensor([True, False, True, False])
    proxies = torch.tensor([[[1.0, 0.0]], [[0.0, 1.0]]])
    target = kernels.future_merge_plan(keys, keep, proxies, threshold=0.95)
    assert target.tolist() == [-1, 0, -1, -1]
    values = torch.arange(4.0).reshape(4, 1, 1)
    merged = kernels.merge_values(values, target, torch.ones(4))
    assert merged[0, 0, 0].item() == pytest.approx(0.5)


# ------------------------------------------------------------------- lifecycle
GEOMETRY = CacheGeometry(2, 2, 4, 1, 0, 16, 4, 2, 6)


def _runtime(method, parameters, cache):
    return OnlinePolicyRuntime(
        cache=cache, geometry=GEOMETRY, policy=SimpleNamespace(name=method, parameters=parameters),
        request=SimpleNamespace(seed=2, raw={}, generation={"generation_kv_policy": "method-native"}),
        pipeline=SimpleNamespace(), timesteps=torch.arange(4), emit=lambda row: None,
    )


def _drive(method, parameters, chunks, *, attender=lambda q, k, v: q, future=False):
    cache = allocate_case_cache(GEOMETRY, method=method, parameters=parameters, dtype=torch.float32, device="cpu")
    runtime = _runtime(method, parameters, cache)
    tokens = GEOMETRY.tokens_per_chunk
    outputs = []
    for chunk in range(chunks):
        phase = "context" if chunk < GEOMETRY.context_frames // GEOMETRY.chunk_size else "generation"
        runtime.prepare_chunk(chunk)
        units = active_history_token_head_units_before_current(cache, current_noisy_tokens=tokens)
        runtime.before_clean(chunk, phase=phase)
        for layer_idx, layer in enumerate(cache):
            g = torch.Generator().manual_seed(100 * layer_idx + chunk)
            q, k, v = (torch.randn(1, tokens, 2, 4, generator=g) for _ in range(3))
            aux = {"future_queries": torch.randn(int(parameters.get("horizon", 1)), 3, 2, 4, generator=g)} if future else None
            outputs.append(cached_attention(q, k, v, cache=layer, current_start=chunk * tokens, frame_tokens=GEOMETRY.frame_tokens,
                                            local_attn_size=-1, sink_size=0, max_attention_size=10**9, attender=attender, aux=aux))
        runtime.after_clean(chunk_idx=chunk, clean_x=torch.zeros(1), context_scores=None, phase=phase)
        outputs.append(units)
    return cache, runtime, outputs


def _live_ids(layer):
    live = int(layer["local_end_index"].item())
    return layer["source_ids"][0, :live].tolist()


def test_every_new_method_is_registered_and_configurable():
    for name in NEW_METHODS:
        spec = resolve(name)
        cfg = RunConfig(protocol="p", method=name, case_id="c", seed=2, checkpoint="x", dataset_version="d", request_file="r")
        assert cfg.normalized()["method"] == spec.id


@pytest.mark.parametrize("method", ["snapkv", "future_forcing"])
def test_attention_selectors_keep_ratio_per_layer(method):
    parameters = dict(resolve(method).defaults) | {"keep_ratio": 0.25, "query_samples": 3, "pool_kernel": 1}
    cache, runtime, _ = _drive(method, parameters, chunks=8, future=method == "future_forcing")
    tokens = GEOMETRY.tokens_per_chunk
    for layer in cache:
        ids = _live_ids(layer)
        assert ids == sorted(ids)
        per_chunk = [sum(1 for value in ids if value // tokens == chunk) for chunk in range(8)]
        assert per_chunk[0] == tokens  # dense sink
        assert per_chunk[6] == tokens and per_chunk[7] == tokens  # recent window still dense
        assert all(count == round(0.25 * tokens) for count in per_chunk[1:6])
    assert runtime.rebuttal.diagnostics()["finalized_chunks"] == 5


def test_densitykv_bank_is_bounded():
    parameters = dict(resolve("densitykv").defaults) | {"max_entries": 15, "density_scale": 1.0}
    cache, runtime, _ = _drive("densitykv", parameters, chunks=10)
    tokens = GEOMETRY.tokens_per_chunk
    for layer in cache:
        ids = _live_ids(layer)
        bank = [value for value in ids if tokens <= value < 8 * tokens]
        assert len(bank) <= 15
    assert runtime.rebuttal.diagnostics()["finalized_chunks"] == 7


def test_worldkv_compression_and_bounded_visibility():
    parameters = dict(resolve("worldkv").defaults) | {"retrieval": "query", "retrieved_chunks": 2, "query_samples": 3}
    cache, runtime, outputs = _drive("worldkv", parameters, chunks=8, attender=sdpa)
    tokens, frame = GEOMETRY.tokens_per_chunk, GEOMETRY.frame_tokens
    stored_size = frame + round(0.25 * (tokens - frame))
    for layer in cache:
        ids = _live_ids(layer)
        for chunk in range(1, 6):
            chunk_ids = [value for value in ids if value // tokens == chunk]
            assert len(chunk_ids) == stored_size
            assert set(range(chunk * tokens, chunk * tokens + frame)) <= set(chunk_ids)  # anchor kept
    visible = runtime.rebuttal.visible
    assert len(visible.stored_chunks) == 5
    assert all(len(row["selected"]) <= 2 for row in visible.selected_log)
    # accounted visible history at the last chunk: sink + window-dense + 2 retrieved stored chunks
    last_units = outputs[-1]
    assert last_units == len(cache) * 2 * (3 * tokens + 2 * stored_size)


def test_focused_forcing_equals_dense_attention_when_budget_covers_history():
    parameters = dict(resolve("focused_forcing").defaults) | {"frame_budget": 100, "sink_frames": 1, "recent_frames": 1}
    _, _, sparse = _drive("focused_forcing", parameters, chunks=5, attender=sdpa)
    _, _, dense = _drive("fullkv", {}, chunks=5, attender=sdpa)
    for a, b in zip(sparse, dense):
        if isinstance(a, int):
            continue
        assert torch.allclose(a, b, atol=1e-5)


def test_focused_forcing_limits_frames_per_query_frame():
    parameters = dict(resolve("focused_forcing").defaults) | {"frame_budget": 4, "sink_frames": 1, "recent_frames": 1}
    cache, runtime, outputs = _drive("focused_forcing", parameters, chunks=8, attender=sdpa)
    assert all(torch.isfinite(t).all() for t in outputs if not isinstance(t, int))
    # every layer stores the full history (sparse attention, no eviction)
    assert all(int(layer["local_end_index"].item()) == 8 * GEOMETRY.tokens_per_chunk for layer in cache)
    assert outputs[-1] == len(cache) * 2 * 4 * GEOMETRY.frame_tokens


def test_contextfree_uses_the_no_cache_probe_score():
    parameters = dict(resolve("contextfree").defaults)
    cache = allocate_case_cache(GEOMETRY, method="contextfree", parameters=parameters, dtype=torch.float32, device="cpu")
    runtime = _runtime("contextfree", parameters, cache)
    assert runtime.is_consistency
    calls = []

    def model(*, x, t, kv_cache, current_start, **unused):
        calls.append(kv_cache)
        return [torch.zeros_like(x[0])]

    runtime.pipeline = SimpleNamespace(model=model, scheduler=SimpleNamespace(), _convert_flow_pred_to_x0=lambda flow, noisy, step, scheduler: flow + 1.0)
    noisy = torch.zeros(16, 2, 2, 6)
    runtime.capture_generation_prediction(2, noisy)
    runtime.generation_probe(step_index=2, noisy=noisy, step=torch.tensor(1.0), kwargs={"kv_cache": cache, "current_start": 0})
    assert len(calls) == 1 and calls[0] is runtime._isolated_probe_cache
    assert torch.equal(runtime._generation_prediction, noisy + 1.0)
    assert torch.equal(runtime._generation_reference, noisy)  # with-cache prediction kept for budget matching


def test_contextfree_keeps_decoprune_count_but_ranks_by_its_own_score():
    from cmbench_rebuild.runtime.online_policies import ConsistencyRecord

    parameters = dict(resolve("contextfree").defaults) | {"threshold": 0.5}
    cache = allocate_case_cache(GEOMETRY, method="contextfree", parameters=parameters, dtype=torch.float32, device="cpu")
    runtime = _runtime("contextfree", parameters, cache)
    n = GEOMETRY.tokens_per_chunk
    reference = torch.zeros(n); reference[:3] = 1.0          # DeCoPrune would keep 3 tokens (the first three)
    own = torch.arange(n, dtype=torch.float32)               # context-free score prefers the LAST tokens
    record = ConsistencyRecord(1, torch.arange(n, 2 * n), own, "context", reference_scores=reference)
    mask = runtime._consistency_mask(record)
    assert int(mask.sum()) == 3 and mask[-3:].all()


def test_sdpa_fallback_matches_masked_softmax():
    from cmbench_rebuild.runtime.attention_fallback import sdpa_flash_attention

    g = torch.Generator().manual_seed(3)
    q, k, v = torch.randn(2, 5, 4, 8, generator=g), torch.randn(2, 7, 4, 8, generator=g), torch.randn(2, 7, 4, 8, generator=g)
    k_lens = torch.tensor([7, 3])
    out = sdpa_flash_attention(q, k, v, k_lens=k_lens, dtype=torch.float32)
    logits = torch.einsum("bqhd,bkhd->bhqk", q, k) / math.sqrt(8)
    logits[1, :, :, 3:] = -float("inf")
    ref = torch.einsum("bhqk,bkhd->bqhd", torch.softmax(logits, dim=-1), v)
    assert torch.allclose(out.float(), ref, atol=1e-5)

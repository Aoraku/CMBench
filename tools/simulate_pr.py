#!/usr/bin/env python3
"""Exact seqPR of content-independent budgets at the real CMBench geometry (CPU, no model).

For SnapKV, FutureForcing, WorldKV and FocusedForcing the retained/visible token
count depends only on the cache lifecycle and the budget, not on video content.
This replays that lifecycle (one layer, one head, random keys) for a context of
``--context-frames`` latent frames and ``--output-frames`` generated frames and
reports seqPR exactly as the runtime's ledger computes it.  Use it to set budgets
before a GPU run.  (ContextFree is matched to DeCoPrune chunk by chunk; DensityKV
depends on content and must be measured on a real run.)

    python tools/simulate_pr.py --methods snapkv future_forcing worldkv focused_forcing \
        --params '{"snapkv": {"keep_ratio": 0.10}}'
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch  # noqa: E402

from cmbench_rebuild.core.metrics import active_history_token_head_units_before_current  # noqa: E402
from cmbench_rebuild.methods.registry import resolve  # noqa: E402
from cmbench_rebuild.runtime.cache_attention import cached_attention  # noqa: E402
from cmbench_rebuild.runtime.cache_factory import CacheGeometry, allocate_case_cache  # noqa: E402
from cmbench_rebuild.runtime.online_policies import OnlinePolicyRuntime  # noqa: E402


def simulate(method: str, params: dict, *, context_frames: int, output_frames: int, frame_tokens: int, chunk: int) -> dict:
    geometry = CacheGeometry(1, 1, 2, 1, 0, context_frames, output_frames, chunk, frame_tokens)
    parameters = dict(resolve(method).defaults) | params
    if method == "worldkv":
        parameters["retrieval"] = "query"  # same visible count as camera retrieval
    if "query_samples" in parameters:
        parameters["query_samples"] = 8
    cache = allocate_case_cache(geometry, method=method, parameters=parameters, dtype=torch.float32, device="cpu")
    runtime = OnlinePolicyRuntime(cache=cache, geometry=geometry, policy=SimpleNamespace(name=method, parameters=parameters),
                                  request=SimpleNamespace(seed=2, raw={}, generation={"generation_kv_policy": "method-native"}),
                                  pipeline=SimpleNamespace(), timesteps=torch.arange(4), emit=lambda row: None)
    tokens = geometry.tokens_per_chunk
    context_chunks, total_chunks = context_frames // chunk, (context_frames + output_frames) // chunk
    active = dense = 0
    generator = torch.Generator().manual_seed(0)
    for index in range(total_chunks):
        phase = "context" if index < context_chunks else "generation"
        runtime.prepare_chunk(index)
        if phase == "generation":
            active += active_history_token_head_units_before_current(cache, current_noisy_tokens=tokens)
            dense += index * tokens
        runtime.before_clean(index, phase=phase)
        q, k, v = (torch.randn(1, tokens, 1, 2, generator=generator) for _ in range(3))
        aux = {"future_queries": torch.randn(int(parameters.get("horizon", 1)), 8, 1, 2, generator=generator)} if method == "future_forcing" else None
        cached_attention(q, k, v, cache=cache[0], current_start=index * tokens, frame_tokens=frame_tokens, local_attn_size=-1,
                         sink_size=0, max_attention_size=10**12, attender=lambda a, b, c: a, aux=aux)
        runtime.after_clean(chunk_idx=index, clean_x=torch.zeros(1), context_scores=None, phase=phase)
    return {"method": method, "params": params, "seqPR": 1.0 - active / dense}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--methods", nargs="+", default=["snapkv", "future_forcing", "worldkv", "focused_forcing"])
    parser.add_argument("--params", type=json.loads, default={})
    parser.add_argument("--context-frames", type=int, default=244)
    parser.add_argument("--output-frames", type=int, default=16)
    parser.add_argument("--frame-tokens", type=int, default=1560)
    parser.add_argument("--chunk", type=int, default=4)
    args = parser.parse_args()
    for method in args.methods:
        print(json.dumps(simulate(method, args.params.get(method, {}), context_frames=args.context_frames,
                                  output_frames=args.output_frames, frame_tokens=args.frame_tokens, chunk=args.chunk)))


if __name__ == "__main__":
    main()

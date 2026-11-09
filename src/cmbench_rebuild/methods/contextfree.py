"""Context-free control: DeCoPrune's step-to-final discrepancy scored WITHOUT the history cache.

Identical lifecycle, threshold rule and physical compaction as DeCoPrune; the only
change is the probe that produces the score.  The clean prediction at the probe
timestep is computed with an empty KV cache (no sink, no history), so the score
reflects intrinsic denoising difficulty rather than redundancy with the cache.
Online generation therefore needs one extra no-cache model call per chunk.
"""
from .spec import MethodSpec

SPEC = MethodSpec(
    "contextfree",
    "ContextFree",
    "contextfree_reindex",
    "consistency_contextfree",
    {
        "step_index": 2,
        "generation_step_index": 2,
        "threshold": 0.10,
        "score_type": "mse",
        "local_window_chunks": 1,
        "min_keep_per_spatial_block": 0,
        "spatial_floor_threshold": float("-inf"),
    },
)

"""Future Forcing (Luo et al., 2026), reimplemented from the paper (no public code).

A future-query proxy is the mean pre-RoPE query of the latest chunk, rotated to
the next ``horizon`` frame positions.  Each candidate token's score is
``future_weight * future_attention + (1 - future_weight) * historical_attention``.
When a chunk leaves the recent window each layer keeps the top ``keep_ratio``
tokens and, if ``merge`` is set, routes every evicted token to its most similar
kept token (cosine of the future-query projections >= ``merge_threshold``) and
merges its value weighted by future attention mass.
"""
from .spec import MethodSpec

SPEC = MethodSpec(
    "future_forcing",
    "FutureForcing",
    "future_forcing_reindex",
    "attention_future_forcing",
    {
        "keep_ratio": 0.146,
        "horizon": 6,
        "future_weight": 0.5,
        "merge": True,
        "merge_threshold": 0.95,
        "query_samples": 256,
        "local_window_chunks": 1,
    },
)

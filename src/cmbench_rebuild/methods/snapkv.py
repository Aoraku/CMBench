"""Attention-based selector in the style of SnapKV / H2O.

When a chunk leaves the protected recent window, each layer keeps the
``keep_ratio`` fraction of that chunk's tokens that received the most softmax
attention from the observation queries of the chunks that followed it
(``query_samples`` evenly spaced rows of every clean pass).  Scores are averaged
over heads and smoothed with SnapKV's pooling on each frame's 2-D token grid.
Selection is per layer; the sink chunk and the recent window stay dense, exactly
as for DeCoPrune.
"""
from .spec import MethodSpec

SPEC = MethodSpec(
    "snapkv",
    "SnapKV",
    "snapkv_reindex",
    "attention_snapkv",
    {
        "keep_ratio": 0.146,
        "query_samples": 256,
        "pool_kernel": 3,
        "local_window_chunks": 1,
    },
)

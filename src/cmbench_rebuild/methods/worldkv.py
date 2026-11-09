"""WorldKV (Yi et al., 2026): world compression plus world retrieval.

Compression (paper Eq. 5): when a chunk leaves the recent window, each layer keeps
the chunk's first (anchor) frame plus the ``nonanchor_keep_ratio`` non-anchor
tokens with the lowest mean cosine similarity to the anchor keys.

Retrieval: compressed chunks stay in the store, but each attention call only sees
the sink chunk(s), the recent window and ``retrieved_chunks`` stored chunks.
``retrieval="camera"`` ranks stored chunks by rotation distance between their
estimated camera pose and the current chunk's target pose (context poses are
recovered from the observed video); when no robust context trajectory can be
estimated, or with ``retrieval="query"``, chunks are ranked by attention of the
current queries to each chunk's mean key (the paper's query-based alternative).
"""
from .spec import MethodSpec

SPEC = MethodSpec(
    "worldkv",
    "WorldKV",
    "worldkv_reindex",
    "worldkv_retrieval",
    {
        "nonanchor_keep_ratio": 0.25,
        "retrieved_chunks": 14,
        "retrieval": "camera",
        "sink_chunks": 1,
        "recent_chunks": 1,
        "local_window_chunks": 1,
        "query_samples": 256,
    },
)

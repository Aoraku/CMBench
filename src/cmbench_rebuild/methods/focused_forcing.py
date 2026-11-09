"""Focused Forcing (Cai et al., 2026), reimplemented from the paper (code not released).

Sparse attention over a fully stored history.  For every query frame of the
current chunk and every head, historical frames are ranked by
``S = attention_weight * z(A) + (1 - attention_weight) * z(D)`` where ``A`` is the
pooled query-key frame score (paper Eq. 8) and ``D`` the negative cosine between a
frame's key and the mean historical key (Eq. 9-10).  The head attends to its
reserved sink and recent frames plus the top-scoring frames up to
``frame_budget`` frames.  Port notes: head budgets are uniform (the paper's offline
head-importance calibration needs one masked rollout per layer-head pair), and the
diversity term uses frame-mean keys.
"""
from .spec import MethodSpec

SPEC = MethodSpec(
    "focused_forcing",
    "FocusedForcing",
    "focused_forcing_reindex",
    "sparse_focused_forcing",
    {
        "frame_budget": 36,
        "attention_weight": 0.5,
        "sink_frames": 4,
        "recent_frames": 4,
        "head_group_size": 8,
    },
)

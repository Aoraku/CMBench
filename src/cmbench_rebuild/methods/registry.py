"""Discover the method modules by their small public specs."""
from __future__ import annotations

from .consistency_prune import SPEC as CONSISTENCY_PRUNE
from .consistency_prune_hs import SPEC as CONSISTENCY_PRUNE_HS
from .contextfree import SPEC as CONTEXTFREE
from .densitykv import SPEC as DENSITYKV
from .focused_forcing import SPEC as FOCUSED_FORCING
from .future_forcing import SPEC as FUTURE_FORCING
from .snapkv import SPEC as SNAPKV
from .worldkv import SPEC as WORLDKV
from .dummy_forcing import SPEC as DUMMY_FORCING
from .forcingkv import SPEC as FORCINGKV
from .fullkv import SPEC as FULLKV
from .patchification import SPEC as PATCHIFICATION
from .random import SPEC as RANDOM
from .spec import MethodSpec
from .streaming import SPEC as STREAMING


_SPECS = (
    FULLKV,
    STREAMING,
    DUMMY_FORCING,
    FORCINGKV,
    PATCHIFICATION,
    RANDOM,
    CONSISTENCY_PRUNE,
    CONSISTENCY_PRUNE_HS,
    CONTEXTFREE,
    SNAPKV,
    FOCUSED_FORCING,
    WORLDKV,
    FUTURE_FORCING,
    DENSITYKV,
)
REGISTRY = {spec.id: spec for spec in _SPECS}
LEGACY_REGISTRY = {spec.legacy_id: spec for spec in _SPECS}

ALIASES = {"consistency_prune": "decoprune", "consistency_prune_hs": "decoprune_hs",
           "DeCoPrune": "decoprune", "DeCoPrune-HS": "decoprune_hs",
           "context_free": "contextfree", "attention": "snapkv", "focusedforcing": "focused_forcing",
           "futureforcing": "future_forcing", "density_kv": "densitykv", "world_kv": "worldkv"}

def resolve(name: str) -> MethodSpec:
    name = ALIASES.get(name, name)
    try:
        return REGISTRY.get(name) or LEGACY_REGISTRY[name]
    except KeyError as exc:
        raise ValueError(f"unknown method {name!r}; choices: {', '.join(REGISTRY)}") from exc

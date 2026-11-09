"""DensityKV (Zhao et al., 2026), ported from the public release's bank update.

Every completed chunk that leaves the recent window is offered to a bounded
per-layer historical bank.  Candidates are admitted in order of how little they
raise existing entries' Soft-Riesz density over post-RoPE keys, under the
published density-growth limit; capacity overflow evicts growth violators and
then the densest entries.  The sink chunk and recent window stay dense.
Port note: densities are averaged over heads so all heads of a layer share one
token set (the runtime's dense bank has one token identity per layer).
"""
from .spec import MethodSpec

SPEC = MethodSpec(
    "densitykv",
    "DensityKV",
    "densitykv_reindex",
    "density_bank",
    {
        "max_entries": 54000,
        "density_scale": 8.0,
        "riesz_power": 2.0,
        "riesz_eps": 1.0,
        "growth_limit": 2.0,
        "local_window_chunks": 1,
        "work_chunk_size": 512,
    },
)

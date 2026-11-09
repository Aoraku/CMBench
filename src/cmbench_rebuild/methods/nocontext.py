"""No-context lower bound: generation keeps none of the observed context's KV.

The context is replayed as usual (keeping only the most recent chunk so the replay
stays causal and cheap), and when generation starts every context token is dropped,
including the sink.  Generated chunks keep their own KV densely.  seqPR is therefore
close to 1 and the score shows how much the benchmark depends on remembered context.
"""
from .spec import MethodSpec

SPEC = MethodSpec("nocontext", "NoContext", "nocontext_reindex", "no_context_memory", {})

"""No-context lower bound: no prefill of the observed context.

Only the final observed chunk (the frames the continuation starts from) is written to
the KV cache; the rest of the context is never encoded into memory.  Generation then
starts from that chunk and the text prompt, and generated chunks are cached as usual
(streaming).  seqPR is therefore close to 1.
"""
from .spec import MethodSpec

SPEC = MethodSpec("nocontext", "NoContext", "nocontext_reindex", "no_context_memory", {})

"""Quality-of-life helpers: derived values the UI and the doctor show about a launch plan.

The effective context is what a request can actually use: with a unified KV cache the whole context is shared by
every slot, while with separate KV each slot gets about ``context / parallel`` tokens. These numbers are read-only
and never change the command line; they only explain it.
"""

from __future__ import annotations

from typing import Any


def effective_context(context: int, parallel: int, kv_unified: bool) -> dict[str, Any]:
    """Per-slot context and a one-line note describing the effective context semantics."""
    parallel = max(1, int(parallel or 1))
    context = max(1, int(context))
    if kv_unified or parallel == 1:
        per_slot = context
        if parallel > 1:
            note = f"{context:,} tokens total, shared by all {parallel} slots through one unified KV cache"
        else:
            note = f"{context:,} tokens total"
    else:
        per_slot = max(1, context // parallel)
        leftover = context % parallel
        if leftover:
            note = f"about {per_slot:,} tokens per slot ({context:,} tokens total, separate KV per slot)"
        else:
            note = f"{per_slot:,} tokens per slot ({context:,} tokens total, separate KV per slot)"
    return {
        "context": context,
        "parallel": parallel,
        "kv_unified": bool(kv_unified),
        "per_slot": per_slot,
        "note": note,
    }

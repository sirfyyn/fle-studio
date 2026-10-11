"""Checking a result: KL divergence and top-1 agreement against the full-precision model, from `llama-perplexity`.

Two runs on the same text with the same context and chunk count:
  1. the reference model with `--kl-divergence-base FILE` stores its token probabilities,
  2. the candidate with `--kl-divergence --kl-divergence-base FILE` compares itself with them and prints statistics.

This module builds the two command lines, parses the statistics block and turns it into a result card. The stored
probabilities of the first run are large (they grow with the number of tokens times the vocabulary), so the default
text is short; use the same text, context and chunks for every candidate you want to compare, and the same build of
the tool on the same hardware, because a different order of arithmetic moves the numbers slightly.
"""

from __future__ import annotations

import re
from typing import Any

from .. import catalog

PM = r"(?:±|\+/-|Â±|�+)"  # the plus-minus sign, also as it arrives from a console in another code page
NUM = r"(-?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?|nan|-?inf)"


def _find(pattern: str, text: str) -> re.Match[str] | None:
    return re.search(pattern, text, re.M)


def _f(value: str) -> float | None:
    try:
        v = float(value)
    except ValueError:
        return None
    return v if v == v and abs(v) != float("inf") else None


def parse(text: str) -> dict[str, Any]:
    """Statistics of one comparison run. Missing parts are simply absent; `complete` says whether the KLD is there."""
    out: dict[str, Any] = {}

    def pair(key: str, label: str) -> None:
        m = _find(rf"^\s*{label}\s*:\s*{NUM}\s*(?:{PM}\s*{NUM})?", text)
        if m:
            out[key] = _f(m.group(1))
            if m.group(2) is not None:
                out[key + "_se"] = _f(m.group(2))

    pair("kld_mean", r"Mean\s+KLD")
    pair("kld_max", r"Maximum\s+KLD")
    pair("kld_999", r"99\.9%\s+KLD")
    pair("kld_99", r"99\.0%\s+KLD")
    pair("kld_95", r"95\.0%\s+KLD")
    pair("kld_median", r"Median\s+KLD")
    pair("ppl_q", r"Mean\s+PPL\(Q\)")
    pair("ppl_base", r"Mean\s+PPL\(base\)")
    pair("ppl_ratio", r"Mean\s+PPL\(Q\)/PPL\(base\)")
    pair("ln_ppl_ratio", r"Mean\s+ln\(PPL\(Q\)/PPL\(base\)\)")
    pair("delta_p_mean", r"Mean\s+Δp")
    pair("delta_p_rms", r"RMS\s+Δp")
    pair("top1", r"Same\s+top\s+p")
    m = _find(rf"^\s*Cor\(ln\(PPL\(Q\)\),\s*ln\(PPL\(base\)\)\)\s*:\s*{NUM}\s*%", text)
    if m:
        out["ppl_correlation"] = _f(m.group(1))
    m = _find(r"tokens\s*[:=]\s*(\d+)|(\d+)\s+tokens", text)
    if m:
        out["tokens"] = int(m.group(1) or m.group(2))
    out["complete"] = out.get("kld_mean") is not None
    return out


def reference_cards() -> list[dict[str, Any]]:
    """The published FLE figures for orientation. They were measured on the FLE project's own texts, not on yours."""
    cards = []
    for m in catalog.CATALOG:
        if m.kld:
            cards.append({"name": m.name, "kld": list(m.kld), "size": m.size, "texts": "E / W / K", "source": m.repo})
    return cards


def card(
    stats: dict[str, Any],
    *,
    name: str,
    size: int,
    params: int | None,
    reference: str,
    text: str,
    context: int,
    chunks: int,
    tool_version: str,
) -> dict[str, Any]:
    """The result card: what was measured, against what, and how big the file is."""
    bpw = round(size * 8 / params, 3) if params else None
    top1 = stats.get("top1")
    return {
        "name": name,
        "complete": bool(stats.get("complete")),
        "size": size,
        "params": params,
        "bpw": bpw,
        "kld": stats.get("kld_mean"),
        "kld_se": stats.get("kld_mean_se"),
        "kld_99": stats.get("kld_99"),
        "kld_max": stats.get("kld_max"),
        "top1": None if top1 is None else top1 / 100.0,
        "top1_se": None if stats.get("top1_se") is None else stats["top1_se"] / 100.0,
        "ppl_ratio": stats.get("ppl_ratio"),
        "ppl_q": stats.get("ppl_q"),
        "ppl_base": stats.get("ppl_base"),
        "reference": reference,
        "text": text,
        "context": context,
        "chunks": chunks,
        "tool": tool_version,
        "note": "Compare only runs with the same text, context, chunks, tool build and hardware.",
    }

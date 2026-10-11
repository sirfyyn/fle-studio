"""Forge: build your own quant with the FLE recipe, from inside FLE Studio.

ggufio   reading and writing GGUF files (no dependencies)
recipe   the recipe JSON (schema 1)
tools    finding llama-imatrix, llama-quantize, llama-perplexity, llama-fle-reap
plan     the recipe calculator and the memory preview
build    writing a model from pieces of other models (hybrid build), resumable
reap     expert pruning
verify   KL divergence results
corpus   calibration text
jobs     background jobs with steps, log, cancel and resume
steps    the step implementations
api      what the server calls
"""

from __future__ import annotations

SCHEMA = 1

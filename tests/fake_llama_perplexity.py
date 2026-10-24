#!/usr/bin/env python3
"""A stand-in for ``llama-perplexity`` (llama.cpp tools/perplexity) for tests and CI. Standard library only.

    fake_llama_perplexity.py -m MODEL.gguf -f TEXT.txt --kl-divergence-base BASE.kld [--kl-divergence]
                             [-c N] [--chunks N] [-t N] [-ngl N]

Without ``--kl-divergence`` it plays the reference run: it "saves the output probabilities" (a small JSON file) to the
base file. With it, it reads that file and prints the three statistics blocks in the shape of llama.cpp: perplexity,
KL divergence and token probabilities. The numbers are made up but deterministic and sensible: the KL divergence
falls as the bits per weight in the model's file name rise (Q2_K ... Q8_0, BF16), the reference itself reads 0.

Environment: FAKE_PPL_CHUNK_SECONDS (default 0.02) time per chunk; FAKE_PPL_FAIL=1 exit with an error before the report.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
import re
import sys
import time
from pathlib import Path

BPW = (
    ("IQ2_XXS", 2.06),
    ("IQ2_XS", 2.31),
    ("IQ2_S", 2.5),
    ("IQ2_M", 2.7),
    ("IQ3_XXS", 3.06),
    ("IQ3_XS", 3.3),
    ("IQ3_S", 3.44),
    ("IQ3_M", 3.66),
    ("IQ4_NL", 4.5),
    ("IQ4_XS", 4.25),
    ("Q2_K", 2.96),
    ("Q3_K_S", 3.5),
    ("Q3_K_M", 3.9),
    ("Q4_K_S", 4.6),
    ("Q4_K_M", 4.85),
    ("Q5_K_M", 5.7),
    ("Q6_K", 6.56),
    ("Q8_0", 8.5),
    ("BF16", 16.0),
    ("F16", 16.0),
)


def bits_per_weight(model: str) -> float:
    name = Path(model).name.upper()
    for token, bpw in BPW:
        if re.search(rf"(^|[^A-Z0-9]){token}($|[^A-Z0-9])", name):
            return bpw
    return 4.85


def parse(argv: list[str]) -> dict[str, str]:
    opts: dict[str, str] = {}
    i = 0
    while i < len(argv):
        token = argv[i]
        i += 1
        if token == "--kl-divergence":
            opts["kl"] = "1"
        elif token.startswith("-") and i < len(argv) and not argv[i].startswith("-"):
            opts[token] = argv[i]
            i += 1
    return opts


def pct(values: list[float], q: float) -> float:
    return values[min(len(values) - 1, max(0, int(q * (len(values) - 1) + 0.5)))]


def main(argv: list[str]) -> int:
    opts = parse(argv)
    model, text_file, base_file = opts.get("-m", ""), opts.get("-f", ""), opts.get("--kl-divergence-base", "")
    if not model or not Path(model).is_file():
        print(f"error: failed to load model '{model}'", file=sys.stderr)
        return 1
    if not text_file or not Path(text_file).is_file():
        print(f"error: failed to open file '{text_file}'", file=sys.stderr)
        return 1
    ctx = int(opts.get("-c", "512"))
    text = Path(text_file).read_bytes()
    n_tokens = max(1, len(text) // 4)
    n_chunks = n_tokens // ctx
    if int(opts.get("--chunks", "0")) > 0:
        n_chunks = min(n_chunks, int(opts["--chunks"]))
    if n_chunks < 1:
        print(f"error: not enough tokens ({n_tokens}) for one chunk of {ctx}", file=sys.stderr)
        return 1
    bpw = bits_per_weight(model)
    delay = float(os.environ.get("FAKE_PPL_CHUNK_SECONDS", "0.02"))
    print(f"system_info: n_threads = {opts.get('-t', '4')}", file=sys.stderr)
    print(
        f"perplexity: tokenizing the input ..\nperplexity: tokenization took 1.0 ms\n"
        f"perplexity: calculating perplexity over {n_chunks} chunks, n_ctx={ctx}",
        file=sys.stderr,
    )
    sha = hashlib.sha256(text).hexdigest()
    if "kl" not in opts:
        # the reference run: print the running perplexity and save "the logits"
        for i in range(n_chunks):
            time.sleep(delay)
            print(f"[{i + 1}]{5.0 + 0.1 * (i % 3):.4f},", end="", flush=True)
        print()
        if os.environ.get("FAKE_PPL_FAIL") == "1":
            print("error: the reference run failed", file=sys.stderr)
            return 1
        Path(base_file).write_text(
            json.dumps({"model": model, "bpw": bpw, "chunks": n_chunks, "ctx": ctx, "text_sha256": sha}),
            encoding="utf-8",
        )
        print(f"\nFinal estimate: PPL = {5.1:.4f} +/- 0.31000")
        return 0

    try:
        base = json.loads(Path(base_file).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        print(f"error: failed to open '{base_file}' (run the reference model first)", file=sys.stderr)
        return 1
    if base.get("text_sha256") != sha or base.get("chunks") != n_chunks or base.get("ctx") != ctx:
        print("error: the saved reference does not match this text, chunk size or chunk count", file=sys.stderr)
        return 1
    if os.environ.get("FAKE_PPL_FAIL") == "1":
        print("error: the candidate run failed", file=sys.stderr)
        return 1
    rng = random.Random(int(hashlib.sha256(Path(model).name.encode()).hexdigest()[:8], 16))
    scale = 0.0 if abs(bpw - base["bpw"]) < 1e-9 else 0.8 * math.exp(-0.9 * (min(bpw, base["bpw"]) - 1.0))
    if bpw > base["bpw"]:
        scale = 0.8 * math.exp(-0.9 * (base["bpw"] - 1.0)) * 0.5  # a candidate better than its reference: small
    n_values = n_tokens
    klds = sorted(max(0.0, scale * rng.lognormvariate(-0.6, 1.1) / 1.2) for _ in range(min(n_values, 2000)))
    print(f"kl_divergence: computing over {n_chunks} chunks, n_ctx={ctx}, batch_size=2048, n_seq=1", file=sys.stderr)
    for i in range(n_chunks):
        time.sleep(delay)
        print(
            f"{i + 1:4d}  {5.0 + scale:9.4f}  {scale * 0.02:9.6f}  {scale:9.6f}  {scale * 4:8.3f} %  "
            f"{100 - scale * 20:8.3f} %",
            flush=True,
        )
    mean = sum(klds) / len(klds)
    unc = (sum((x - mean) ** 2 for x in klds) / max(1, len(klds) - 1)) ** 0.5 / math.sqrt(len(klds))
    ppl_base, ppl_q = 5.1, 5.1 * math.exp(mean * 0.3)
    same_top = max(0.0, 100.0 - 60.0 * mean - 0.5 * scale)
    print()
    print("====== Perplexity statistics ======")
    print(f"Mean PPL(Q)                   : {ppl_q:10.6f} ±  {0.31:9.6f}")
    print(f"Mean PPL(base)                : {ppl_base:10.6f} ±  {0.30:9.6f}")
    print(f"Cor(ln(PPL(Q)), ln(PPL(base))): {100 - 8 * mean:6.2f}%")
    print(f"Mean ln(PPL(Q)/PPL(base))     : {math.log(ppl_q / ppl_base):10.6f} ±  {unc:9.6f}")
    print(f"Mean PPL(Q)/PPL(base)         : {ppl_q / ppl_base:10.6f} ±  {unc:9.6f}")
    print(f"Mean PPL(Q)-PPL(base)         : {ppl_q - ppl_base:10.6f} ±  {unc * 5:9.6f}")
    print()
    print("====== KL divergence statistics ======")
    print(f"Mean    KLD: {mean:10.6f} ±  {unc:9.6f}")
    print(f"Maximum KLD: {klds[-1]:10.6f}")
    for label, q in (("99.9%", 0.999), ("99.0%", 0.99), ("95.0%", 0.95), ("90.0%", 0.9)):
        print(f"{label}   KLD: {pct(klds, q):10.6f}")
    print(f"Median  KLD: {pct(klds, 0.5):10.6f}")
    for label, q in (("10.0%", 0.1), (" 5.0%", 0.05), (" 1.0%", 0.01), (" 0.1%", 0.001)):
        print(f"{label}   KLD: {pct(klds, q):10.6f}")
    print(f"Minimum KLD: {klds[0]:10.6f}")
    print()
    print("====== Token probability statistics ======")
    print(f"Mean    Δp: {-0.8 * scale:6.3f} ± {0.05:5.3f} %")
    print(f"Maximum Δp: {min(99.0, 30 * scale + 5):6.3f}%")
    print(f"Median  Δp: {-0.001 * scale:6.3f}%")
    print(f"Minimum Δp: {-min(99.0, 30 * scale + 5):6.3f}%")
    print()
    print(f"RMS Δp    : {4.0 * math.sqrt(mean) * 10:6.3f} ± {0.1:5.3f} %")
    print(f"Same top p: {same_top:6.3f} ± {0.2:5.3f} %")
    return 0


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    sys.exit(main(sys.argv[1:]))

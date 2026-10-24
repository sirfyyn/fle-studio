#!/usr/bin/env python3
"""A stand-in for ``llama-bench`` (llama.cpp tools/llama-bench) for tests and CI. Standard library only.

    fake_llama_bench.py -m MODEL.gguf [-t 4,6,8] [-b 512,1024] [-ub 512] [-ncmoe 0,8,16] [-p 512] [-n 128] [-r 3]
                        [-o json|csv|md] [-ngl N]

Comma lists are run as every combination, like the real tool. For every combination one prompt test (``n_prompt > 0``)
and one generation test (``n_gen > 0``) are printed, each with ``r`` samples in ``samples_ts`` and their mean in
``avg_ts`` (the field names of the JSON output of llama-bench). The speeds follow the same made-up model as
``fake_llama_server.py`` with FAKE_SPEED_MODEL=1: a sweet spot for the thread count (FAKE_BEST_THREADS, default 6) and
for ``--n-cpu-moe`` (FAKE_BEST_NCMOE, default 8), and a larger batch reads the prompt faster. Each sample varies by up
to FAKE_RUN_NOISE (default 0.02), deterministically.

FAKE_BENCH_FAIL=1 exits with an error.
"""

from __future__ import annotations

import itertools
import json
import os
import random
import sys
from pathlib import Path

BASE_TG = 80.0
BASE_PP = 2000.0


def parse(argv: list[str]) -> dict[str, list[str]]:
    opts: dict[str, list[str]] = {}
    i = 0
    while i < len(argv):
        token = argv[i]
        i += 1
        if token.startswith("-") and i < len(argv) and not argv[i].startswith("-"):
            opts[token] = argv[i].split(",")
            i += 1
    return opts


def ints(opts: dict[str, list[str]], *names: str, default: int) -> list[int]:
    for n in names:
        if n in opts:
            return [int(x) for x in opts[n]]
    return [default]


def main(argv: list[str]) -> int:
    opts = parse(argv)
    model = (opts.get("-m") or opts.get("--model") or [""])[0]
    if not model or not Path(model).is_file():
        print(f"error: failed to load model '{model}'", file=sys.stderr)
        return 1
    if os.environ.get("FAKE_BENCH_FAIL") == "1":
        print("error: the benchmark failed", file=sys.stderr)
        return 1
    threads = ints(opts, "-t", "--threads", default=os.cpu_count() or 4)
    batches = ints(opts, "-b", "--batch-size", default=2048)
    ubatches = ints(opts, "-ub", "--ubatch-size", default=512)
    ncmoe = ints(opts, "-ncmoe", "--n-cpu-moe", default=0)
    n_prompt = ints(opts, "-p", "--n-prompt", default=512)[0]
    n_gen = ints(opts, "-n", "--n-gen", default=128)[0]
    reps = ints(opts, "-r", "--repetitions", default=5)[0]
    fmt = (opts.get("-o") or ["md"])[0]
    best_threads = int(os.environ.get("FAKE_BEST_THREADS", "6"))
    best_ncmoe = int(os.environ.get("FAKE_BEST_NCMOE", "8"))
    noise = float(os.environ.get("FAKE_RUN_NOISE", "0.02"))
    rows = []
    for t, b, ub, nc in itertools.product(threads, batches, ubatches, ncmoe):
        f_threads = max(0.3, 1.0 - 0.05 * abs(t - best_threads))
        f_decode = f_threads * max(0.4, 1.0 - 0.02 * abs(nc - best_ncmoe))
        f_prefill = f_threads * (min(max(b, 64), 2048) / 2048) ** 0.25
        for kind, count_p, count_g, rate in (
            ("pp", n_prompt, 0, BASE_PP * f_prefill),
            ("tg", 0, n_gen, BASE_TG * f_decode),
        ):
            rng = random.Random(f"{t}-{b}-{ub}-{nc}-{kind}")
            samples = [rate * (1 + noise * (2 * rng.random() - 1)) for _ in range(reps)]
            mean = sum(samples) / len(samples)
            sd = (sum((x - mean) ** 2 for x in samples) / max(1, len(samples) - 1)) ** 0.5
            rows.append(
                {
                    "build_commit": "fake0000",
                    "build_number": 1,
                    "model_filename": model,
                    "n_batch": b,
                    "n_ubatch": ub,
                    "n_threads": t,
                    "n_cpu_moe": nc,
                    "n_prompt": count_p,
                    "n_gen": count_g,
                    "n_depth": 0,
                    "avg_ts": mean,
                    "stddev_ts": sd,
                    "samples_ts": samples,
                }
            )
    if fmt == "json":
        print(json.dumps(rows, indent=1))
    else:
        print("| model | threads | batch | test | t/s |")
        for r in rows:
            test = f"pp{r['n_prompt']}" if r["n_prompt"] else f"tg{r['n_gen']}"
            print(
                f"| {Path(model).name} | {r['n_threads']} | {r['n_batch']} | {test} | "
                f"{r['avg_ts']:.2f} ± {r['stddev_ts']:.2f} |"
            )
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

"""Stand-ins for the llama.cpp command-line tools Forge drives (llama-quantize, -imatrix, -perplexity, -fle-reap).

They are started like the real ones (`python llama-quantize.py ...`, see tools.command_for) and print the same kind of
progress lines, so the job runner, the progress parsing and the cancel path are exercised without any model code.
Knobs (environment variables):
  FAKE_TOOLS_LOG    file that receives one JSON line {"tool", "argv"} per call
  FAKE_DELAY        seconds to sleep between progress lines (default 0)
  FAKE_FAIL         name of a tool that exits with code 3 (for example "quantize")
  FAKE_KLD          mean KL divergence printed by the comparison run (default 0.0123)
  FAKE_NO_STATS     if set, the comparison run prints nothing useful
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import time
from pathlib import Path

HELP = {
    "quantize": "usage: llama-quantize [--help] [--allow-requantize] [--leave-output-tensor] [--imatrix file]\n"
    "  --output-tensor-type ggml_type\n  --token-embedding-type ggml_type\n  --tensor-type TENSOR=TYPE\n",
    "imatrix": "usage: llama-imatrix -m model -f text -o out\n  --chunks N\n  --in-file file\n  -ngl N\n",
    "perplexity": "usage: llama-perplexity -m model -f text\n  --kl-divergence\n  --kl-divergence-base file\n"
    "  --chunks N\n  -ngl N\n",
    "fle-reap": "usage: llama-fle-reap -m model -f text -o table.tsv\n  --chunks N\n  -cmoe\n  -fit on|off\n",
}

STATS = """
====== Perplexity statistics ======
Mean PPL(Q)                   :   6.234567 ±   0.123456
Mean PPL(base)                :   6.100000 ±   0.120000
Cor(ln(PPL(Q)), ln(PPL(base))):  99.12%
Mean ln(PPL(Q)/PPL(base))     :   0.021750 ±   0.001000
Mean PPL(Q)/PPL(base)         :   1.021988 ±   0.001020
Mean PPL(Q)-PPL(base)         :   0.134567 ±   0.006000

====== KL divergence statistics ======
Mean    KLD:   {kld:.6f} ±   0.000123
Maximum KLD:   1.234567
99.9%   KLD:   0.200000
99.0%   KLD:   0.100000
95.0%   KLD:   0.050000
Median  KLD:   0.005000

====== Token probability statistics ======
Mean    Δp: -0.123 ± 0.010 %
RMS Δp    :  3.210 ± 0.020 %
Same top p: 95.123 ± 0.050 %
"""


def say(text: str) -> None:
    sys.stdout.buffer.write((text + "\n").encode("utf-8"))
    sys.stdout.flush()


def pause() -> None:
    delay = float(os.environ.get("FAKE_DELAY", "0") or 0)
    if delay:
        time.sleep(delay)


def value(argv: list[str], flag: str) -> str | None:
    return argv[argv.index(flag) + 1] if flag in argv else None


def quantize(argv: list[str]) -> int:
    options_with_value = {"--imatrix", "--output-tensor-type", "--token-embedding-type", "--tensor-type"}
    positional, i = [], 0
    while i < len(argv):
        if argv[i] in options_with_value:
            i += 2
        elif argv[i].startswith("--"):
            i += 1
        else:
            positional.append(argv[i])
            i += 1
    source, dest = positional[0], positional[1]
    total = 12
    for n in range(1, total + 1):
        say(f"[{n:4d}/{total:4d}]  blk.{n}.ffn_down_exps.weight - quantizing")
        pause()
    shutil.copyfile(source, dest)
    say("main: quantize time = 1.00 ms")
    return 0


def imatrix(argv: list[str]) -> int:
    from forge_synth import make_imatrix

    from fle_studio_forge import ggufio

    model = ggufio.open_model(value(argv, "-m") or "")
    layers, experts = model.number("block_count") or 1, model.number("expert_count") or 1
    chunks = int(value(argv, "--chunks") or 4)
    say(f"compute_imatrix: computing over {chunks} chunks, n_ctx={value(argv, '-c')}")
    for k in range(1, chunks + 1):
        say(f"[{k}]{5 + k / 10:.4f},")
        pause()
    counts = {layer: [(e * 7 + layer * 3) % 11 + 1 for e in range(experts)] for layer in range(layers)}
    make_imatrix(value(argv, "-o") or "imatrix.gguf", layers, experts, counts)
    say("save_imatrix: stored collected data")
    return 0


def perplexity(argv: list[str]) -> int:
    chunks = int(value(argv, "--chunks") or 3)
    base = value(argv, "--kl-divergence-base")
    say(f"perplexity: calculating perplexity over {chunks} chunks")
    for k in range(1, chunks + 1):
        say(f"[{k}]{6 + k / 10:.4f},")
        pause()
    if "--kl-divergence" in argv:
        if not base or not Path(base).is_file():
            say("error: the stored probabilities are missing")
            return 2
        if os.environ.get("FAKE_NO_STATS"):
            say("nothing to report")
            return 0
        say(STATS.format(kld=float(os.environ.get("FAKE_KLD", "0.0123"))))
    elif base:
        Path(base).write_bytes(b"FAKEKLDBASE" * 10)
    return 0


def fle_reap(argv: list[str]) -> int:
    from fle_studio_forge import ggufio

    model = ggufio.open_model(value(argv, "-m") or "")
    layers, experts = model.number("block_count") or 1, model.number("expert_count") or 1
    chunks = int(value(argv, "--chunks") or 4)
    say(f"computing over {chunks} chunks")
    for k in range(1, chunks + 1):
        say(f"[{k}]1.0,")
        pause()
    lines = ["layer\texpert\tcount\tsum_gate\tsum_saliency\tsum_norm"]
    for layer in range(layers):
        for e in range(experts):
            count = (e * 5 + layer * 2) % 9 + 1
            lines.append(f"{layer}\t{e}\t{count}\t{float(count)}\t{float(count) * (1 + e % 2)}\t{float(count)}")
    Path(value(argv, "-o") or "table.tsv").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return 0


TOOLS = {"quantize": quantize, "imatrix": imatrix, "perplexity": perplexity, "fle-reap": fle_reap}


def main(name: str) -> None:
    argv = sys.argv[1:]
    if argv[:1] == ["--version"]:
        say("version: 9999 (abc1234)")
        raise SystemExit(0)
    if argv[:1] == ["--help"] or argv[:1] == ["-h"]:
        say(HELP[name])
        raise SystemExit(0)
    log = os.environ.get("FAKE_TOOLS_LOG")
    if log:
        with open(log, "a", encoding="utf-8") as f:
            f.write(json.dumps({"tool": name, "argv": argv}) + "\n")
    if os.environ.get("FAKE_FAIL") == name:
        say("error: this tool was told to fail")
        raise SystemExit(3)
    raise SystemExit(TOOLS[name](argv))

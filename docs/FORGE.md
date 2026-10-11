# Forge: build your own quant

Forge is the page of FLE Studio that builds a model file for you. It does not contain a quantizer of its own: it
rearranges GGUF files itself (hybrids, pruning) and drives the command-line tools of your llama.cpp build for
everything else (`llama-quantize`, `llama-imatrix`, `llama-perplexity`, and in FLE builds `llama-fle-reap`). It runs
on macOS, Linux and Windows. Status: in development for 0.2, not in a release yet.

## The flow

1. **Start.** Choose the way: a *hybrid* of quants you already have, or *quantize* a full-precision model. Forge shows
   which tools it found.
2. **Models.** Point at the files. Forge reads the headers only (a split model is one model) and shows layers,
   experts, quant types and bits per weight.
3. **Recipe.** Hybrid: give a size budget; the calculator chooses, for every expert layer, which source it comes from.
   You can change single layers. Quantize: choose the type and rules per tensor.
4. **Prune (optional).** Keep the best K experts in every layer.
5. **Verify.** A calibration text (for the importance matrix and the router statistics) and a comparison text, with
   the reference model.
6. **Build.** Forge checks the plan, runs the job and shows progress and the log. The result card shows size,
   KL divergence, top-1 agreement and the SHA-256. *Add to my models* puts it in the library with a start profile.

## Hybrids

A mixture-of-experts model keeps its experts in a few large tensors per layer (`ffn_gate_exps`, `ffn_up_exps`,
`ffn_down_exps`, or `ffn_gate_up_exps`). Quants of the same model have the same tensors in the same shapes, only
the type differs. A hybrid takes the whole expert layer from one source or another and everything else (metadata,
attention, norms, embeddings) from the base. Nothing is requantized, so nothing is lost beyond what the donors
already lost, and the build is a copy at disk speed.

The calculator solves a multiple-choice knapsack exactly: per layer one source, total size not above the budget,
minimal summed error. Sizes are rounded up to a step, so the result never exceeds the budget. The error per layer
and source is either a table of measured errors (`hybrid.errors`) or, by default, a stand-in `4^-bits` per weight,
which orders the quants correctly but does not know which layers are sensitive. The file size is predicted exactly
from the headers.

The file is written to `NAME.gguf.part` with `NAME.gguf.part.json`. After a cancel or a crash the build continues:
the header is deterministic, the finished prefix is hashed again, and the running SHA-256 is restored. A finished
file is compared at both ends of every tensor with its source.

## Quantizing

`llama-quantize` is called with `--imatrix`, `--tensor-type PATTERN=TYPE`, `--output-tensor-type`,
`--token-embedding-type` and `--allow-requantize` as the recipe says. Forge checks with `--help` which of these the
tool supports and refuses a recipe that needs one it lacks. The size shown before the run is an estimate from the
tensor shapes.

## REAP pruning

For every layer the K experts with the highest score are kept; the others are dropped from the three expert tensors
by copying contiguous byte ranges (the expert axis is the slowest, so this works for every quant type). The router
(`ffn_gate_inp`) and, if present, `exp_probs_b` lose the same rows, and `<arch>.expert_count` becomes K. Scores:
`reap` (saliency over count), `man` (norm over count), `freq`, `mass`, `energy`.

Statistics come from `llama-fle-reap` (a TSV: layer, expert, count, sum of gate, sum of saliency, sum of norm). Without
that tool Forge derives counts and input energy from an importance matrix, which is a weaker signal and is logged as
such. Limits: grouped expert routing is refused; K must be at least the number of experts used per token; pruning
cannot be undone, so prune the best quality you have. A manifest `NAME.gguf.reap.json` records the choice and the
share of routing mass kept per layer.

## Checking

Two runs of `llama-perplexity` on the same text, context and chunk count: the reference model stores its token
probabilities (`--kl-divergence-base`), the result compares itself with them (`--kl-divergence`). Forge parses mean
and tail KL divergence, top-1 agreement, perplexity and its ratio. The stored probabilities grow with chunks times
vocabulary, so the default is short (20 chunks). Compare only runs with the same text, context, chunks, tool build
and hardware. The FLE figures shown for orientation were measured on the FLE project's own texts.

## Calibration text

No text is bundled. A mixture-of-experts model needs plenty of varied text: a short one never reaches many experts.
Forge can mix your own files and folders into one text (seeded draws, a report per source), or take the Python
standard library as a first trial. **Open question:** a text that may ship with Studio needs a license that allows
redistribution; until one is chosen, bring your own and check its license before sharing numbers that depend on it.

## Tools

Searched in this order: the folder on the Start step (`forge.tools_dir`), the folders of the engines installed through
Studio, the folder of your own `llama-server`, then `PATH`. Each is run with `--version` and `--help` once per file
version. A hybrid needs none of them.

## Recipe, schema 1

```json
{
  "schema": 1, "kind": "fle-forge-recipe", "name": "my-quant", "created": "", "studio": "", "notes": "",
  "mode": "hybrid",
  "sources": {"B": {"file": "base-Q8_0.gguf", "path": "", "role": "base", "size": 0, "sha256": ""}},
  "hybrid": {"base": "B", "layers": {"12": "K"}, "rules": [{"source": "K", "pattern": "attn", "layers": "0-11"}],
             "budget_gb": 0, "objective": "bpw", "errors": {"K": {"12": 0.031}}},
  "quantize": {"source": "S", "type": "iq4_nl", "rules": [{"pattern": "ffn_.*_exps", "type": "iq3_s"}],
               "output_type": "", "embedding_type": "", "allow_requantize": false, "imatrix": true},
  "imatrix": {"corpus": "", "file": "", "chunks": 200, "context": 2048, "gpu_layers": 0, "threads": 0},
  "reap": {"enabled": false, "keep": 0, "score": "reap", "table": "", "chunks": 256, "gpu_layers": 0, "threads": 0},
  "verify": {"enabled": true, "reference": "", "text": "", "context": 2048, "chunks": 20, "gpu_layers": 0,
             "threads": 0, "base_logits": ""}
}
```

A shared recipe carries file names but no folders. A recipe never contains a command line: Forge builds every
command itself, patterns are limited to a safe character set and types to the list `llama-quantize` knows, so a recipe
from someone else cannot make Studio run anything but its own steps. Output goes only to `<models>/forge/`. A finished
job also writes `NAME.recipe.json` (portable, with the SHA-256 of the result) and `NAME.forge.json` (tool versions,
sizes, the result card) next to the model.

## Jobs

One job runs at a time. Its state is `data/forge/jobs/<id>/job.json` with a log. Steps: importance matrix, build,
router statistics, prune, reference probabilities, KL divergence, checksums. A job that was running when Studio
closed comes back as *interrupted*; *Resume* repeats only the steps that are not done.

## API

`GET /api/forge`; `POST /api/forge/{tools,inspect,calculate,corpus,check,jobs}`, `POST /api/forge/recipe/{import,share}`,
`GET /api/forge/jobs/<id>[/log?since=N|/recipe]`, `POST /api/forge/jobs/<id>/{cancel,resume,delete,publish}`. Changing
calls need the page token like everything else in Studio.

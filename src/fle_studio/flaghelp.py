"""One short sentence for every flag and environment variable in the engine command.

The command FLE Studio shows is what really runs; people who did not write it need to know what each part does
before they dare to change it. ``explain`` turns a command into a list of ``{name, value, text}`` lines (shown under
the command), ``table`` gives the same sentences by name for the tooltips on the command itself.

Only what is known is explained: a flag that is not in the table below gets ``known: false`` and no guess. The
sentences for the FLE engine's own settings say what the name stands for and point to the model card for the numbers,
because the numbers there come from measurements on one machine.
"""

from __future__ import annotations

import re
from typing import Any

# canonical name -> (aliases, takes a value, sentence)
_FLAGS: dict[str, tuple[tuple[str, ...], bool, str]] = {
    "-m": (("--model",), True, "The model file to load. For a split model, the first part (…-00001-of-0000N.gguf)."),
    "-md": (
        ("--model-draft", "--spec-draft-model"),
        True,
        "A second, small model that guesses the next tokens so the main model can check several at once "
        "(speculative decoding). With the FLE engine this is the MTP head.",
    ),
    "--spec-type": (
        (),
        True,
        "Which kind of guessing to use. draft-mtp uses the MTP head that comes with the model.",
    ),
    "--spec-draft-n-max": (
        ("--draft-max", "--draft", "--draft-n"),
        True,
        "The most tokens guessed ahead per step. More can speed up easy text and wastes work on hard text.",
    ),
    "--spec-draft-p-min": (
        ("--draft-p-min",),
        True,
        "A guess is only offered when the draft model is at least this sure (0 to 1).",
    ),
    "--no-spec-draft-backend-sampling": (
        (),
        False,
        "Part of the FLE reference command for MTP: the draft tokens are not sampled inside the GPU backend.",
    ),
    "-cmoed": (
        ("--cpu-moe-draft",),
        False,
        "Keep the draft model's expert weights on the CPU, so the GPU memory goes to the main model.",
    ),
    "-c": (
        ("--ctx-size",),
        True,
        "How many tokens the model can see at once (prompt plus answer), shared by all parallel requests. "
        "A larger value needs more memory.",
    ),
    "--parallel": (
        ("-np", "--n-parallel"),
        True,
        "How many requests are served at the same time. Without a unified cache each gets an equal share of the "
        "context.",
    ),
    "--kv-unified": (
        ("-kvu",),
        False,
        "One shared pool of context for all parallel requests instead of fixed shares, so one long chat can use the "
        "room the others do not need.",
    ),
    "--jinja": (
        (),
        False,
        "Use the chat template stored in the model file. Needed for correct roles, reasoning and tool calls.",
    ),
    "-b": (
        ("--batch-size",),
        True,
        "How many prompt tokens are handed to the engine per round. Larger reads long prompts faster and needs more "
        "memory.",
    ),
    "-ub": (
        ("--ubatch-size",),
        True,
        "How many tokens the GPU computes at once while reading the prompt. Larger is faster up to a point, then it "
        "only costs memory.",
    ),
    "-ctk": (
        ("--cache-type-k",),
        True,
        "Number format of the keys in the context memory (KV cache). q8_0 takes about half the memory of f16.",
    ),
    "-ctv": (
        ("--cache-type-v",),
        True,
        "Number format of the values in the context memory (KV cache). q8_0 takes about half the memory of f16.",
    ),
    "-fit": (
        ("--fit",),
        True,
        "Let the engine adjust what is not set by hand (how much runs on the GPU) so that everything fits into the "
        "available GPU memory.",
    ),
    "--fit-target": (
        ("-fitt",),
        True,
        "GPU memory in MiB that the fitting step keeps free as a margin.",
    ),
    "--moe-expert-cache-inserts": (
        (),
        True,
        "FLE engine: how many experts may be copied into the GPU expert cache per generated token.",
    ),
    "--cache-ram": (
        ("-cram",),
        True,
        "RAM in MiB for remembering earlier prompts, so a follow-up question does not read the whole chat again. "
        "0 turns it off.",
    ),
    "--load-mode": (
        (),
        True,
        "FLE engine: how the expert weights are held. none keeps them in pinned RAM for fast copies to the GPU "
        "(costs RAM that cannot be swapped out).",
    ),
    "-t": (
        ("--threads",),
        True,
        "CPU threads for generating. Usually one per physical core; the FLE engine leaves a core free for its own "
        "helper threads on processors without hyper-threading.",
    ),
    "-tb": (("--threads-batch",), True, "CPU threads for reading the prompt (default: same as -t)."),
    "-C": (
        ("--cpu-mask",),
        True,
        "Which CPU cores the engine's threads may use, as a hex mask (0x0000FFFF = cores 0 to 15).",
    ),
    "--cpu-strict": (
        (),
        True,
        "1 pins every thread to its core instead of letting the system move it around.",
    ),
    "--temp": (
        ("--temperature",),
        True,
        "Randomness of the answers: lower is more predictable. The value is the one the model's authors recommend.",
    ),
    "--top-p": (
        ("--top_p",),
        True,
        "Only the most likely words that together make up this share (0 to 1) are considered.",
    ),
    "--top-k": (("--top_k",), True, "Only the K most likely words are considered for each step."),
    "--min-p": (("--min_p",), True, "Words less than this share as likely as the best one are dropped."),
    "--presence-penalty": (
        (),
        True,
        "Discourages words that already appeared; stops long answers from going in circles.",
    ),
    "--host": ((), True, "The address the engine listens on. 127.0.0.1 means this computer only."),
    "--port": ((), True, "The port the engine listens on. FLE Studio picks a free one."),
    "--metrics": (
        (),
        False,
        "Turns on the counters that FLE Studio reads for the live speed numbers.",
    ),
    "-a": (("--alias",), True, "The model name the API reports."),
    "-lv": (
        ("--verbosity", "--log-verbosity"),
        True,
        "How much the engine writes to its log. 4 also prints the loading steps and the expert cache hit rate that "
        "FLE Studio shows.",
    ),
    "-ngl": (
        ("--n-gpu-layers", "--gpu-layers"),
        True,
        "How many of the model's layers run on the GPU. Without it the engine decides (see -fit).",
    ),
    "--n-cpu-moe": (
        ("-ncmoe",),
        True,
        "Keep the expert weights of the first N layers on the CPU. A larger N frees GPU memory and is slower; "
        "the best N depends on your GPU memory.",
    ),
    "--cpu-moe": (
        ("-cmoe",),
        False,
        "Keep all expert weights on the CPU. Leaves the GPU for the rest of the model, at a cost in speed.",
    ),
    "-fa": (
        ("--flash-attn",),
        True,
        "Flash attention: a faster, leaner way to compute attention. Usually on or auto.",
    ),
    "--no-mmap": (
        (),
        False,
        "Read the whole model into RAM instead of mapping the file. Slower to start, avoids reading from disk later.",
    ),
    "--mlock": (
        (),
        False,
        "Keep the model in RAM so the system cannot swap it out.",
    ),
    "-sm": (("--split-mode",), True, "How a model is split over several GPUs: none, layer or row."),
    "-ts": (("--tensor-split",), True, "How much of the model goes to each GPU, as proportions like 3,1."),
    "--no-warmup": (
        (),
        False,
        "Skip the empty test run at start. Starts sooner, and the first answer is slower.",
    ),
    "--chat-template-kwargs": (
        (),
        True,
        "Extra settings for the chat template as JSON, for example to switch thinking on or off.",
    ),
    "--api-key": ((), True, "A key that clients must send to use the API."),
}

_ENV: dict[str, str] = {
    "FLE_MOE_CACHE_BUDGET_MIB": "FLE engine: GPU memory in MiB for the expert cache, the experts it keeps ready "
    "on the GPU.",
    "FLE_MOE_CACHE_BUDGET_FIT": "FLE engine: 1 counts the cache budget in the fitting step, so other buffers shrink "
    "instead of running out of memory.",
    "FLE_MOE_PROFILE": "FLE engine: file where the engine keeps which experts this model uses most, so the cache "
    "starts warm next time.",
    "FLE_MOE_PROFILE_SAVE_SEC": "FLE engine: how often, in seconds, that profile is saved.",
    "FLE_MTP_VOCAB": "FLE engine: the reduced word list (32k) the MTP head guesses from.",
    "FLE_MTP_VOCAB_N": "FLE engine: how many words the reduced list has.",
    "CUDA_VISIBLE_DEVICES": "Which NVIDIA GPUs the engine may use, by number.",
    "CUDA_DEVICE_ORDER": "PCI_BUS_ID numbers the GPUs the way nvidia-smi does.",
    "LLAMA_ARG_LOG_VERBOSITY": "Same as -lv: how much the engine writes to its log.",
}
_ENV_GENERIC_FLE = (
    "FLE engine setting from the reference command (cache policy or compute switch). The value is the one measured "
    "on the reference machine; the model card explains the setting."
)

_BY_NAME: dict[str, str] = {}
for _canon, (_aliases, _, _) in _FLAGS.items():
    _BY_NAME[_canon] = _canon
    for _alias in _aliases:
        _BY_NAME[_alias] = _canon


def lookup(name: str) -> tuple[str, bool, str] | None:
    """(canonical name, takes a value, sentence) for a flag, or None if it is not in the table."""
    canon = _BY_NAME.get(name)
    if canon is None:
        return None
    return canon, _FLAGS[canon][1], _FLAGS[canon][2]


def names_of(name: str) -> tuple[str, ...]:
    """Every spelling of a flag (-t, --threads), canonical name first; just ``(name,)`` for one that is not known."""
    canon = _BY_NAME.get(name)
    return (canon, *_FLAGS[canon][0]) if canon else (name,)


def env_text(name: str) -> str | None:
    if name in _ENV:
        return _ENV[name]
    if name.startswith("FLE_"):
        return _ENV_GENERIC_FLE
    if name.startswith("LLAMA_ARG_"):
        return "The same as the command line flag of that name (llama.cpp reads it from the environment)."
    return None


def table() -> dict[str, Any]:
    """Every known name (with its aliases) and sentence, for the tooltips: {flags: {name: {text, value}}, env}."""
    flags = {}
    for name, canon in _BY_NAME.items():
        _, takes, text = _FLAGS[canon]
        flags[name] = {"text": text, "value": takes, "canonical": canon}
    return {"flags": flags, "env": dict(_ENV), "fle_env_generic": _ENV_GENERIC_FLE}


_NUMBER = re.compile(r"^-\d")


def explain(args: list[str], env: dict[str, str] | None = None) -> list[dict[str, Any]]:
    """The command's environment and arguments, one entry each: {kind, name, value, text, known}."""
    out: list[dict[str, Any]] = []
    for key, value in (env or {}).items():
        text = env_text(key)
        out.append({"kind": "env", "name": key, "value": value, "text": text or "", "known": text is not None})
    i = 0
    while i < len(args):
        token = args[i]
        i += 1
        if not token.startswith("-") or _NUMBER.match(token):
            continue
        name, eq, inline = token.partition("=") if token.startswith("--") else (token, "", "")
        info = lookup(name)
        takes = info[1] if info else (bool(eq) or (i < len(args) and not args[i].startswith("-")))
        value = ""
        if eq:
            value = inline
        elif takes and i < len(args) and (not args[i].startswith("-") or _NUMBER.match(args[i])):
            value = args[i]
            i += 1
        out.append(
            {
                "kind": "flag",
                "name": name,
                "value": value,
                "text": info[2] if info else "",
                "known": info is not None,
            }
        )
    return out

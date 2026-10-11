#!/usr/bin/env python3
"""A stand-in for ``llama-server`` (llama.cpp and the FLE fork) for tests, demos and CI.

Standard library only, runs on macOS, Linux and Windows::

    python fake_llama_server.py -m MODEL.gguf --host 127.0.0.1 --port N [-c N] [-a ALIAS] [--metrics] [-md DRAFT.gguf]

Unknown flags, with or without a value, are accepted and ignored (the real server would refuse them).

What is imitated, and where the shapes come from
------------------------------------------------
Log lines, HTTP routes and JSON bodies follow llama.cpp tag b11540 (tools/server/server.cpp, server-context.cpp,
server-task.cpp, server-common.cpp, server-http.cpp, common/log.cpp) and the FLE fork at commit 1efd83707.

* The log goes to stderr in the real shape ``M.SS.mmm.uuu L srv  function: message`` and honours the verbosity
  threshold like the real tool: ``LOG_INF`` lines are visible by default (threshold 3), but everything the llama
  library prints at INFO level (model loader, ``print_info``, ``load_tensors``, ``llama_context``, ``llama_kv_cache``,
  FLE's ``MoE expert cache enabled`` and ``moe-cache: steps=...``) only appears with ``-lv 4`` or ``--verbose``.
  The readiness line is ``srv  llama_server: listening on http://HOST:PORT`` (there is no "server is listening").
* The HTTP server binds before the model "loads"; every route except the UI answers 503 "Loading model" until then.
* ``GET /health /props /slots /metrics /v1/models`` and ``POST /v1/chat/completions`` (streaming and not).
* Chat replies are deterministic. Unless ``chat_template_kwargs.enable_thinking`` is false, about 40 tokens of
  ``reasoning_content`` come first. Prompt cache, slot selection, timings, usage, Prometheus counters and the
  ``print_timings`` log block follow the real accounting. Timings come from a virtual model clock (the configured
  speed plus a few percent of deterministic jitter), so they stay stable on a loaded CI runner.
* ``-md`` / ``--spec-type`` simulate speculative decoding (``draft_n``, ``draft_n_accepted``, the ``draft acceptance``
  log line, ``spec_decode_*`` metrics).
* FLE mode (env ``FLE_MOE_CACHE_BUDGET_MIB`` set, or ``FAKE_ENGINE=fle``) switches to the FLE identity, startup
  notices and expert-cache lines.

Not implemented: ``/completion``, ``/v1/completions``, embeddings, tokenize, LoRA, slot save/restore, router mode.

Environment knobs
-----------------
FAKE_LOAD_SECONDS        model "load" time, default 1.0
FAKE_TOK_PER_SEC         generation speed, default 80
FAKE_PROMPT_TOK_PER_SEC  prompt processing speed, default 2000
FAKE_JITTER              relative speed jitter, default 0.04
FAKE_ANSWER_TOKENS       length of the answer, default derived from the prompt (32..55 tokens)
FAKE_THINK_TOKENS        length of the reasoning, default 40
FAKE_FAIL=oom            fail the load like a CUDA out-of-memory error, exit code 1
FAKE_CRASH_AFTER=S       exit with code 134 S seconds after the server became ready
FAKE_ENGINE=fle|upstream force the engine identity
FAKE_STRICT_GGUF=1       refuse model files that do not start with the GGUF magic
FAKE_REJECT_ARGS=a,b     refuse these flags with ``error: invalid argument``, like a server that does not know them
FAKE_LOAD_DOTS=1         print the loader's progress dots (visible with -lv 4 only)
FAKE_VISION=1            report vision support in /props and accept picture parts in chat messages (otherwise they
                         are refused like a server without an mmproj file does)

Tools and answer formats
------------------------
A request with ``tools`` makes the fake call a tool when the user's last message contains
``CALL_TOOL name {"json": "arguments"}`` (several of them give several parallel calls) and the request offers a tool
of that name. The reply is a streamed ``tool_calls`` delta (or ``message.tool_calls``) with finish reason
``tool_calls``. When the last message is a tool result, the answer quotes it ("Tool said ..."). A forced answer
format (``response_format`` other than text, ``json_schema``, ``grammar``) gives a JSON-only answer and never a tool
call, even if tools were sent: this is the trap that FLE Studio must not fall into.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import random
import re
import select
import signal
import socket
import socketserver
import string
import struct
import sys
import threading
import time
from collections import deque
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any, NoReturn
from urllib.parse import parse_qs, urlsplit

T0 = time.monotonic()
IS_WINDOWS = os.name == "nt"
INT_MAX = 2**31 - 1
N_BATCH = 2048  # prompt tokens per decode call
ACCEPT_P = 0.72  # chance that the target model accepts the next drafted token
ALNUM = string.digits + string.ascii_uppercase + string.ascii_lowercase
NUMBER_RE = re.compile(r"^-\d")
TRUTHY = {"1", "true", "on", "yes", "enabled"}
FALSY = {"0", "false", "off", "no", "disabled"}

ENGINES = {
    "upstream": {"version": "0.6.0-dev", "build": 11540, "commit": "1e6f04a75", "port": 9931},
    "fle": {"version": "0.4.0-dev", "build": 10825, "commit": "1efd83707", "port": 8080},
}

CHAT_TEMPLATE = (
    "{%- for message in messages %}{{- '<|im_start|>' + message.role + '\\n' + message.content + '<|im_end|>\\n' }}"
    "{%- endfor %}{%- if add_generation_prompt %}{{- '<|im_start|>assistant\\n' }}{%- endif %}"
)
CHAT_TEMPLATE_CAPS = {
    "supports_object_arguments": True,
    "supports_parallel_tool_calls": True,
    "supports_preserve_reasoning": False,
    "supports_reasoning_effort": False,
    "supports_string_content": True,
    "supports_system_role": True,
    "supports_tool_calls": True,
    "supports_tools": True,
    "supports_typed_content": False,
}
SAMPLERS = ["penalties", "dry", "top_n_sigma", "top_k", "typ_p", "top_p", "min_p", "xtc", "temperature"]

THINK_SENTENCES = (
    "The user sent a short message, so a short reply is enough.",
    "I will read it once more to make sure I understood the request.",
    "Nothing in it is ambiguous, so I can answer directly.",
    "A clear and friendly tone fits best here.",
    "Let me put the answer together step by step.",
)
ANSWER_SENTENCES = (
    "This is a canned reply from the fake llama-server that FLE Studio uses for its tests.",
    "The text is deterministic, so the same prompt always produces the same words.",
    "Speed, token counts and timings follow the configured rates, not a real model.",
    "You can stop the generation at any time and the server releases the slot at once.",
    "Everything else about the API follows the real server closely.",
)

# (file name token, name printed by llama_ftype_name, bits per weight)
FTYPES = (
    ("IQ2_XXS", "IQ2_XXS - 2.0625 bpw", 2.06), ("IQ2_XS", "IQ2_XS - 2.3125 bpw", 2.31),
    ("IQ2_S", "IQ2_S - 2.5 bpw", 2.5), ("IQ2_M", "IQ2_M - 2.7 bpw", 2.7),
    ("IQ3_XXS", "IQ3_XXS - 3.0625 bpw", 3.06), ("IQ3_XS", "IQ3_XS - 3.3 bpw", 3.3),
    ("IQ3_S", "IQ3_S - 3.4375 bpw", 3.44), ("IQ3_M", "IQ3_S mix - 3.66 bpw", 3.66),
    ("IQ4_NL", "IQ4_NL - 4.5 bpw", 4.5), ("IQ4_XS", "IQ4_XS - 4.25 bpw", 4.25),
    ("Q2_K", "Q2_K - Medium", 2.96), ("Q3_K_S", "Q3_K - Small", 3.5), ("Q3_K_M", "Q3_K - Medium", 3.9),
    ("Q4_K_S", "Q4_K - Small", 4.6), ("Q4_K_M", "Q4_K - Medium", 4.85), ("Q5_K_M", "Q5_K - Medium", 5.7),
    ("Q6_K", "Q6_K", 6.56), ("Q8_0", "Q8_0", 8.5), ("BF16", "BF16", 16.0), ("F16", "F16", 16.0),
)


def env_float(name: str, default: float) -> float:
    try:
        return float(os.environ[name])
    except (KeyError, ValueError):
        return default


def now_us() -> int:
    return time.monotonic_ns() // 1000


def dumps(obj: Any) -> bytes:
    """Compact, insertion-ordered JSON with raw non-ASCII, like the real server's ordered_json dump."""
    return json.dumps(obj, separators=(",", ":"), ensure_ascii=False).encode("utf-8", "replace")


def random_string() -> str:
    return "".join(random.SystemRandom().choices(ALNUM, k=32))


# ------------------------------------------------------------------------------------------------ configuration


class Config:
    """The command line plus LLAMA_ARG_* environment, reduced to what the fake needs."""

    def __init__(self) -> None:
        self.argv: list[str] = []
        self.unknown: list[str] = []
        self.model = ""
        self.alias = ""
        self.host = "127.0.0.1"
        self.port: int | None = None
        self.ctx = 0
        self.parallel = 1
        self.draft = ""
        self.spec_type = ""
        self.draft_max = 3
        self.verbosity = 3
        self.log_prefix = True
        self.log_timestamps = True
        self.log_disable = False
        self.api_keys: list[str] = []
        self.threads = 0
        self.cache_ram = 8192
        self.fit = True
        self.fit_target = 1024
        self.temp, self.top_k, self.top_p, self.min_p = 0.8, 40, 0.95, 0.05
        self.presence_penalty, self.repeat_penalty, self.frequency_penalty = 0.0, 1.0, 0.0
        self.n_predict = -1
        self.metrics = False
        self.slots = True
        self.props = False
        self.ui = True
        self.kv_unified = False
        self.warmup = True
        self.version = False
        self.help = False
        self.chat_kwargs: dict[str, Any] = {}
        self.engine = "upstream"


VALUE_FLAGS = {
    "-m": "model", "--model": "model", "-a": "alias", "--alias": "alias", "--host": "host", "--port": "port",
    "-c": "ctx", "--ctx-size": "ctx", "-np": "parallel", "--parallel": "parallel",
    "-md": "draft", "--model-draft": "draft", "--spec-draft-model": "draft",
    "--spec-type": "spec_type", "--spec-draft-n-max": "draft_max", "--draft-max": "draft_max", "--draft": "draft_max",
    "--draft-n": "draft_max", "-lv": "verbosity", "--verbosity": "verbosity", "--log-verbosity": "verbosity",
    "--api-key": "api_keys", "-t": "threads", "--threads": "threads", "--cache-ram": "cache_ram",
    "-cram": "cache_ram", "-fit": "fit", "--fit": "fit", "--fit-target": "fit_target", "-fitt": "fit_target",
    "--temp": "temp", "--temperature": "temp", "--top-k": "top_k", "--top-p": "top_p", "--min-p": "min_p",
    "--presence-penalty": "presence_penalty", "--repeat-penalty": "repeat_penalty",
    "--frequency-penalty": "frequency_penalty", "-n": "n_predict", "--n-predict": "n_predict",
    "--predict": "n_predict", "--chat-template-kwargs": "chat_kwargs",
}
SWITCHES = {
    "--metrics": ("metrics", True), "--no-metrics": ("metrics", False), "--slots": ("slots", True),
    "--no-slots": ("slots", False), "--props": ("props", True), "--ui": ("ui", True), "--webui": ("ui", True),
    "--no-ui": ("ui", False), "--no-webui": ("ui", False), "--kv-unified": ("kv_unified", True),
    "-kvu": ("kv_unified", True), "--no-kv-unified": ("kv_unified", False), "--warmup": ("warmup", True),
    "--no-warmup": ("warmup", False), "-v": ("verbosity", INT_MAX), "--verbose": ("verbosity", INT_MAX),
    "--log-verbose": ("verbosity", INT_MAX), "--log-prefix": ("log_prefix", True),
    "--no-log-prefix": ("log_prefix", False), "--log-timestamps": ("log_timestamps", True),
    "--no-log-timestamps": ("log_timestamps", False), "--log-disable": ("log_disable", True),
    "--version": ("version", True), "-h": ("help", True), "--help": ("help", True), "--usage": ("help", True),
}
ENV_FLAGS = {
    "LLAMA_ARG_MODEL": "model", "LLAMA_ARG_ALIAS": "alias", "LLAMA_ARG_HOST": "host", "LLAMA_ARG_PORT": "port",
    "LLAMA_ARG_CTX_SIZE": "ctx", "LLAMA_ARG_N_PARALLEL": "parallel", "LLAMA_ARG_LOG_VERBOSITY": "verbosity",
    "LLAMA_ARG_THREADS": "threads", "LLAMA_ARG_ENDPOINT_METRICS": "metrics", "LLAMA_API_KEY": "api_keys",
    "LLAMA_ARG_LOG_PREFIX": "log_prefix", "LLAMA_ARG_LOG_TIMESTAMPS": "log_timestamps",
}


def _assign(cfg: Config, attr: str, raw: str) -> None:
    """Store one option value; a value that does not parse leaves the default in place."""
    current = getattr(cfg, attr)
    try:
        if attr == "api_keys":
            cfg.api_keys.extend(k for k in raw.split(",") if k)
        elif attr == "fit":
            cfg.fit = raw.strip().lower() not in FALSY
        elif attr == "chat_kwargs":
            value = json.loads(raw)
            if isinstance(value, dict):
                cfg.chat_kwargs = value
        elif isinstance(current, bool):
            setattr(cfg, attr, raw.strip().lower() in TRUTHY)
        elif isinstance(current, int):
            setattr(cfg, attr, int(raw))
        elif isinstance(current, float):
            setattr(cfg, attr, float(raw))
        elif attr == "port":
            cfg.port = int(raw)
        else:
            setattr(cfg, attr, raw)
    except ValueError:
        pass


def parse_args(argv: list[str], env: dict[str, str]) -> Config:
    cfg = Config()
    cfg.argv = list(argv)
    seen: set[str] = set()
    i, n = 0, len(argv)
    while i < n:
        token = argv[i]
        i += 1
        if not token.startswith("-") or token == "-":
            continue  # a stray positional argument is ignored
        name, eq, inline = token.partition("=") if token.startswith("--") else (token, "", "")
        if name in SWITCHES:
            attr, setting = SWITCHES[name]
            setattr(cfg, attr, setting)
            seen.add(attr)
            continue
        # every other flag may take a value: the next token, unless it looks like another flag
        value: str | None = None
        if eq:
            value = inline
        elif i < n and (not argv[i].startswith("-") or NUMBER_RE.match(argv[i])):
            value = argv[i]
            i += 1
        attr = VALUE_FLAGS.get(name, "")
        if attr and value is not None:
            _assign(cfg, attr, value)
            seen.add(attr)
        elif not attr:
            cfg.unknown.append(token)
    for var, attr in ENV_FLAGS.items():
        if attr not in seen and env.get(var):
            _assign(cfg, attr, env[var])
    forced = env.get("FAKE_ENGINE", "").strip().lower()
    if forced in ENGINES:
        cfg.engine = forced
    elif env.get("FLE_MOE_CACHE_BUDGET_MIB"):
        cfg.engine = "fle"
    if cfg.port is None:
        cfg.port = int(ENGINES[cfg.engine]["port"])
    if cfg.parallel < 0:  # "auto"
        cfg.parallel, cfg.kv_unified = 4, True
    cfg.parallel = max(1, cfg.parallel)
    return cfg


def build_strings() -> tuple[str, str]:
    system = platform.system() or "Linux"
    machine = platform.machine() or "x86_64"
    compiler = {"Darwin": "AppleClang 17.0.0.17000603", "Windows": "MSVC 19.44.35219.0"}.get(system, "GNU 13.3.0")
    return compiler, f"{system} {machine}"


# ------------------------------------------------------------------------------------------------ model file


def read_gguf_meta(path: str, limit: int = 16 << 20) -> dict[str, Any]:
    """Scalar and string metadata of a GGUF file, as far as it can be read; {} for anything else."""
    out: dict[str, Any] = {}
    scalars = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f", 7: "<?", 10: "<Q", 11: "<q", 12: "<d"}
    try:
        with open(path, "rb") as fh:
            buf = fh.read(limit)
        if buf[:4] != b"GGUF":
            return out
        version, _n_tensors, n_kv = struct.unpack_from("<IQQ", buf, 4)
        if version < 2 or n_kv > 100_000:
            return out

        def skip(kind: int, at: int) -> int:
            if kind in scalars:
                return at + struct.calcsize(scalars[kind])
            if kind == 8:
                return at + 8 + struct.unpack_from("<Q", buf, at)[0]
            if kind == 9:
                inner, count = struct.unpack_from("<IQ", buf, at)
                at += 12
                if inner in scalars:
                    return at + count * struct.calcsize(scalars[inner])
                for _ in range(count):
                    at = skip(inner, at)
                return at
            raise ValueError("unknown GGUF value type")

        pos = 24
        for _ in range(n_kv):
            key_len = struct.unpack_from("<Q", buf, pos)[0]
            key = buf[pos + 8:pos + 8 + key_len].decode("utf-8", "replace")
            pos += 8 + key_len
            kind = struct.unpack_from("<I", buf, pos)[0]
            pos += 4
            if kind in scalars:
                out[key] = struct.unpack_from(scalars[kind], buf, pos)[0]
            elif kind == 8:
                size = struct.unpack_from("<Q", buf, pos)[0]
                out[key] = buf[pos + 8:pos + 8 + size].decode("utf-8", "replace")
            pos = skip(kind, pos)
    except (OSError, struct.error, ValueError, IndexError, OverflowError, MemoryError):
        pass
    return out


class ModelInfo:
    def __init__(self, path: str) -> None:
        self.path = path
        self.file = Path(path) if path else None
        self.exists = bool(self.file and self.file.is_file())
        self.size = self.file.stat().st_size if self.exists and self.file else 0
        meta = read_gguf_meta(path) if self.exists else {}
        self.arch = str(meta.get("general.architecture") or "qwen35moe")

        def num(key: str, default: int) -> int:
            try:
                return int(meta.get(f"{self.arch}.{key}") or default)
            except (TypeError, ValueError):
                return default

        self.name = str(meta.get("general.name") or (self.file.stem if self.file else "model"))
        self.n_ctx_train = num("context_length", 262144)
        self.n_embd = num("embedding_length", 2048)
        self.n_layer = num("block_count", 40)
        self.n_expert = num("expert_count", 512)
        self.n_expert_used = num("expert_used_count", 10)
        self.n_vocab = 151936
        base = (self.file.name if self.file else "").upper()
        self.ftype, self.bpw = "Q4_K - Medium", 4.85
        for token, text, bpw in FTYPES:
            if re.search(rf"(^|[^A-Z0-9]){token}($|[^A-Z0-9])", base):
                self.ftype, self.bpw = text, bpw
                break
        # a stub file is far smaller than any real model: pretend to be a 200B-parameter model of that file type
        real = self.size > 100 << 20
        self.n_params = int(self.size * 8 / self.bpw) if real else 212_000_000_000
        self.size_pretended = self.size if real else int(self.n_params * self.bpw / 8)


# ------------------------------------------------------------------------------------------------ logging


class Log:
    """stderr logger with the llama.cpp line shape (common/log.cpp) and its verbosity filter."""

    def __init__(self, threshold: int, prefix: bool = True, timestamps: bool = True, disabled: bool = False) -> None:
        self.threshold = threshold
        self.prefix = prefix
        self.timestamps = timestamps
        self.disabled = disabled
        self._lock = threading.Lock()

    def _emit(self, verbosity: int, letter: str, text: str) -> None:
        if self.disabled or verbosity > self.threshold:
            return
        head = ""
        if self.prefix:
            if self.timestamps:
                us = int((time.monotonic() - T0) * 1_000_000)
                head = f"{us // 60_000_000}.{us // 1_000_000 % 60:02d}.{us // 1000 % 1000:03d}.{us % 1000:03d} "
            head += letter + " "
        with self._lock:
            sys.stderr.write(f"{head}{text}\n")
            sys.stderr.flush()

    def err(self, text: str) -> None:  # LOG_ERR and LLAMA_LOG_ERROR
        self._emit(1, "E", text)

    def wrn(self, text: str) -> None:  # LOG_WRN and LLAMA_LOG_WARN
        self._emit(2, "W", text)

    def inf(self, text: str) -> None:  # LOG_INF: visible by default
        self._emit(3, "I", text)

    def trc(self, text: str) -> None:  # LOG_TRC and LLAMA_LOG_INFO: hidden unless -lv 4
        self._emit(4, "I", text)

    def cont(self, text: str) -> None:  # LLAMA_LOG_CONT: no prefix, no newline
        if not self.disabled and self.threshold >= 4:
            with self._lock:
                sys.stderr.write(text)
                sys.stderr.flush()


def srv(fn: str) -> str:
    return f"srv  {fn:>12.12}: "


def cmn(fn: str) -> str:
    return f"cmn  {fn:>12.12}: "


def spc(fn: str) -> str:
    return f"spec {fn:>12.12}: "


def slt(fn: str, slot_id: int, task_id: int) -> str:
    return f"slot {fn:>12.12}: id {slot_id:2d} | task {task_id} | "


# ------------------------------------------------------------------------------------------------ state


class Bucket:
    __slots__ = ("count", "steps", "time_us")

    def __init__(self) -> None:
        self.count = self.steps = self.time_us = 0

    def add(self, count: int, steps: int, time_us: int) -> None:
        self.count += count
        self.steps += steps
        self.time_us += time_us

    def per_second(self) -> float:
        return self.steps / self.time_us * 1e6 if self.time_us > 0 else 0.0


class Metrics:
    """server_metrics (server-common.h) and its Prometheus rendering (server-task.cpp)."""

    def __init__(self) -> None:
        self.t_start = now_us()
        self.prompt, self.predict = Bucket(), Bucket()
        self.prompt_bucket, self.predict_bucket = Bucket(), Bucket()
        self.n_prompt_cached = self.n_tokens_max = self.n_decode = self.n_busy_slots = 0
        self.n_draft_tokens = self.n_draft_accepted = self.n_draft_verif_steps = 0
        self.n_accepted_per_pos: list[int] = []

    def add_prompt(self, n_tokens: int, t_us: int) -> None:
        self.prompt.add(n_tokens, n_tokens, t_us)
        self.prompt_bucket.add(n_tokens, n_tokens, t_us)

    def reset_bucket(self) -> None:
        self.prompt_bucket, self.predict_bucket = Bucket(), Bucket()

    def render(self, n_processing: int, n_deferred: int) -> str:
        counters = (
            ("prompt_tokens_total", "Number of prompt tokens processed, excluding cached tokens", self.prompt.count),
            ("prompt_tokens_cached_total", "Number of prompt tokens reused from the cache", self.n_prompt_cached),
            ("prompt_seconds_total", "Total time spent processing prompts", self.prompt.time_us / 1e6),
            ("tokens_predicted_total", "Number of generation tokens processed", self.predict.count),
            ("tokens_predicted_seconds_total", "Total time spent generating tokens", self.predict.time_us / 1e6),
            ("n_decode_total", "Total number of llama_decode() calls, excluding speculative decoding and "
                               "multimodal decoding", self.n_decode),
            ("n_tokens_max", "Largest observed sequence length (prompt + generation)", self.n_tokens_max),
            ("spec_decode_num_draft_tokens_total", "Speculative: Total draft tokens generated", self.n_draft_tokens),
            ("spec_decode_num_accepted_tokens_total", "Speculative: Total draft tokens accepted by the target model",
             self.n_draft_accepted),
            ("spec_decode_num_drafts_total", "Speculative: Total speculative decoding verification steps",
             self.n_draft_verif_steps),
        )
        gauges = (
            ("prompt_tokens_seconds", "Average prompt throughput in tokens/s", self.prompt_bucket.per_second()),
            ("predicted_tokens_seconds", "Average generation throughput in tokens/s",
             self.predict_bucket.per_second()),
            ("requests_processing", "Number of requests processing", n_processing),
            ("requests_deferred", "Number of requests deferred", n_deferred),
            ("n_busy_slots_per_decode", "Average number of busy slots per llama_decode() call",
             self.n_busy_slots / max(self.n_decode, 1)),
        )
        lines: list[str] = []
        for kind, items in (("counter", counters), ("gauge", gauges)):
            for name, desc, value in items:
                lines += [f"# HELP llamacpp:{name} {desc}", f"# TYPE llamacpp:{name} {kind}",
                          f"llamacpp:{name} {value:g}"]
        if self.n_accepted_per_pos:
            name = "spec_decode_num_accepted_tokens_per_pos_total"
            lines += [f"# HELP llamacpp:{name} Accepted tokens per draft position", f"# TYPE llamacpp:{name} counter"]
            lines += [f'llamacpp:{name}{{position="{i}"}} {v}' for i, v in enumerate(self.n_accepted_per_pos)]
        return "\n".join(lines) + "\n"


class Slot:
    def __init__(self, slot_id: int, n_ctx: int) -> None:
        self.id = slot_id
        self.n_ctx = n_ctx
        self.busy = False
        self.t_last_used = -1  # microseconds on a monotonic clock, -1 = never used
        self.cache_text = ""  # rendered prompt plus generated text, what the KV cache holds
        self.cache_tokens = 0
        self.saved_text = ""  # what the FLE early save wrote into the RAM cache last
        self.task: dict[str, Any] | None = None  # the running task, or the previous one (task_prev)


class Cancelled(Exception):
    """The client went away, or the server is shutting down."""


def client_gone(sock: socket.socket) -> bool:
    """True once the peer has closed its end (FIN seen) or the connection is broken."""
    try:
        readable, _, _ = select.select([sock], [], [], 0)
        return bool(readable) and sock.recv(1, socket.MSG_PEEK) == b""
    except ValueError:  # descriptor too large for select(): assume the client is still there
        return False
    except OSError:
        return True


def words(sentences: tuple[str, ...], offset: int, n: int) -> list[str]:
    out: list[str] = []
    i = offset
    while len(out) < n:
        out.extend(sentences[i % len(sentences)].split())
        i += 1
    return out[:n]


def flatten_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, dict) and isinstance(part.get("text"), str):
                parts.append(part["text"])
            elif isinstance(part, dict) and part.get("type") == "image_url":
                parts.append("<__media__>")
            elif isinstance(part, str):
                parts.append(part)
        return "\n".join(parts)
    return ""


def has_image(messages: list[Any]) -> bool:
    """Does any message carry a picture part (OpenAI `image_url`)?"""
    return any(isinstance(m, dict) and isinstance(m.get("content"), list)
               and any(isinstance(p, dict) and p.get("type") == "image_url" for p in m["content"]) for m in messages)


CALL_RE = re.compile(r"CALL_TOOL[ \t]+(\S+)(?:[ \t]+(\{[^\n]*\}))?")


def requested_tool_calls(req: dict[str, Any], last_user: str) -> list[tuple[str, str]]:
    """The tools this fake "decides" to call: one for every `CALL_TOOL name {json arguments}` in the user's last
    message, if the request offers a tool of that name and the last message is the user's."""
    tools, messages = req.get("tools"), req.get("messages") or []
    if not isinstance(tools, list) or not tools or req.get("tool_choice") == "none":
        return []
    if not messages or not isinstance(messages[-1], dict) or messages[-1].get("role") != "user":
        return []
    offered = {t["function"].get("name") for t in tools if isinstance(t, dict) and isinstance(t.get("function"), dict)}
    calls = []
    for name, args in CALL_RE.findall(last_user):
        if name in offered:
            try:
                json.loads(args or "{}")
            except ValueError:
                args = ""
            calls.append((name, args or "{}"))
    return calls


def forced_format(req: dict[str, Any]) -> bool:
    """Does the request force the shape of the answer (JSON mode, JSON schema, grammar)?"""
    fmt = req.get("response_format")
    if isinstance(fmt, dict) and fmt.get("type") not in (None, "text"):
        return True
    return bool(req.get("json_schema") or req.get("grammar"))


def lcp(a: str, b: str) -> int:
    """Length of the longest common prefix."""
    n = min(len(a), len(b))
    if a[:n] == b[:n]:
        return n
    lo, hi = 0, n
    while lo < hi:  # binary search on the prefix length
        mid = (lo + hi + 1) // 2
        if a[:mid] == b[:mid]:
            lo = mid
        else:
            hi = mid - 1
    return lo


def render_prompt(messages: list[Any], think: bool) -> tuple[str, str]:
    """The chat template of the pretend model; returns the prompt and the last user message."""
    parts, last_user = [], ""
    for msg in messages:
        role, text = ("user", flatten_content(msg))
        if isinstance(msg, dict):
            role, text = str(msg.get("role", "user")), flatten_content(msg.get("content"))
        if role == "user":
            last_user = text
        parts.append(f"<|im_start|>{role}\n{text}<|im_end|>\n")
    parts.append("<|im_start|>assistant\n" + ("<think>\n" if think else "<think>\n\n</think>\n\n"))
    return "".join(parts), last_user


def make_tokens(seed: int, last_user: str, think: bool, limit: int, ignore_eos: bool,
                lead: str = "") -> tuple[list, str]:
    """The deterministic token list [(piece, channel)] and the finish reason; channel is reasoning|content|hidden.
    `lead` replaces the first sentence of the answer."""
    n_think = max(0, int(env_float("FAKE_THINK_TOKENS", 40))) if think else 0
    n_answer = max(0, int(env_float("FAKE_ANSWER_TOKENS", 32 + seed % 24)))
    snippet = re.sub(r"\s+", " ", last_user).strip().replace('"', "'")[:40]
    out: list[tuple[str, str]] = []
    if think:
        out += [(w if i == 0 else " " + w, "reasoning") for i, w in enumerate(words(THINK_SENTENCES, seed, n_think))]
        out.append(("", "hidden"))  # </think>
    text = lead.split() if lead else f'Fake reply to "{snippet}".'.split() if snippet else ["Fake", "reply."]
    text += words(ANSWER_SENTENCES, seed, max(0, n_answer - len(text)))
    out += [(w if i == 0 else " " + w, "content") for i, w in enumerate(text[:n_answer])]
    if ignore_eos:  # the model never stops by itself: keep going until the limit
        k = 0
        while len(out) < limit:
            out.append((" " + ANSWER_SENTENCES[k % len(ANSWER_SENTENCES)].split()[k % 4], "content"))
            k += 1
        return out[:limit], "length"
    if len(out) + 1 > limit:  # no room left for the end-of-sequence token
        return out[:limit], "length"
    out.append(("", "hidden"))  # EOS
    return out, "stop"


def tool_call_tokens(seed: int, think: bool, calls: list[tuple[str, str]]) -> tuple[list, str]:
    """Reasoning, then the tool calls: the channel `toolname:I` carries a call's name, `toolargs:I` its arguments."""
    n_think = max(0, int(env_float("FAKE_THINK_TOKENS", 40))) if think else 0
    out: list[tuple[str, str]] = []
    if think:
        out += [(w if i == 0 else " " + w, "reasoning") for i, w in enumerate(words(THINK_SENTENCES, seed, n_think))]
        out.append(("", "hidden"))  # </think>
    for i, (name, args) in enumerate(calls):
        out.append((name, f"toolname:{i}"))
        out += [(args[k:k + 8], f"toolargs:{i}") for k in range(0, len(args), 8)]
    out.append(("", "hidden"))  # EOS
    return out, "tool_calls"


def json_tokens(seed: int, think: bool, text: str) -> tuple[list, str]:
    """An answer that is JSON only, as the real server gives under a forced answer format."""
    n_think = max(0, int(env_float("FAKE_THINK_TOKENS", 40))) if think else 0
    out: list[tuple[str, str]] = []
    if think:
        out += [(w if i == 0 else " " + w, "reasoning") for i, w in enumerate(words(THINK_SENTENCES, seed, n_think))]
        out.append(("", "hidden"))
    out += [(text[k:k + 6], "content") for k in range(0, len(text), 6)]
    out.append(("", "hidden"))
    return out, "stop"


def plan_timeline(n_tokens: int, t_prompt: float, rate: float, jitter: float, draft_max: int,
                  rng: random.Random) -> tuple[list[float], list[tuple[int, int, int]]]:
    """Virtual time of every token, plus the speculative rounds as (tokens so far, drafted, accepted)."""
    times = [t_prompt] if n_tokens > 0 else []  # the first token comes with the last prompt batch
    rounds: list[tuple[int, int, int]] = []
    t, done = t_prompt, 1
    while done < n_tokens:
        accepted = 0
        step = 1
        if draft_max:
            while accepted < draft_max and rng.random() < ACCEPT_P:
                accepted += 1
            step = min(accepted + 1, n_tokens - done)
        t += step / rate * (1 + jitter * (2 * rng.random() - 1))
        done += step
        times.extend([t] * step)
        if draft_max:
            rounds.append((done, draft_max, min(accepted, step - 1)))
    return times, rounds


def spec_stats(rounds: list[tuple[int, int, int]], n_sent: int, width: int) -> dict[str, Any]:
    """Speculative statistics for the first n_sent tokens of a request."""
    stats: dict[str, Any] = {"drafted": 0, "accepted": 0, "verif": 0, "acc_drafts": 0, "per_pos": [0] * width,
                             "width": width}
    for end, drafted, accepted in rounds:
        if end > n_sent:
            break
        stats["drafted"] += drafted
        stats["accepted"] += accepted
        stats["verif"] += 1
        stats["acc_drafts"] += 1 if accepted else 0
        for j in range(accepted):
            stats["per_pos"][j] += 1
    return stats


# ------------------------------------------------------------------------------------------------ the fake server


class Fake:
    def __init__(self, cfg: Config, log: Log) -> None:
        self.cfg, self.log = cfg, log
        self.engine = ENGINES[cfg.engine]
        self.is_fle = cfg.engine == "fle"
        self.compiler, self.target = build_strings()
        self.model = ModelInfo(cfg.model)
        self.aliases = sorted({a for a in cfg.alias.split(",") if a})
        self.model_name = self.aliases[0] if self.aliases else cfg.model
        self.build_info = f"b{self.engine['build']}-{self.engine['commit']}"
        total = cfg.ctx if cfg.ctx > 0 else min(self.model.n_ctx_train, 32768)
        total = (total + 255) // 256 * 256
        if cfg.kv_unified:
            self.n_ctx_slot = total
        else:
            self.n_ctx_slot = (total // cfg.parallel + 255) // 256 * 256
            total = self.n_ctx_slot * cfg.parallel
        self.n_ctx_total = total
        self.load_seconds = max(0.0, env_float("FAKE_LOAD_SECONDS", 1.0))
        self.tok_rate = max(0.1, env_float("FAKE_TOK_PER_SEC", 80.0))
        self.prompt_rate = max(0.1, env_float("FAKE_PROMPT_TOK_PER_SEC", 2000.0))
        self.jitter = max(0.0, min(0.5, env_float("FAKE_JITTER", 0.04)))
        if cfg.spec_type not in ("", "none"):
            self.spec_type = cfg.spec_type
        elif cfg.draft:
            self.spec_type = "draft-mtp" if "mtp" in cfg.draft.lower() else "draft-simple"
        else:
            self.spec_type = "none"
        self.draft_width = max(1, cfg.draft_max) if self.spec_type != "none" else 0
        self.vision = os.environ.get("FAKE_VISION", "").strip().lower() in TRUTHY
        self.cache_budget = int(env_float("FLE_MOE_CACHE_BUDGET_MIB", 0)) if self.is_fle else 0
        self.cache_slot_mib = 3.9  # a made-up size for one expert slice
        self.early_save = self.is_fle and env_float("FLE_SLOT_EARLY_SAVE", 0) != 0 and cfg.cache_ram > 0
        self.media_marker = "<__media_" + random_string() + "__>"
        self.stop = threading.Event()
        self.ready = threading.Event()
        self.terminating = False
        self.cv = threading.Condition()
        self.queue: deque[object] = deque()
        self.slots = [Slot(i, self.n_ctx_slot) for i in range(cfg.parallel)]
        self.metrics = Metrics()
        self.spec_total: dict[str, Any] = {"begin": 0, "rounds": 0, "acc_drafts": 0, "drafted": 0, "accepted": 0,
                                           "per_pos": [0] * self.draft_width}
        self.next_task = 0
        self.n_reused = 0
        self.moe_steps = 0
        self.moe_hits = 0.0
        self.moe_misses = 0.0
        self.early_n = 0
        self.early_ms = 0.0
        self.t_ready = 0.0
        self.http: ThreadingServer | None = None

    # -------------------------------------------------------------------------------------- small helpers

    def new_task_id(self) -> int:
        with self.cv:
            task_id = self.next_task
            self.next_task += 1
            return task_id

    def sleep_until(self, deadline: float, gone: Any = None) -> None:
        """Sleep until a perf_counter deadline; raises Cancelled when the client leaves or the server stops."""
        while True:
            left = deadline - time.perf_counter()
            if self.stop.is_set() or (gone is not None and gone()):
                raise Cancelled
            if left <= 0:
                return
            time.sleep(min(left, 0.05))

    def at(self, t_begin: float, fraction: float) -> None:
        """During the load: wait until a fraction of the load time has passed (SIGTERM interrupts it)."""
        while True:
            left = t_begin + fraction * self.load_seconds - time.monotonic()
            if left <= 0:
                return
            if self.stop.wait(min(left, 0.1)):
                raise Cancelled

    # -------------------------------------------------------------------------------------- startup

    def print_params_info(self) -> None:
        log, cfg = self.log, self.cfg
        fn = "common_params_print_info"
        log.trc(cmn(fn) + f"{fn}: build {self.engine['build']} ({self.engine['commit']}) with {self.compiler} "
                          f"for {self.target}")
        log.inf(cmn(fn) + f"{fn}: verbosity = {log.threshold} (adjust with the `-lv N` CLI arg)")
        log.trc(cmn(fn) + "device_info:")
        log.trc(cmn(fn) + "  - CPU     : Fake CPU (65536 MiB, 32768 MiB free)")
        n_hw = os.cpu_count() or 1
        n_threads = cfg.threads if cfg.threads > 0 else n_hw
        log.trc(cmn(fn) + f"system_info: n_threads = {n_threads} (n_threads_batch = {n_threads}) / {n_hw} | CPU : "
                          "SSE3 = 1 | SSSE3 = 1 | AVX = 1 | AVX2 = 1 | F16C = 1 | FMA = 1 | LLAMAFILE = 1 | OPENMP = 1 "
                          "| REPACK = 1 |")

    def warn_cors(self) -> None:
        if self.cfg.api_keys:
            return
        head = srv("llama_server")
        if self.is_fle:
            self.log.wrn(head + "-----------------")
            self.log.wrn(head + "CORS is set to allow all origins ('*') and no API key is set")
            self.log.wrn(head + "this can be a security risk (cross-origin attacks)")
            self.log.wrn(head + "more info: https://github.com/ggml-org/llama.cpp/pull/25655")
            self.log.wrn(head + "-----------------")
        else:
            self.log.wrn(head + "security: no API key is set and CORS allows all origins "
                                "(see https://github.com/ggml-org/llama.cpp/pull/25655)")

    def clean_up_line(self) -> None:
        self.log.inf(srv("operator()") + "operator(): cleaning up before exit...")

    def listening_address(self, port: int) -> str:
        host = self.cfg.host
        shown = f"[{host}]" if ":" in host and not host.startswith("[") else host
        return f"http://{shown}:{port}"

    def open_failure(self, path: str, why: str) -> None:
        """The error chain of a model file the loader cannot open (gguf.cpp, llama.cpp, llama-model-loader.cpp)."""
        log = self.log
        if why == "missing":
            log.err(f"gguf_init_from_file: failed to open GGUF file '{path}' (No such file or directory)")
        elif why == "empty":
            log.err("gguf_init_from_reader: failed to read magic")
        else:
            magic = "".join(c if c.isprintable() and ord(c) < 128 else "?" for c in why)
            log.err(f"gguf_init_from_reader: invalid magic characters: '{magic}', expected 'GGUF'")
        log.err(f"llama_model_load: error loading model: llama_model_loader: failed to load model from {path}")
        log.err("llama_model_load_from_file_impl: failed to load model")

    def check_file(self, path: str) -> str:
        """'' when the loader would accept the file, otherwise the reason."""
        p = Path(path)
        if not p.is_file():
            return "missing"
        if os.environ.get("FAKE_STRICT_GGUF") == "1":
            try:
                with open(p, "rb") as fh:
                    head = fh.read(4)
            except OSError:
                return "missing"
            if not head:
                return "empty"
            if head != b"GGUF":
                return head.decode("latin-1")
        return ""

    def load_model(self) -> bool:
        """The load sequence. The caller treats Cancelled (SIGTERM during the load) as a normal shutdown."""
        log, cfg, m = self.log, self.cfg, self.model
        t0 = time.monotonic()
        log.inf(srv("load_model") + f"loading model '{cfg.model}'")
        log.trc(srv("load_model") + f"local path '{cfg.model}'")
        bad = self.check_file(cfg.model)
        if bad:
            self.open_failure(cfg.model, bad)
            log.err(cmn("common_init_") + f"failed to load model '{cfg.model}'")
            log.err(srv("load_model") + f"failed to load model, '{cfg.model}'")
            return False
        n_fit = cfg.fit_target
        if (self.is_fle and cfg.fit and self.cache_budget > 0
                and os.environ.get("FLE_MOE_CACHE_BUDGET_FIT") not in (None, "", "0")):
            n_fit += self.cache_budget
            log.inf(f"common_init_result: MoE cache budget {self.cache_budget} MiB added to the fit margin of "
                    f"device 0 (now {n_fit} MiB)")
        if cfg.fit:
            log.trc(cmn("common_init_") + "fitting params to device memory ...")
            log.trc(cmn("common_init_") + "(for bugs during this step try to reproduce them with -fit off, or "
                                          "provide --verbose logs if the bug only occurs with -fit on)")
        n_gib = m.size_pretended / 1024**3
        self.at(t0, 0.05)
        log.trc(f"llama_model_loader: loaded meta data with 52 key-value pairs and 733 tensors from {cfg.model} "
                "(version GGUF V3 (latest))")
        log.trc("llama_model_loader: Dumping metadata keys/values. Note: KV overrides do not apply in this output.")
        log.trc("llama_model_loader: - type  f32:  361 tensors")
        log.trc("llama_model_loader: - type iq3_s:  372 tensors")
        self.at(t0, 0.1)
        log.trc("print_info: file format = GGUF V3 (latest)")
        log.trc(f"print_info: file type   = {m.ftype}")
        log.trc(f"print_info: file size   = {n_gib:.2f} GiB ({m.bpw:.2f} BPW) ")
        log.trc(f"print_info: arch                  = {m.arch}")
        log.trc("print_info: vocab_only            = 0")
        log.trc(f"print_info: n_ctx_train           = {m.n_ctx_train}")
        log.trc(f"print_info: n_embd                = {m.n_embd}")
        log.trc(f"print_info: n_layer               = {m.n_layer}")
        log.trc(f"print_info: n_expert              = {m.n_expert}")
        log.trc(f"print_info: n_expert_used         = {m.n_expert_used}")
        log.trc("print_info: model type            = fake")
        log.trc(f"print_info: model params          = {m.n_params * 1e-9:.2f} B")
        log.trc(f"print_info: general.name          = {m.name}")
        self.at(t0, 0.2)
        if os.environ.get("FAKE_FAIL") == "oom":
            self.at(t0, 0.4)
            size = max(int(m.size_pretended * 0.9), 1 << 30)
            log.err(f"ggml_backend_cuda_buffer_type_alloc_buffer: allocating {size / 1048576:.2f} MiB on device 0: "
                    "cudaMalloc failed: out of memory")
            if self.is_fle:
                log.err(f"alloc_tensor_range: failed to allocate CUDA0 buffer of size {size}")
            else:
                log.err(f"ggml_backend_buft_alloc_buffer_n_default: failed to allocate CUDA0 buffer of size {size}")
            log.err("llama_model_load: error loading model: unable to allocate CUDA0 buffer")
            log.err("llama_model_load_from_file_impl: failed to load model")
            log.err(cmn("common_init_") + f"failed to load model '{cfg.model}'")
            log.err(srv("load_model") + f"failed to load model, '{cfg.model}'")
            return False
        log.trc("load_tensors: offloading output layer to GPU")
        log.trc(f"load_tensors: offloading {m.n_layer - 1} repeating layers to GPU")
        log.trc(f"load_tensors: offloaded {m.n_layer + 1}/{m.n_layer + 1} layers to GPU")
        log.trc(f"load_tensors:   CPU_Mapped model buffer size = {n_gib * 1024 * 0.7:8.2f} MiB")
        log.trc(f"load_tensors:        CUDA0 model buffer size = {n_gib * 1024 * 0.3:8.2f} MiB")
        if os.environ.get("FAKE_LOAD_DOTS") == "1":
            for k in range(50):
                self.at(t0, 0.2 + 0.6 * (k + 1) / 50)
                log.cont(".")
            log.cont("\n")
        self.at(t0, 0.8)
        if cfg.draft:
            bad = self.check_file(cfg.draft)
            log.inf(f"common_speculative_init_result: loading draft model '{cfg.draft}'")
            if bad:
                self.open_failure(cfg.draft, bad)
                log.err(f"common_speculative_init_result: failed to load draft model, '{cfg.draft}'")
                log.err(srv("load_model") + f"failed to load draft model, '{cfg.draft}'")
                return False
        elif self.spec_type == "draft-mtp":
            log.inf(f"common_speculative_init_result: creating MTP draft context against the target model "
                    f"'{cfg.model}'")
        kv_mib = self.n_ctx_total * m.n_layer * 0.0625
        log.trc(f"llama_context: n_seq_max             = {cfg.parallel}")
        log.trc(f"llama_context: n_ctx                 = {self.n_ctx_total}")
        log.trc(f"llama_context: n_ctx_seq             = {self.n_ctx_slot}")
        log.trc("llama_context: n_batch               = 2048")
        log.trc("llama_context: n_ubatch              = 512")
        log.trc("llama_context: causal_attn           = 1")
        log.trc("llama_context: flash_attn            = auto")
        log.trc(f"llama_context: kv_unified            = {'true' if cfg.kv_unified else 'false'}")
        log.trc("llama_context: freq_base             = 10000000.0")
        log.trc("llama_context: freq_scale            = 1")
        self.at(t0, 0.9)
        half = kv_mib / 2
        log.trc(f"llama_kv_cache: size = {kv_mib:7.2f} MiB ({self.n_ctx_total:6d} cells, {m.n_layer:3d} layers, "
                f"{cfg.parallel:2d}/{cfg.parallel} seqs), K (f16): {half:7.2f} MiB, V (f16): {half:7.2f} MiB")
        log.trc(f"llama_context:      CUDA0 compute buffer size = {512.0:8.2f} MiB")
        self.at(t0, 0.92)
        n_threads = cfg.threads if cfg.threads > 0 else (os.cpu_count() or 1)
        log.inf(cmn("init") + f"llama threadpool init, n_threads = {n_threads}")
        if self.cache_budget > 0:
            layers = m.n_layer
            slots = max(1, min(m.n_expert, int(self.cache_budget / (layers * self.cache_slot_mib))))
            vram = slots * layers * self.cache_slot_mib
            log.trc(f"operator(): MoE expert cache enabled: {layers} layers x {slots} slots, 1 inserts/step, "
                    f"{vram:.1f} MiB device memory")
            log.trc(f"operator(): [fle-moe-budget] EVIDENCE allocated {vram:.1f} MiB of budget {self.cache_budget} "
                    f"MiB ({100.0 * vram / self.cache_budget:.1f} %) in {layers} layers, usable slots per layer "
                    f"min/mean/max {slots}/{slots:.1f}/{slots} (\"x {slots} slots\" above is the maximum)")
        if cfg.warmup:
            log.trc(cmn("common_init_") + "warming up the model with an empty run - please wait ... "
                                          "(--no-warmup to disable)")
        if self.draft_width:
            log.trc(spc("common_specu") + f"adding speculative implementation '{self.spec_type}'")
        self.at(t0, 1.0)
        log.inf(srv("load_model") + f"initializing, n_slots = {cfg.parallel}, n_ctx_slot = {self.n_ctx_slot}, "
                                    f"kv_unified = '{'true' if cfg.kv_unified else 'false'}'")
        for slot in self.slots:
            log.trc(slt("init", slot.id, -1) + f"new slot, n_ctx = {slot.n_ctx}")
        return True

    def print_memory_breakdown(self) -> None:
        """common_memory_breakdown_print (common/fit.cpp). TRC level, so only visible with -lv 4."""
        if self.log.threshold < 4:
            return
        total, free = 32607, 4012
        model, context, compute = int(self.model.size_pretended / 1048576 * 0.3), 1500, 512
        own = model + context + compute
        rows = [
            ("header", ["memory breakdown [MiB]", "total", "free", "self", "model", "context", "compute",
                        "unaccounted"]),
            ("gpu", ["  - CUDA0 (Fake GPU)", str(total), str(free), str(own), str(model), str(context),
                     str(compute), str(total - free - own)]),
            ("other", ["  - Host", "", "", "1024", "512", "0", "512", ""]),
        ]
        templates = {
            "header": "{fn}: | {0} | {1}   {2}    {3}   {4}   {5}   {6}    {7} |",
            "gpu": "{fn}: | {0} | {1} = {2} + ({3} = {4} + {5} + {6}) + {7} |",
            "other": "{fn}: | {0} | {1}   {2}    {3} = {4} + {5} + {6}    {7} |",
        }
        for col in range(8):  # the first column is left aligned, the others right aligned
            width = max(len(cells[col]) for _, cells in rows)
            for _, cells in rows:
                cells[col] = cells[col].ljust(width) if col == 0 else cells[col].rjust(width)
        for kind, cells in rows:
            self.log.trc(templates[kind].format(*cells, fn="common_memory_breakdown_print"))

    # -------------------------------------------------------------------------------------- slots

    def acquire(self, prompt: str, n_prompt: int, id_slot: int, gone: Any) -> Slot:
        """Wait for a free slot (first come, first served) and pick one like get_available_slot()."""
        ticket = object()
        with self.cv:
            self.queue.append(ticket)
            try:
                while True:
                    if self.stop.is_set() or gone():
                        raise Cancelled
                    if self.queue[0] is ticket:
                        slot = self._pick(prompt, n_prompt, id_slot)
                        if slot is not None:
                            slot.busy = True
                            return slot
                    self.cv.wait(0.05)
            finally:
                self.queue.remove(ticket)
                self.cv.notify_all()

    def _pick(self, prompt: str, n_prompt: int, id_slot: int) -> Slot | None:
        log = self.log
        idle = [s for s in self.slots if not s.busy]
        wanted = next((s for s in self.slots if s.id == id_slot), None) if id_slot != -1 else None
        if wanted is not None:
            if wanted.busy:
                return None  # the real server defers the task until that slot is free
            log.inf(slt("get_availabl", wanted.id, -1) + f"selected slot by id ({id_slot})")
            return wanted
        thold = 0.1
        best: Slot | None = None
        best_sim = 0.0
        for s in idle:
            if not s.cache_text:
                continue
            sim = (lcp(s.cache_text, prompt) // 4) / max(n_prompt, 1)
            if sim > best_sim and sim > thold:
                best, best_sim = s, sim
        if best is not None:
            f_keep = best_sim * n_prompt / max(best.cache_tokens, 1)
            log.inf(slt("get_availabl", best.id, -1) + f"selected slot by LCP similarity, f_sim_best = "
                                                      f"{best_sim:.3f} (> {thold:.3f} thold), f_keep = {f_keep:.3f}")
            return best
        chosen, t_last = None, -1
        for s in idle:
            if chosen is None or s.t_last_used <= t_last:  # '<=': among equals the last slot wins, like the real code
                chosen, t_last = s, s.t_last_used
        if chosen is not None:
            log.inf(slt("get_availabl", chosen.id, -1) + f"selected slot by LRU, t_last = {t_last}")
        return chosen

    def release(self, slot: Slot, task_id: int, keep: tuple[str, int] | None) -> None:
        """slot.release(): log the line, remember the cache, free the slot."""
        with self.cv:
            n_tokens = keep[1] if keep else 0
            self.log.inf(slt("release", slot.id, task_id) + f"stop processing: n_tokens = {n_tokens}, truncated = 0")
            slot.t_last_used = now_us()
            if keep is not None:
                slot.cache_text, slot.cache_tokens = keep
            slot.busy = False
            idle = not any(s.busy for s in self.slots)
            self.cv.notify_all()
        if idle:
            self.log.trc(srv("update_slots") + "all slots are idle")
        if self.early_save and keep is not None:
            self.do_early_save(slot)

    def do_early_save(self, slot: Slot) -> None:
        ms = 0.8 + slot.cache_tokens * 0.02
        is_new = slot.cache_text != slot.saved_text
        slot.saved_text = slot.cache_text
        self.early_n += 1
        self.early_ms += ms
        what = "saved:" if is_new else "not copied (already in cache or rejected):"
        self.log.inf(slt("fle_early_sa", slot.id, -1)
                     + f"[fle-slot-early-save] {what} {slot.cache_tokens} tokens in {ms:.2f} ms "
                       f"(n = {self.early_n}, total {self.early_ms:.1f} ms)")

    # -------------------------------------------------------------------------------------- documents

    def params_json(self, req: dict[str, Any] | None = None, stream: bool = True, n_predict: int | None = None,
                    ignore_eos: bool = False, timings_per_token: bool = False) -> dict[str, Any]:
        """task_params::to_json(only_metrics = true), the shape used by /props and /slots."""
        cfg, req = self.cfg, req or {}

        def pick(key: str, default: Any) -> Any:
            value = req.get(key)
            return default if value is None else value

        n_pred = cfg.n_predict if n_predict is None else n_predict
        return {
            "seed": pick("seed", 4294967295), "temperature": pick("temperature", cfg.temp),
            "dynatemp_range": 0.0, "dynatemp_exponent": 1.0, "top_k": pick("top_k", cfg.top_k),
            "top_p": pick("top_p", cfg.top_p), "min_p": pick("min_p", cfg.min_p), "top_n_sigma": -1.0,
            "xtc_probability": 0.0, "xtc_threshold": 0.1, "typical_p": 1.0, "repeat_last_n": 64,
            "repeat_penalty": pick("repeat_penalty", cfg.repeat_penalty),
            "presence_penalty": pick("presence_penalty", cfg.presence_penalty),
            "frequency_penalty": pick("frequency_penalty", cfg.frequency_penalty), "dry_multiplier": 0.0,
            "dry_base": 1.75, "dry_allowed_length": 2, "dry_penalty_last_n": 64, "mirostat": 0,
            "mirostat_tau": 5.0, "mirostat_eta": 0.1, "adaptive_target": -1.0, "adaptive_decay": 0.9,
            "max_tokens": n_pred, "n_predict": n_pred, "n_keep": 0, "n_discard": 0, "ignore_eos": ignore_eos,
            "stream": stream, "n_probs": 0, "min_keep": 0, "chat_format": "Content-only",
            "reasoning_format": "deepseek", "reasoning_in_content": False, "generation_prompt": "",
            "samplers": SAMPLERS, "speculative.types": self.spec_type, "timings_per_token": timings_per_token,
            "post_sampling_probs": False, "backend_sampling": False, "lora": [],
        }

    def slot_json(self, slot: Slot) -> dict[str, Any]:
        out: dict[str, Any] = {"id": slot.id, "n_ctx": slot.n_ctx, "speculative": self.draft_width > 0,
                               "is_processing": slot.busy}
        task = slot.task
        if task:
            out["id_task"] = task["id"]
            out["n_prompt_tokens"] = task["n_tokens"]
            out["n_prompt_tokens_processed"] = task["n_processed"]
            out["n_prompt_tokens_cache"] = task["n_cached"]
            out["params"] = task["params"]
            out["next_token"] = [{"has_next_token": task["has_next"], "has_new_line": False,
                                  "n_remain": task["n_remain"], "n_decoded": task["n_decoded"]}]
        return out

    def props_json(self) -> dict[str, Any]:
        cfg = self.cfg
        return {
            "default_generation_settings": {"params": self.params_json(), "n_ctx": self.n_ctx_slot},
            "total_slots": cfg.parallel, "model_alias": self.model_name, "model_ftype": self.model.ftype,
            "model_path": cfg.model,
            "modalities": {"vision": self.vision, "video": False, "audio": False},
            "media_marker": self.media_marker,
            "endpoint_slots": cfg.slots, "endpoint_props": cfg.props, "endpoint_metrics": cfg.metrics, "ui": cfg.ui,
            "ui_settings": {}, "chat_template": CHAT_TEMPLATE, "chat_template_caps": CHAT_TEMPLATE_CAPS,
            "bos_token": "", "eos_token": "<|im_end|>", "build_info": self.build_info, "is_sleeping": False,
            "cors_proxy_enabled": False,
        }

    def models_json(self) -> dict[str, Any]:
        m = self.model
        return {
            "models": [{
                "name": self.model_name, "model": self.model_name, "modified_at": "", "size": "", "digest": "",
                "type": "model", "description": "", "tags": [""], "capabilities": ["completion"], "parameters": "",
                "details": {"parent_model": "", "format": "gguf", "family": "", "families": [""],
                            "parameter_size": "", "quantization_level": ""},
            }],
            "object": "list",
            "data": [{
                "id": self.model_name, "aliases": self.aliases, "tags": [], "object": "model",
                "architecture": {"input_modalities": ["text", "image"] if self.vision else ["text"],
                                 "output_modalities": ["text"]},
                "created": int(time.time()), "owned_by": "llamacpp",
                "meta": {"vocab_type": 2, "n_vocab": m.n_vocab, "n_ctx": self.n_ctx_slot,
                         "n_ctx_train": m.n_ctx_train, "n_embd": m.n_embd, "n_params": m.n_params,
                         "size": m.size_pretended, "ftype": m.ftype},
            }],
        }

    # -------------------------------------------------------------------------------------- accounting

    def count_moe_step(self) -> None:
        """FLE expert cache: one decode step, with the periodic hit-rate line of the cache code."""
        if self.cache_budget <= 0:
            return
        with self.cv:
            self.moe_steps += 1
            lookups = self.model.n_layer * self.model.n_expert_used
            miss_rate = 0.12 + 0.43 * 2.718281828 ** (-self.moe_steps / 6000)
            self.moe_misses += lookups * miss_rate
            self.moe_hits += lookups * (1 - miss_rate)
            if self.moe_steps % 512 == 0:
                hits, misses = int(self.moe_hits), int(self.moe_misses)
                rate = 100.0 * hits / (hits + misses) if hits + misses else 0.0
                self.log.trc(f"moe-cache: steps={self.moe_steps} hits={hits} misses={misses} hit-rate={rate:.1f}%")

    def add_prompt_metrics(self, cache_n: int, prompt_n: int, t_prompt: float) -> None:
        with self.cv:
            self.metrics.n_prompt_cached += cache_n
            self.metrics.add_prompt(prompt_n, int(t_prompt * 1e6))

    def log_timings(self, slot_id: int, task_id: int, t: dict[str, Any], spec: dict[str, Any]) -> None:
        """slot.print_timings()."""
        log = self.log
        pre = slt("print_timing", slot_id, task_id)
        n_gen, prompt_n = t["predicted_n"], t["prompt_n"]
        total_ms = t["prompt_ms"] + t["predicted_ms"]
        log.inf(pre + f"prompt eval time = {t['prompt_ms']:10.2f} ms / {prompt_n:5d} tokens "
                      f"({t['prompt_per_token_ms']:8.2f} ms per token, "
                      f"{t['prompt_per_second']:8.2f} tokens per second)")
        log.inf(pre + f"       eval time = {t['predicted_ms']:10.2f} ms / {n_gen:5d} tokens "
                      f"({t['predicted_per_token_ms']:8.2f} ms per token, "
                      f"{t['predicted_per_second']:8.2f} tokens per second)")
        log.inf(pre + f"      total time = {total_ms:10.2f} ms / {prompt_n + n_gen:5d} tokens")
        with self.cv:
            self.n_reused += max(0, n_gen - 2)
            n_reused = self.n_reused
        log.inf(pre + f"   graphs reused = {n_reused:10d}")
        if spec["drafted"] > 0:
            mean = 1.0 + spec["accepted"] / spec["verif"] if spec["verif"] > 0 else 1.0
            rates = ", ".join(f"{c / spec['verif']:.3f}" for c in spec["per_pos"]) if spec["verif"] else ""
            log.inf(pre + f"draft acceptance = {spec['accepted'] / spec['drafted']:.5f} ({spec['accepted']:5d} "
                          f"accepted / {spec['drafted']:5d} generated), mean len = {mean:5.2f}")
            log.trc(pre + f"     acc per pos = ({rates})")
        if self.draft_width:
            tot = self.spec_total
            calls = tot["rounds"]
            mean = 1.0 + tot["accepted"] / calls if calls else 1.0
            rates = ", ".join(f"{c / calls:.3f}" for c in tot["per_pos"]) if calls else ""
            extra = f", #mean acc len = {mean:.2f}, #acc rate/pos = ({rates})" if calls else ""
            log.trc(spc("common_specu") + f"statistics {self.spec_type:>16}: #calls(b,g,a) = {tot['begin']:4d} "
                                          f"{calls:6d} {calls:6d}, #gen drafts = {calls:6d}, #acc drafts = "
                                          f"{tot['acc_drafts']:5d}, #gen tokens = {tot['drafted']:6d}, #acc tokens = "
                                          f"{tot['accepted']:5d}{extra}")

    def settle(self, run: Completion, spec: dict[str, Any], timing: dict[str, Any] | None) -> None:
        """What the server does when a task ends: speculative totals, print_timings, metrics, slot release."""
        n_gen, t_last = run.sent, run.t_last(run.sent)
        with self.cv:
            if self.draft_width and run.prompt_done:
                tot = self.spec_total
                tot["begin"] += 1
                tot["rounds"] += spec["verif"]
                tot["acc_drafts"] += spec["acc_drafts"]
                tot["drafted"] += spec["drafted"]
                tot["accepted"] += spec["accepted"]
                for j, count in enumerate(spec["per_pos"]):
                    tot["per_pos"][j] += count
        if timing is not None:
            self.log_timings(run.slot.id, run.task_id, timing, spec)
        m = self.metrics
        with self.cv:
            if n_gen > 0:
                t_gen_us = max(1, int((t_last - run.t_prompt) * 1e6))
                m.predict.add(n_gen, n_gen - 1, t_gen_us)
                m.predict_bucket.add(n_gen, n_gen - 1, t_gen_us)
            m.n_draft_tokens += spec["drafted"]
            m.n_draft_accepted += spec["accepted"]
            m.n_draft_verif_steps += spec["verif"]
            if spec["width"]:
                m.n_accepted_per_pos += [0] * max(0, spec["width"] - len(m.n_accepted_per_pos))
                for j, count in enumerate(spec["per_pos"]):
                    m.n_accepted_per_pos[j] += count
            calls = -(-run.prompt_n // N_BATCH) if run.prompt_done else 0
            if not spec["width"]:  # speculative decoding is not counted in n_decode
                calls += max(0, n_gen - 1)
            m.n_decode += calls
            m.n_busy_slots += calls * max(1, sum(1 for s in self.slots if s.busy))
            m.n_tokens_max = max(m.n_tokens_max, run.n_prompt + max(0, n_gen - 1))
            self.next_task += max(0, n_gen - 1)  # the real queue spends a task id on internal work as well
        self.release(run.slot, run.task_id, (run.cache_text(), run.n_prompt + max(0, n_gen - 1)))

    # -------------------------------------------------------------------------------------- lifecycle

    def make_http(self) -> ThreadingServer:
        host = self.cfg.host.strip("[]")
        family = socket.AF_INET6 if ":" in host else socket.AF_INET
        httpd = ThreadingServer((host, int(self.cfg.port or 0)), Handler, family=family)
        httpd.fake = self
        return httpd

    def on_signal(self, signum: int, frame: Any) -> None:
        if self.terminating:
            sys.stderr.write("Received second interrupt, terminating immediately.\n")
            sys.stderr.flush()
            os._exit(1)
        self.terminating = True
        self.stop.set()

    def finish(self, code: int) -> NoReturn:
        sys.stderr.flush()
        sys.stdout.flush()
        os._exit(code)

    def run(self) -> int:
        log, cfg = self.log, self.cfg
        if not self.is_fle:
            log.inf(srv("llama_server") + "initializing ...")
        self.print_params_info()
        self.warn_cors()
        try:
            httpd = self.make_http()
        except OSError:
            log.err(srv("start") + f"couldn't bind HTTP server socket, hostname: {cfg.host}, port: {cfg.port}")
            self.clean_up_line()
            log.err(srv("llama_server") + "exiting due to HTTP server error")
            return 1
        self.http = httpd
        port = httpd.server_address[1]
        thread = threading.Thread(target=httpd.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True)
        thread.start()
        try:
            loaded = self.load_model()
        except Cancelled:
            loaded = True  # SIGTERM during the load: shut down below
        if not loaded:
            self.clean_up_line()
            httpd.shutdown()
            log.err(srv("llama_server") + "exiting due to model loading error")
            return 1
        if not self.stop.is_set():
            self.ready.set()
            self.t_ready = time.monotonic()
            log.inf(srv("llama_server") + "model loaded")
            log.inf(srv("llama_server") + f"listening on {self.listening_address(port)}")
            if self.is_fle and self.listening_address(port).endswith(":8080"):
                log.wrn(srv("llama_server") + "NOTICE: server default port will be changed to :9931 in a future "
                                              "release")
                log.wrn(srv("llama_server") + "        ref: https://github.com/ggml-org/llama.cpp/pull/26508")
            if self.early_save:
                log.inf(srv("operator()") + "[fle-slot-early-save] ON: slots are saved into the RAM cache when they "
                                            "become free")
            log.trc(srv("update_slots") + "all slots are idle")
        crash_after = env_float("FAKE_CRASH_AFTER", 0)
        while not self.stop.wait(0.1):
            if crash_after > 0 and time.monotonic() - self.t_ready > crash_after:
                self.finish(134)
        self.clean_up_line()
        httpd.shutdown()
        httpd.server_close()
        self.print_memory_breakdown()
        return 0


# ------------------------------------------------------------------------------------------------ one completion


class ChatOptions:
    """The request fields the fake looks at."""

    def __init__(self, req: dict[str, Any], cfg: Config) -> None:
        self.stream = bool(req.get("stream", False))
        stream_options = req.get("stream_options")
        self.include_usage = isinstance(stream_options, dict) and bool(stream_options.get("include_usage"))
        self.per_token = bool(req.get("timings_per_token"))
        self.progress = bool(req.get("return_progress"))
        self.ignore_eos = bool(req.get("ignore_eos", False))
        self.cache_prompt = req.get("cache_prompt", True) is not False
        kwargs = dict(cfg.chat_kwargs)
        if isinstance(req.get("chat_template_kwargs"), dict):
            kwargs.update(req["chat_template_kwargs"])
        self.think = kwargs.get("enable_thinking", True) is not False
        self.cap = cfg.n_predict
        for key in ("n_predict", "max_tokens", "max_completion_tokens"):
            value = req.get(key)
            if isinstance(value, int) and not isinstance(value, bool):
                self.cap = value
                break
        try:
            self.id_slot = int(req.get("id_slot", -1))
        except (TypeError, ValueError):
            self.id_slot = -1


class Completion:
    """One POST /v1/chat/completions, from slot selection to the last byte."""

    def __init__(self, fake: Fake, handler: Handler, req: dict[str, Any]) -> None:
        self.fake, self.h, self.req = fake, handler, req
        self.opts = ChatOptions(req, fake.cfg)
        self.prompt, self.last_user = render_prompt(req["messages"], self.opts.think)
        self.json_only = forced_format(req)  # under a forced format the real server never calls tools
        self.tool_calls = [] if self.json_only else requested_tool_calls(req, self.last_user)
        trailing: list[str] = []
        for msg in reversed(req["messages"]):
            if not isinstance(msg, dict) or msg.get("role") != "tool":
                break
            trailing.insert(0, flatten_content(msg.get("content")))
        self.tool_result = " ".join(trailing)
        self.n_prompt = max(1, len(self.prompt) // 4)
        digest = hashlib.sha256(self.prompt.encode("utf-8", "replace")).digest()
        self.seed = int.from_bytes(digest[:8], "big")
        self.chat_id = "chatcmpl-" + random_string()
        self.created = int(time.time())
        self.task_id = -1
        self.slot: Slot
        self.cache_n = 0
        self.prompt_n = self.n_prompt
        self.prompt_done = False
        self.t_prompt = 0.0
        self.tokens: list[tuple[str, str]] = []
        self.finish_reason = "stop"
        self.times: list[float] = []
        self.rounds: list[tuple[int, int, int]] = []
        self.sent = 0
        self.reasoning: list[str] = []
        self.content: list[str] = []
        self.finished = False

    def gone(self) -> bool:
        return client_gone(self.h.connection)

    def t_last(self, n: int) -> float:
        return self.times[n - 1] if n > 0 else self.t_prompt

    def cache_text(self) -> str:
        think = "".join(self.reasoning) + "\n</think>\n\n" if self.reasoning else ""
        return self.prompt + think + "".join(self.content)

    # ---- building blocks of the reply

    def chunk(self, delta: dict[str, Any] | None, reason: str | None = None, usage: dict[str, Any] | None = None,
              timings: dict[str, Any] | None = None, extra: dict[str, Any] | None = None) -> dict[str, Any]:
        fake = self.fake
        body: dict[str, Any] = {
            "choices": [] if delta is None else [{"finish_reason": reason, "index": 0, "delta": delta}],
            "created": self.created, "id": self.chat_id, "model": fake.model_name,
            "system_fingerprint": fake.build_info, "object": "chat.completion.chunk",
        }
        if usage is not None:
            body["usage"] = usage
        if timings is not None:
            body["timings"] = timings
        if extra:
            body.update(extra)
        return body

    def call_id(self, index: int) -> str:
        return f"call_{self.seed % 100000:05d}_{index}"

    def tool_delta(self, channel: str, piece: str) -> dict[str, Any]:
        """One streamed piece of a tool call: the first carries id and name, the rest the argument text."""
        kind, _, number = channel.partition(":")
        index = int(number)
        if kind == "toolname":
            return {"index": index, "id": self.call_id(index), "type": "function",
                    "function": {"name": piece, "arguments": ""}}
        return {"index": index, "function": {"arguments": piece}}

    def timings_at(self, n_gen: int) -> dict[str, Any]:
        """server_slot_stats::to_json() after n_gen generated tokens."""
        t_prompt_ms = self.t_prompt * 1000.0
        t_gen_ms = max(1e-3, (self.t_last(n_gen) - self.t_prompt) * 1000.0) if n_gen > 0 else 0.0
        steps = max(0, n_gen - 1)
        out: dict[str, Any] = {
            "cache_n": self.cache_n, "prompt_n": self.prompt_n, "prompt_ms": t_prompt_ms,
            "prompt_per_token_ms": t_prompt_ms / self.prompt_n if self.prompt_n else 0.0,
            "prompt_per_second": 1e3 / t_prompt_ms * self.prompt_n if t_prompt_ms > 0 else 0.0,
            "predicted_n": n_gen, "predicted_ms": t_gen_ms,
            "predicted_per_token_ms": t_gen_ms / steps if steps else 0.0,
            "predicted_per_second": 1e3 / t_gen_ms * steps if t_gen_ms > 0 else 0.0,
        }
        spec = self.spec_at(n_gen)
        if spec["drafted"] > 0:
            out["draft_n"], out["draft_n_accepted"] = spec["drafted"], spec["accepted"]
        return out

    def spec_at(self, n: int) -> dict[str, Any]:
        return spec_stats(self.rounds, n, self.fake.draft_width)

    def send_progress(self, processed: int, time_ms: int) -> None:
        progress = {"prompt_progress": {"total": self.n_prompt, "cache": self.cache_n, "processed": processed,
                                        "time_ms": time_ms}}
        self.h.sse(self.chunk({"role": "assistant", "content": None}, extra=progress))

    # ---- the request

    def run(self) -> None:
        fake, log, opts, h = self.fake, self.fake.log, self.opts, self.h
        self.task_id = fake.new_task_id()
        try:
            self.slot = fake.acquire(self.prompt, self.n_prompt, opts.id_slot, self.gone)
        except Cancelled:
            log.wrn(srv("stop") + f"cancel task, id_task = {self.task_id}")
            h.close_connection = True
            return
        slot = self.slot
        view = {"id": self.task_id, "n_tokens": 0, "n_processed": 0, "n_cached": 0, "n_decoded": 0,
                "n_remain": opts.cap, "has_next": True,
                "params": fake.params_json(self.req, opts.stream, opts.cap, opts.ignore_eos, opts.per_token)}
        slot.task = view
        log.inf(slt("launch_slot_", slot.id, self.task_id) + "processing task, is_child = 0")
        if self.n_prompt >= slot.n_ctx:
            msg = (f"request ({self.n_prompt} tokens) exceeds the available context size ({slot.n_ctx} tokens), "
                   "try increasing it")
            log.err(srv("send_error") + f"task id = {self.task_id}, error: {msg}")
            fake.release(slot, self.task_id, None)
            self.finished = True
            h.send_json(400, {"error": {"code": 400, "message": msg, "type": "exceed_context_size_error",
                                        "n_prompt_tokens": self.n_prompt, "n_ctx": slot.n_ctx}})
            return
        self.plan(view)
        try:
            self.generate(view)
        except (Cancelled, OSError):
            h.close_connection = True
            if self.finished:  # the slot is already released, only the socket failed
                return
            log.wrn(srv("stop") + f"cancel task, id_task = {self.task_id}")
            view["has_next"] = False
            fake.settle(self, self.spec_at(self.sent), None)

    def plan(self, view: dict[str, Any]) -> None:
        """Prompt cache, token list and virtual timeline."""
        fake, opts, slot = self.fake, self.opts, self.slot
        if opts.cache_prompt and slot.cache_text:
            self.cache_n = min(lcp(slot.cache_text, self.prompt) // 4, self.n_prompt - 1)
        else:
            slot.cache_text, slot.cache_tokens = "", 0
        self.prompt_n = self.n_prompt - self.cache_n
        view.update(n_tokens=self.n_prompt, n_processed=self.prompt_n, n_cached=self.cache_n)
        fake.log.trc(slt("update_slots", slot.id, self.task_id) + f"new prompt, n_ctx_slot = {slot.n_ctx}, n_keep = 0, "
                                                                 f"task.n_tokens = {self.n_prompt}")
        room = max(0, slot.n_ctx - self.n_prompt)
        limit = room if opts.cap < 0 else min(opts.cap, room)
        if self.json_only:
            snippet = re.sub(r"\s+", " ", self.last_user).strip().replace('"', "'")[:40]
            reply = ({"result": "no tools were called"} if self.req.get("tools") else {"reply": snippet})
            self.tokens, self.finish_reason = json_tokens(self.seed, opts.think, json.dumps(reply))
        elif self.tool_calls:
            self.tokens, self.finish_reason = tool_call_tokens(self.seed, opts.think, self.tool_calls)
        else:
            lead = ""
            if self.tool_result:
                said = re.sub(r"\s+", " ", self.tool_result).strip().replace('"', "'")[:80]
                lead = f'Tool said "{said}".'
            self.tokens, self.finish_reason = make_tokens(self.seed, self.last_user, opts.think, limit,
                                                          opts.ignore_eos, lead)
        rng = random.Random(self.seed ^ 0x9E3779B97F4A7C15)
        self.t_prompt = self.prompt_n / fake.prompt_rate * (1 + fake.jitter * (2 * rng.random() - 1))
        self.times, self.rounds = plan_timeline(len(self.tokens), self.t_prompt, fake.tok_rate, fake.jitter,
                                                fake.draft_width, rng)

    def generate(self, view: dict[str, Any]) -> None:
        fake, log, opts, h = self.fake, self.fake.log, self.opts, self.h
        tokens, times = self.tokens, self.times
        origin = time.perf_counter()
        if opts.stream:  # the real server sends the headers as soon as the prompt starts processing
            h.start_stream()
            if opts.progress:
                self.send_progress(self.cache_n, 0)
        n_batches = max(1, -(-self.prompt_n // N_BATCH))
        for k in range(1, n_batches + 1):
            done = min(self.prompt_n, k * N_BATCH)
            fraction = done / max(self.prompt_n, 1)
            fake.sleep_until(origin + self.t_prompt * fraction, self.gone)
            if opts.stream and opts.progress:
                self.send_progress(self.cache_n + done, int(self.t_prompt * fraction * 1000))
        fake.add_prompt_metrics(self.cache_n, self.prompt_n, self.t_prompt)
        self.prompt_done = True
        t_print_last, n_gen_last = self.t_prompt, 0
        n_total = len(tokens)
        for i, (piece, channel) in enumerate(tokens):
            if i > 0:
                fake.sleep_until(origin + times[i], self.gone)
            n = i + 1
            self.sent = n
            view.update(n_decoded=n, n_tokens=self.n_prompt + n - 1, n_remain=-1 if opts.cap < 0 else opts.cap - n)
            if i == 0 or times[i] != times[i - 1]:
                fake.count_moe_step()
            if channel == "reasoning":
                self.reasoning.append(piece)
            elif channel == "content":
                self.content.append(piece)
            if n >= 100 and times[i] - t_print_last >= 3.0:  # slot.print_timings_tg()
                rate = (n - 1) / (times[i] - self.t_prompt)
                window = (n - n_gen_last) / (times[i] - t_print_last)
                log.inf(slt("print_timing", self.slot.id, self.task_id)
                        + f"n_gen = {n:6d}, tg = {rate:6.2f} t/s, tg_3s = {window:6.2f} t/s")
                t_print_last, n_gen_last = times[i], n
            if not opts.stream:
                continue
            events = []
            if i == 0:
                events.append(self.chunk({"role": "assistant", "content": None}))
            if channel.startswith("tool"):
                events.append(self.chunk({"tool_calls": [self.tool_delta(channel, piece)]}))
            elif channel != "hidden":
                key = "reasoning_content" if channel == "reasoning" else "content"
                with_timings = opts.per_token or (n == n_total and self.finish_reason == "length")
                events.append(self.chunk({key: piece}, timings=self.timings_at(n) if with_timings else None))
            if events:
                h.sse(*events)
        view["has_next"] = False
        timing = self.timings_at(n_total)
        usage = {"completion_tokens": n_total, "prompt_tokens": self.n_prompt,
                 "total_tokens": n_total + self.n_prompt, "prompt_tokens_details": {"cached_tokens": self.cache_n}}
        # print_timings, metrics and release come first; the reply is only the last step, as seen from outside
        fake.settle(self, self.spec_at(n_total), timing)
        self.finished = True
        if opts.stream:
            tail = [self.chunk({}, self.finish_reason)]
            if opts.include_usage:
                tail.append(self.chunk(None, usage=usage))
            tail[-1]["timings"] = timing
            h.sse(*tail)
            h.sse_done()
        else:
            message: dict[str, Any] = {"role": "assistant", "content": "".join(self.content)}
            if self.reasoning:
                message["reasoning_content"] = "".join(self.reasoning)
            if self.tool_calls:
                message["tool_calls"] = [{"type": "function", "id": self.call_id(i),
                                          "function": {"name": name, "arguments": args}}
                                         for i, (name, args) in enumerate(self.tool_calls)]
            h.send_json(200, {
                "choices": [{"finish_reason": self.finish_reason, "index": 0, "message": message}],
                "created": self.created, "model": fake.model_name, "system_fingerprint": fake.build_info,
                "object": "chat.completion", "usage": usage, "id": self.chat_id, "timings": timing,
            })


# ------------------------------------------------------------------------------------------------ HTTP


class ThreadingServer(socketserver.ThreadingMixIn, HTTPServer):
    daemon_threads = True
    allow_reuse_address = not IS_WINDOWS  # on Windows SO_REUSEADDR would let a second server share the port
    request_queue_size = 128
    fake: Fake

    def __init__(self, address: tuple[str, int], handler: type, family: int = socket.AF_INET) -> None:
        self.address_family = family
        super().__init__(address, handler)

    def server_bind(self) -> None:
        if IS_WINDOWS and hasattr(socket, "SO_EXCLUSIVEADDRUSE"):
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
        socketserver.TCPServer.server_bind(self)  # HTTPServer.server_bind would do a slow getfqdn() lookup
        host, port = self.server_address[:2]
        self.server_name, self.server_port = str(host), port

    def handle_error(self, request: Any, client_address: Any) -> None:
        pass  # a vanished client is routine; the default handler would print a traceback


UI_PAGE = (b"<!doctype html><html><head><meta charset=\"utf-8\"><title>llama.cpp</title></head>"
           b"<body><h1>llama.cpp</h1><p>fake llama-server</p></body></html>")


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    disable_nagle_algorithm = True
    server: ThreadingServer

    def version_string(self) -> str:
        return "llama.cpp"

    def log_message(self, format: str, *args: Any) -> None:
        pass  # the real server does not log requests either

    # ---- response helpers

    def head(self, status: int, ctype: str, length: int | None, extra: tuple[tuple[str, str], ...] = ()) -> None:
        self.send_response_only(status)
        self.send_header("Server", "llama.cpp")
        self.send_header("Access-Control-Allow-Origin", self.headers.get("Origin", ""))
        for key, value in extra:
            self.send_header(key, value)
        self.send_header("Content-Type", ctype)
        if length is None:
            self.send_header("Transfer-Encoding", "chunked")
        else:
            self.send_header("Content-Length", str(length))
        self.end_headers()

    def send_body(self, status: int, body: bytes, ctype: str = "application/json; charset=utf-8",
                  extra: tuple[tuple[str, str], ...] = ()) -> None:
        self.head(status, ctype, len(body), extra)
        if self.command != "HEAD":
            self.wfile.write(body)

    def send_json(self, status: int, obj: Any) -> None:
        self.send_body(status, dumps(obj))

    def fail(self, message: str, kind: str, code: int) -> None:
        """format_error_response(): the key order here is code, message, type."""
        self.send_json(code, {"error": {"code": code, "message": message, "type": kind}})

    def middleware_error(self, status: int, message: str, kind: str) -> None:
        """The pre-routing middleware (503, 401, 404) uses the key order message, type, code."""
        self.send_json(status, {"error": {"message": message, "type": kind, "code": status}})

    def start_stream(self) -> None:
        self.head(200, "text/event-stream", None)

    def sse(self, *events: dict[str, Any]) -> None:
        data = b"".join(b"data: " + dumps(e) + b"\n\n" for e in events)
        self.wfile.write(f"{len(data):x}\r\n".encode("ascii") + data + b"\r\n")

    def sse_done(self) -> None:
        data = b"data: [DONE]\n\n"
        self.wfile.write(f"{len(data):x}\r\n".encode("ascii") + data + b"\r\n0\r\n\r\n")

    # ---- request flow

    def read_body(self) -> bytes:
        if "chunked" in self.headers.get("Transfer-Encoding", "").lower():
            data = b""
            while True:
                size = int(self.rfile.readline().split(b";")[0].strip() or b"0", 16)
                if size == 0:
                    while self.rfile.readline().strip():
                        pass
                    return data
                data += self.rfile.read(size)
                self.rfile.readline()
        length = int(self.headers.get("Content-Length") or 0)
        return self.rfile.read(length) if length > 0 else b""

    def do_GET(self) -> None:
        self.dispatch()

    def do_POST(self) -> None:
        self.dispatch()

    def do_OPTIONS(self) -> None:
        self.dispatch()

    def do_HEAD(self) -> None:
        self.dispatch()

    def do_PUT(self) -> None:
        self.dispatch()

    def do_DELETE(self) -> None:
        self.dispatch()

    def dispatch(self) -> None:
        fake = self.server.fake
        try:
            body = self.read_body()
        except (ValueError, OSError):
            self.close_connection = True
            return
        parts = urlsplit(self.path)
        path = parts.path
        if self.command == "OPTIONS":
            self.send_body(200, b"", "text/html", (
                ("Access-Control-Allow-Credentials", "true"),
                ("Access-Control-Allow-Methods", "GET, POST, DELETE, OPTIONS"),
                ("Access-Control-Allow-Headers", "*")))
            return
        ui_path = path in ("/", "/index.html") and fake.cfg.ui
        if not fake.ready.is_set() and not ui_path:
            self.middleware_error(503, "Loading model", "unavailable_error")
            return
        if fake.cfg.api_keys and path not in ("/health", "/v1/health") and not ui_path:
            key = self.headers.get("Authorization") or self.headers.get("X-Api-Key") or ""
            key = key[7:] if key.startswith("Bearer ") else key
            if key not in fake.cfg.api_keys:
                fake.log.wrn(srv("operator()") + "unauthorized: Invalid API Key")
                self.middleware_error(401, "Invalid API Key", "authentication_error")
                return
        try:
            self.route(fake, path, parse_qs(parts.query), body)
        except OSError:
            self.close_connection = True
        except Exception as exc:  # the real server also answers 500 for anything unexpected
            fake.log.err(srv("operator()") + f"got exception: {exc}")
            self.close_connection = True
            try:
                self.fail(str(exc), "server_error", 500)
            except OSError:
                pass

    def route(self, fake: Fake, path: str, query: dict[str, list[str]], body: bytes) -> None:
        cfg, method = fake.cfg, self.command
        if method in ("GET", "HEAD"):
            if path in ("/health", "/v1/health"):
                return self.send_json(200, {"status": "ok"})
            if path == "/props":
                return self.send_json(200, fake.props_json())
            if path in ("/v1/models", "/models"):
                return self.send_json(200, fake.models_json())
            if path == "/metrics":
                if not cfg.metrics:
                    return self.fail("This server does not support metrics endpoint. Start it with `--metrics`",
                                     "not_supported_error", 501)
                fake.new_task_id()
                with fake.cv:
                    text = fake.metrics.render(sum(1 for s in fake.slots if s.busy), len(fake.queue))
                    fake.metrics.reset_bucket()
                return self.send_body(200, text.encode("utf-8"), "text/plain; version=0.0.4",
                                      (("Process-Start-Time-Unix", str(fake.metrics.t_start)),))
            if path == "/slots":
                if not cfg.slots:
                    return self.fail("This server does not support slots endpoint. Start it with `--slots`",
                                     "not_supported_error", 501)
                fake.new_task_id()
                with fake.cv:
                    data = [fake.slot_json(s) for s in fake.slots]
                    n_idle = sum(1 for s in fake.slots if not s.busy)
                if query.get("fail_on_no_slot") and n_idle == 0:
                    return self.fail("no slot available", "unavailable_error", 503)
                return self.send_json(200, data)
            if path in ("/", "/index.html") and cfg.ui:
                return self.send_body(200, UI_PAGE, "text/html; charset=utf-8")
        elif method == "POST":
            if path in ("/v1/chat/completions", "/chat/completions"):
                return self.chat(fake, body)
            if path == "/props":
                if not cfg.props:
                    return self.fail("This server does not support changing global properties. Start it with "
                                     "`--props`", "not_supported_error", 501)
                return self.send_json(200, {"success": True})
        self.middleware_error(404, "File Not Found", "not_found_error")

    def chat(self, fake: Fake, body: bytes) -> None:
        try:
            req = json.loads(body.decode("utf-8")) if body else None
        except ValueError as exc:
            return self.fail(f"[json.exception.parse_error.101] parse error: {exc}", "invalid_request_error", 400)
        if not isinstance(req, dict):
            return self.fail("Expected a JSON object", "invalid_request_error", 400)
        if "messages" not in req:
            return self.fail("'messages' is required", "invalid_request_error", 400)
        if not isinstance(req["messages"], list):
            return self.fail("Expected 'messages' to be an array", "invalid_request_error", 400)
        if has_image(req["messages"]) and not fake.vision:
            return self.fail("image input is not supported - hint: if this is unexpected, you may need to provide "
                             "the mmproj", "invalid_request_error", 400)
        Completion(fake, self, req).run()


# ------------------------------------------------------------------------------------------------ entry point


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    for stream in (sys.stderr, sys.stdout):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    cfg = parse_args(argv, dict(os.environ))
    engine = ENGINES[cfg.engine]
    if cfg.version:
        compiler, target = build_strings()
        sys.stderr.write(f"version: {engine['version']} (build {engine['build']}, commit {engine['commit']})\n"
                         f"built with {compiler} for {target}\n")
        return 0
    if cfg.help:
        # The two option names FLE Studio looks for in the real help (engines.detect): -lv support and the FLE flavor.
        lines = ["usage: fake_llama_server.py -m MODEL.gguf [--host H] [--port N] [-c N] [-a ALIAS] "
                 "[--metrics] [-np N] [-md DRAFT.gguf] [-lv N]",
                 "-lv,   --verbosity, --log-verbosity N      Set the verbosity threshold (default: 3)"]
        if cfg.engine == "fle":
            lines.append("--moe-expert-cache-inserts N            expert-cache inserts per step")
        sys.stdout.write("\n".join(lines) + "\n")
        return 0
    rejected = {a for a in os.environ.get("FAKE_REJECT_ARGS", "").split(",") if a}
    for token in argv:
        if token.split("=")[0] in rejected:
            sys.stderr.write(f"error: invalid argument: {token}\n")
            return 1
    if not cfg.model:
        sys.stderr.write("error: no model given, use -m MODEL.gguf (the fake has no router mode)\n")
        return 1
    log = Log(cfg.verbosity, cfg.log_prefix, cfg.log_timestamps, cfg.log_disable)
    fake = Fake(cfg, log)
    signal.signal(signal.SIGINT, fake.on_signal)
    signal.signal(signal.SIGTERM, fake.on_signal)
    if hasattr(signal, "SIGBREAK"):  # Windows: terminate() is a hard kill, tests send CTRL_BREAK_EVENT instead
        signal.signal(signal.SIGBREAK, fake.on_signal)
    fake.finish(fake.run())


if __name__ == "__main__":
    sys.exit(main())

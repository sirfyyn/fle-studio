"""How an engine is started: its command line and environment.

Two ways to get one:

- **auto**: derived from this machine, the model and a few settings. What most people use.
- **a profile**: a saved, complete command with placeholders: the reference command from the model card, a server's
  production command, an experiment. A profile is pasted as shell text (environment assignments, binary, arguments)
  and runs exactly as written, plus the few arguments FLE Studio needs to talk to the engine (host, port, metrics).

Placeholders in profile arguments and environment values: {model} {mtp_head} {mtp_vocab} {expert_profile} {port}
{host} {threads} {context} {data}.
"""

from __future__ import annotations

import json
import re
import shlex
import socket
import subprocess
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from . import paths, qol
from .system import GiB, MiB

FIT_TARGET_MIB = 1115  # free GPU memory next to the expert cache, from the reference command
# GPU memory the engine needs besides the always-on weights, the KV cache, the expert cache and the fit margin: CUDA
# context, compute buffers and the MTP head. Measured with the reference command on the reference machine (FLE-IQ3_M,
# 262,144 tokens in two slots, -ub 4096 with ubatch lending): 32,623 MiB card - 4,460 always-on - 3,264 KV - 1,115
# margin - 17,035 cache = 6,749 MiB. Other settings are not measured; they get the same value as a cautious estimate.
ENGINE_OVERHEAD_MIB = 6749

# The command from the model card (sirfyyn/Qwen3.8-Flash-Next-FLE-GGUF, section "FLE engine"), as a profile.
REFERENCE_ENV = {
    "FLE_MOE_CACHE_BUDGET_MIB": "17035",
    "FLE_MOE_CACHE_BUDGET_FIT": "1",
    "FLE_MOE_PROFILE": "{expert_profile}",
    "FLE_MOE_PROFILE_SAVE_SEC": "600",
    "FLE_MOE_ADMIT_FREQ": "2000",
    "FLE_MOE_EVICT_OLDEST": "32",
    "FLE_MOE_CACHE_BIAS": "0.001",
    "FLE_MOE_OBS_MAXTOK": "8",
    "FLE_MC_OVERLAP": "2",
    "FLE_HC_FUSED": "1",
    "FLE_CKPT_TAIL": "512",
    "FLE_MTP_VOCAB": "{mtp_vocab}",
    "FLE_MTP_VOCAB_N": "32768",
    "FLE_MTP_PENALTIES": "1",
    "FLE_PREFILL_LEAN": "1",
    "FLE_SLOT_EARLY_SAVE": "1",
    "FLE_CACHE_GUARD_SLOT": "0",
    "FLE_UBATCH_LEND": "1",
    "FLE_UBATCH_PREFETCH": "1",
    "FLE_UBATCH_PREFETCH_SLOTS": "3",
}
REFERENCE_ARGS = [
    "-m",
    "{model}",
    "-md",
    "{mtp_head}",
    "--spec-type",
    "draft-mtp",
    "--spec-draft-n-max",
    "2",
    "--spec-draft-p-min",
    "0.5",
    "--no-spec-draft-backend-sampling",
    "-cmoed",
    "-c",
    "262144",
    "--parallel",
    "2",
    "--kv-unified",
    "--jinja",
    "-b",
    "4096",
    "-ub",
    "4096",
    "-ctk",
    "q8_0",
    "-ctv",
    "q8_0",
    "-fit",
    "on",
    "--fit-target",
    "1115",
    "--moe-expert-cache-inserts",
    "1",
    "--cache-ram",
    "32768",
    "--load-mode",
    "none",
    "-t",
    "16",
    "-C",
    "0x0000FFFF",
    "--cpu-strict",
    "1",
    "--temp",
    "0.6",
    "--top-p",
    "0.95",
    "--top-k",
    "20",
    "--min-p",
    "0",
    "--presence-penalty",
    "1.5",
]
# Cache policy and compute switches of the reference command that do not depend on the machine.
FLE_POLICY_ENV = {
    k: v for k, v in REFERENCE_ENV.items() if k not in ("FLE_MOE_CACHE_BUDGET_MIB", "FLE_MOE_PROFILE", "FLE_MTP_VOCAB")
}


_PLACEHOLDER = re.compile(r"\{[a-z_]+\}")
_SHELL_SAFE = re.compile(r"[\w@%+=:,./-]+")


def shell_quote(word: str) -> str:
    """shlex.quote that leaves placeholders such as {model} readable ({a,b} and the like still get quoted)."""
    return word if word and _SHELL_SAFE.fullmatch(_PLACEHOLDER.sub("x", word)) else shlex.quote(word)


class PlanError(ValueError):
    """The engine cannot be started like this; the message says why in words a user can act on."""


@dataclass
class Plan:
    engine: str  # fle | upstream | custom
    binary: str
    args: list[str]
    env: dict[str, str]
    port: int
    model: dict[str, Any]
    profile: str = "auto"
    notes: list[str] = field(default_factory=list)
    draft: str | None = None  # path of the draft/MTP model, for the load progress

    def command_line(self) -> str:
        """The command as one would type it in this OS's shell, for display and copying."""
        if paths.IS_WIN:
            lines = [f"set {k}={v}" for k, v in self.env.items()]
            lines.append(subprocess.list2cmdline([self.binary, *self.args]))
            return "\r\n".join(lines)
        env = " ".join(f"{k}={shell_quote(v)}" for k, v in self.env.items())
        cmd = " ".join(shell_quote(a) for a in [self.binary, *self.args])
        return f"{env} \\\n  {cmd}" if env else cmd

    def public(self) -> dict[str, Any]:
        d = asdict(self)
        d["command"] = self.command_line()
        return d


# ------------------------------------------------------------------------------------------------ helpers


def free_port(preferred: int = 0) -> int:
    if preferred:
        with socket.socket() as s:
            try:
                s.bind(("127.0.0.1", preferred))
                return preferred
            except OSError:
                pass
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def split_command(text: str) -> tuple[dict[str, str], str, list[str]]:
    """Parse pasted shell text into (environment, binary, arguments).

    Accepts backslash or caret line continuations, leading `env`, `KEY=VALUE` words before the binary, and a
    systemd-style `ExecStart=` prefix. The binary may be omitted (the text starts with an argument), then it is "".
    """
    text = re.sub(r"\\\r?\n|\^\r?\n|`\r?\n", " ", text.strip())
    text = re.sub(r"^ExecStart=[-@+!:]*", "", text)
    words = _split_words(text)
    env: dict[str, str] = {}
    while words and (words[0] == "env" or re.match(r"^[A-Za-z_][A-Za-z0-9_]*=", words[0])):
        word = words.pop(0)
        if word != "env":
            key, _, value = word.partition("=")
            env[key] = value
    binary = ""
    if words and not words[0].startswith("-"):
        binary = words.pop(0)
    return env, binary, words


def _split_words(text: str) -> list[str]:
    if not paths.IS_WIN:
        try:
            return shlex.split(text)
        except ValueError as exc:
            raise PlanError(f"Could not read the command: {exc}") from None
    # Windows: double quotes group, backslashes are path separators.
    words, cur, quoted, has = [], [], False, False
    for ch in text:
        if ch == '"':
            quoted, has = not quoted, True
        elif ch.isspace() and not quoted:
            if cur or has:
                words.append("".join(cur))
            cur, has = [], False
        else:
            cur.append(ch)
    if cur or has:
        words.append("".join(cur))
    return words


def parse_env_lines(text: str) -> dict[str, str]:
    env = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        key, sep, value = line.partition("=")
        if not sep or not re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", key.strip()):
            raise PlanError(f"Not a KEY=VALUE line: {line}")
        env[key.strip()] = value.strip()
    return env


def _has(args: list[str], *names: str) -> bool:
    return any(a in names or any(a.startswith(n + "=") for n in names if n.startswith("--")) for a in args)


def _arg_value(args: list[str], *names: str) -> str | None:
    for i, a in enumerate(args):
        if a in names and i + 1 < len(args):
            return args[i + 1]
        for n in names:
            if n.startswith("--") and a.startswith(n + "="):
                return a.split("=", 1)[1]
    return None


def _gpu_selection(setting: str, nvidia: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, str], str]:
    """Which NVIDIA GPUs the engine sees: (chosen GPUs, environment, note)."""
    if not nvidia:
        return [], {}, ""
    setting = (setting or "auto").strip().lower()
    if setting == "all" or len(nvidia) == 1 and setting == "auto":
        return nvidia, {}, ""
    if setting == "auto":
        pick = [max(nvidia, key=lambda g: g["memory_total"])]
    else:
        want = {s.strip() for s in setting.split(",") if s.strip()}
        pick = [g for g in nvidia if str(g["index"]) in want]
        if not pick:
            have = ", ".join(str(g["index"]) for g in nvidia)
            raise PlanError(f"No NVIDIA GPU with number {setting}; this machine has {have}")
    env = {"CUDA_DEVICE_ORDER": "PCI_BUS_ID", "CUDA_VISIBLE_DEVICES": ",".join(str(g["index"]) for g in pick)}
    names = ", ".join(f"GPU {g['index']} ({g['name']})" for g in pick)
    return pick, env, f"Engine uses {names}; the other GPUs stay free."


def _round_mib(n_bytes: float, step: int = 256) -> int:
    return max(0, int(n_bytes // MiB) // step * step)


# ------------------------------------------------------------------------------------------------ auto


def auto_plan(
    *,
    model: dict[str, Any],
    engine: dict[str, Any],
    settings: dict[str, Any],
    sysinfo: dict[str, Any],
    port: int,
    data_dir: Path,
) -> Plan:
    """The command FLE Studio picks for this model on this machine."""
    launch, chat = settings["launch"], settings["chat"]
    kind = engine["kind"]
    flavor = engine.get("flavor") or ("fle" if kind == "fle" else "upstream")  # what a custom binary turned out to be
    info = model.get("info") or {}
    notes: list[str] = []
    ctx = int(launch["context"]) or 32768
    if info.get("context_length") and ctx > info["context_length"]:
        ctx = int(info["context_length"])
        notes.append(f"Context capped at the model's maximum of {ctx:,} tokens.")
    parallel = max(1, int(launch.get("parallel") or 1))
    args = ["-m", model["path"], "-c", str(ctx), "--jinja"]
    if parallel > 1:
        args += ["--parallel", str(parallel), "--kv-unified"]
        notes.append(qol.effective_context(ctx, parallel, True)["note"])
    env: dict[str, str] = {}
    nvidia = [g for g in sysinfo.get("gpus", []) if g.get("vendor") == "nvidia"]
    chosen, gpu_env, note = _gpu_selection(launch.get("gpus", "auto"), nvidia)
    env.update(gpu_env)
    if note:
        notes.append(note)
    mem = sysinfo["memory"]
    physical = int(sysinfo["cpu"].get("physical") or 1)
    threads = int(launch.get("threads") or 0) or physical
    draft: str | None = None

    if flavor == "fle":
        if not chosen:
            raise PlanError("The FLE engine needs an NVIDIA GPU. Pick the upstream engine on the Engine page.")
        vram_total = sum(g["memory_total"] for g in chosen)
        vram_free = sum(max(0, g["memory_total"] - g.get("memory_used", 0)) for g in chosen)
        ub = 4096 if vram_total >= 20 * GiB else 2048
        kv = float(info.get("kv_bytes_per_token_q8") or 0) * ctx
        overhead = ENGINE_OVERHEAD_MIB * MiB
        gpu_weights = float(info.get("gpu_bytes") or 0)
        budget_mib = int(launch.get("gpu_cache_mib") or 0)
        if budget_mib <= 0:
            # The budget is added to the fit margin (FLE_MOE_CACHE_BUDGET_FIT=1): too large a budget makes the fit step
            # move weights to the CPU, so it errs on the small side.
            budget_mib = _round_mib(vram_free - gpu_weights - kv - overhead - FIT_TARGET_MIB * MiB, 1)
            notes.append(
                f"GPU expert cache {budget_mib / 1024:.1f} GiB: {vram_free / GiB:.1f} GiB free GPU memory minus "
                f"{gpu_weights / GiB:.1f} GiB of always-on weights, {kv / GiB:.1f} GiB KV cache, "
                f"{overhead / GiB:.1f} GiB for compute buffers and the CUDA context (measured with the reference "
                f"command) and {FIT_TARGET_MIB} MiB margin. Set it by hand on the Engine page if loading fails or "
                "memory is left over."
            )
        if budget_mib < 1024:
            notes.append("Little GPU memory is left for the expert cache, so most experts run on the CPU.")
        if not launch.get("threads") and physical > 1 and int(sysinfo["cpu"].get("logical") or physical) <= physical:
            # With a compute thread on every core, the engine's own threads (the one driving the GPU, the expert
            # cache's copy thread) wait for a free core. Measured with a small MoE model, 8 threads on 8 cores: decode
            # took 19-75 ms instead of 0.6-1.0 ms per token in 3 of 22 runs; with 6 threads none in 10 runs.
            threads = physical - 1
            notes.append(
                f"{threads} threads: this processor runs one thread per core, so one core stays free for "
                "the engine's GPU and expert-cache threads."
            )
        env.update(FLE_POLICY_ENV)
        env["FLE_MOE_CACHE_BUDGET_MIB"] = str(budget_mib)
        env["FLE_MOE_PROFILE"] = str(paths.ensure(data_dir / "expert-profiles") / f"{model['id']}.bin")
        args += [
            "-b",
            str(ub),
            "-ub",
            str(ub),
            "-ctk",
            "q8_0",
            "-ctv",
            "q8_0",
            "-fit",
            "on",
            "--fit-target",
            str(FIT_TARGET_MIB),
            "--moe-expert-cache-inserts",
            "1",
            "-t",
            str(threads),
        ]
        expert = float(info.get("expert_bytes") or 0)
        pinned = launch.get("pinned_load", "auto")
        if pinned == "auto":
            pinned = "on" if paths.IS_LINUX or paths.IS_WIN else "off"
            if mem["total"] - expert < 24 * GiB:
                pinned = "off"
                notes.append("Experts stay memory-mapped (not pinned): pinning them would leave less than 24 GiB RAM.")
        if pinned == "on":
            args += ["--load-mode", "none"]
        spec = str(launch.get("spec") or "auto").lower()
        if spec not in ("auto", "none", "mtp", "ngram-mod", "ngram-simple"):
            raise PlanError(
                f"Unknown speculation mode {spec!r}.",
                "Use auto, none, mtp, ngram-mod or ngram-simple.",
            )
        mtp = model.get("mtp") or {}
        head = mtp.get("head") or ""
        vocab = mtp.get("vocab") or ""
        use_mtp = False
        ngram = ""
        if spec == "none":
            notes.append("Speculation is off.")
        elif spec == "mtp":
            if not (head and vocab):
                raise PlanError(
                    "MTP was requested, but this model has no MTP head and vocabulary file.",
                    "Download the MTP sidecars on the Models page, or set speculation to auto or ngram-mod.",
                )
            use_mtp = True
        elif spec in ("ngram-mod", "ngram-simple"):
            ngram = spec
            notes.append(
                f"Self-speculation: {spec}. It drafts from the prompt/context; acceptance is prompt-dependent."
            )
        else:  # auto
            if launch.get("mtp", True) and head and vocab:
                use_mtp = True
            elif launch.get("mtp", True) and mtp.get("supported") and not head:
                notes.append("MTP is off: the MTP head or its vocabulary file is not downloaded (Models page).")
        if use_mtp:
            draft = head
            args += [
                "-md",
                head,
                "--spec-type",
                "draft-mtp",
                "--spec-draft-n-max",
                "2",
                "--spec-draft-p-min",
                "0.5",
                "--no-spec-draft-backend-sampling",
                "-cmoed",
            ]
            env.update({"FLE_MTP_VOCAB": vocab, "FLE_MTP_VOCAB_N": "32768", "FLE_MTP_PENALTIES": "1"})
        if ngram:
            args += ["--spec-type", ngram]
        pinned_bytes = expert if pinned == "on" else 0.0
    else:
        args += ["-fit", "on"]
        if launch.get("threads"):
            args += ["-t", str(threads)]
        pinned_bytes = 0.0
        if engine.get("backend") == "cpu" and info.get("moe"):
            notes.append("CPU engine: every expert runs on the CPU.")

    cram = int(launch.get("prompt_cache_mib", -1))
    if cram < 0:
        headroom = mem["total"] - pinned_bytes - 16 * GiB
        cram = min(32768, _round_mib(headroom / 4, 1024))
    args += ["--cache-ram", str(cram)]
    if launch.get("cpu_mask"):
        args += ["-C", launch["cpu_mask"], "--cpu-strict", "1"]
    sampling = model.get("sampling") or {}
    if chat.get("sampling") == "custom":
        sampling = {k: chat[k] for k in ("temperature", "top_p", "top_k", "min_p", "presence_penalty")}
    flag = {
        "temperature": "--temp",
        "top_p": "--top-p",
        "top_k": "--top-k",
        "min_p": "--min-p",
        "presence_penalty": "--presence-penalty",
    }
    for key, value in sampling.items():
        if key in flag:
            args += [flag[key], f"{value:g}" if isinstance(value, float) else str(value)]
    args += _words(launch.get("extra_args", ""))
    env.update(parse_env_lines(launch.get("extra_env", "")))
    return _finish(
        Plan(
            engine=kind,
            binary=engine["path"],
            args=args,
            env=env,
            port=port,
            model=model,
            profile="auto",
            notes=notes,
            draft=draft,
        ),
        bool(engine.get("log_verbosity", True)),
    )


def _words(text: str) -> list[str]:
    return _split_words(text) if text and text.strip() else []


def _finish(plan: Plan, log_verbosity: bool) -> Plan:
    """Add what FLE Studio needs to reach and follow the engine, unless the command sets it itself.

    log_verbosity: the engine knows -lv. llama.cpp logs its library lines (loading phases, the FLE expert-cache hit
    rate) at level 4 only, one above the default; without them the live panel shows neither."""
    a = plan.args
    if (
        log_verbosity
        and "LLAMA_ARG_LOG_VERBOSITY" not in plan.env
        and not _has(a, "-v", "--verbose", "--log-verbose", "-lv", "--verbosity", "--log-verbosity")
    ):
        a += ["-lv", "4"]
    if not _has(a, "--host"):
        a += ["--host", "127.0.0.1"]
    if _has(a, "--port"):
        try:
            plan.port = int(_arg_value(a, "--port") or plan.port)
        except ValueError:
            raise PlanError("--port needs a number") from None
    else:
        a += ["--port", str(plan.port)]
    if not _has(a, "--metrics"):
        a += ["--metrics"]
    if not _has(a, "-a", "--alias"):
        a += ["-a", str(plan.model.get("name") or "model")]
    return plan


# ------------------------------------------------------------------------------------------------ profiles


@dataclass
class Profile:
    id: str
    name: str
    engine: str = "fle"  # fle | upstream | custom (custom: `binary` is the engine)
    binary: str = ""
    args: list[str] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict)
    notes: str = ""
    builtin: bool = False
    # What a profile that FLE Studio made knows about itself, e.g. {"tuned": {...}} (see tuner.py); empty for the
    # profiles people write. Kept when the profile is saved again from the editor.
    meta: dict[str, Any] = field(default_factory=dict)

    def command_text(self) -> str:
        env = " ".join(f"{k}={shell_quote(v)}" for k, v in self.env.items())
        cmd = " ".join(shell_quote(a) for a in [self.binary or "{engine}", *self.args])
        return f"{env} \\\n{cmd}" if env else cmd

    def public(self) -> dict[str, Any]:
        d = asdict(self)
        d["command"] = self.command_text()
        return d


BUILTIN = (
    Profile(
        id="fle-reference",
        name="FLE reference (32 GB GPU)",
        engine="fle",
        args=list(REFERENCE_ARGS),
        env=dict(REFERENCE_ENV),
        notes="The exact command from the model card: 262,144-token context shared by two slots, 16 threads pinned to "
        "cores 0-15, a 17 GiB GPU expert cache for a 32 GB card. Needs the MTP head and about 54 GiB of free RAM "
        "for FLE-IQ3_M. Measured only on an RTX PRO 4500 32 GB with 120 GB RAM.",
        builtin=True,
    ),
)


class Profiles:
    def __init__(self, data_dir: Path) -> None:
        self.file = data_dir / "profiles.json"
        self._lock = threading.Lock()
        self.items: list[Profile] = []
        try:
            for raw in json.loads(self.file.read_text(encoding="utf-8")):
                raw.pop("builtin", None)
                raw.pop("command", None)
                self.items.append(Profile(**raw))
        except (OSError, ValueError, TypeError):
            pass

    def all(self) -> list[Profile]:
        with self._lock:
            return [*BUILTIN, *self.items]

    def get(self, ident: str) -> Profile | None:
        return next((p for p in self.all() if p.id == ident), None)

    def save(self, data: dict[str, Any]) -> Profile:
        """Create or replace a profile from {id?, name, engine, command | binary+args+env, notes}."""
        name = str(data.get("name") or "").strip()
        if not name:
            raise PlanError("A profile needs a name")
        if "command" in data:
            env, binary, args = split_command(str(data["command"]))
            if binary in ("{engine}", "llama-server", "llama-server.exe"):
                binary = ""
        else:
            env, binary, args = dict(data.get("env") or {}), str(data.get("binary") or ""), list(data.get("args") or [])
        engine = str(data.get("engine") or ("custom" if binary else "fle"))
        if engine not in ("fle", "upstream", "custom"):
            raise PlanError("engine must be fle, upstream or custom")
        if engine == "custom" and not binary:
            raise PlanError("A custom-engine profile needs the path of its llama-server at the start of the command")
        ident = str(data.get("id") or "").strip()
        if not ident or any(b.id == ident for b in BUILTIN):
            ident = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")[:40] or "profile"
            ident = f"{ident}-{int(time.time()) % 100000}"
        old = self.get(ident)
        meta = data["meta"] if isinstance(data.get("meta"), dict) else (old.meta if old else {})
        prof = Profile(
            id=ident,
            name=name,
            engine=engine,
            binary=binary,
            args=args,
            env=env,
            notes=str(data.get("notes") or ""),
            meta=dict(meta),
        )
        with self._lock:
            self.items = [p for p in self.items if p.id != ident] + [prof]
            self._write()
        return prof

    def delete(self, ident: str) -> None:
        with self._lock:
            self.items = [p for p in self.items if p.id != ident]
            self._write()

    def _write(self) -> None:
        paths.ensure(self.file.parent)
        tmp = self.file.with_suffix(".tmp")
        tmp.write_text(json.dumps([asdict(p) for p in self.items], indent=2), encoding="utf-8")
        tmp.replace(self.file)


def profile_plan(
    *,
    profile: Profile,
    model: dict[str, Any] | None,
    engine: dict[str, Any] | None,
    settings: dict[str, Any],
    sysinfo: dict[str, Any],
    port: int,
    data_dir: Path,
) -> Plan:
    """A profile's command with its placeholders filled in."""
    model = model or {}
    mtp = model.get("mtp") or {}
    values = {
        "model": model.get("path", ""),
        "mtp_head": mtp.get("head") or "",
        "mtp_vocab": mtp.get("vocab") or "",
        "expert_profile": str(paths.ensure(data_dir / "expert-profiles") / f"{model.get('id', profile.id)}.bin"),
        "port": str(port),
        "host": "127.0.0.1",
        "threads": str(int(settings["launch"].get("threads") or 0) or sysinfo["cpu"].get("physical") or 1),
        "context": str(settings["launch"]["context"]),
        "data": str(data_dir),
    }
    needed = {"model": "a model", "mtp_head": "the MTP head (Models page)", "mtp_vocab": "the MTP vocabulary file"}

    def fill(text: str) -> str:
        def sub(m: re.Match[str]) -> str:
            key = m.group(1)
            if key not in values:
                return m.group(0)
            if not values[key] and key in needed:
                raise PlanError(f"Profile “{profile.name}” needs {needed[key]}")
            return values[key]

        return re.sub(r"\{(\w+)\}", sub, text)

    args = [fill(a) for a in profile.args]
    env = {k: fill(v) for k, v in profile.env.items()}
    if not _has(args, "-m", "--model") and model.get("path"):
        args = ["-m", model["path"], *args]
    if not _has(args, "-m", "--model"):
        raise PlanError("Choose a model first")
    notes = [profile.notes] if profile.notes else []
    # The GPU and core choice on the Engine page also steers a profile, unless its command already decides.
    gpus = str(settings["launch"].get("gpus") or "auto")
    if gpus != "auto" and "CUDA_VISIBLE_DEVICES" not in env:
        _, gpu_env, note = _gpu_selection(gpus, [g for g in sysinfo.get("gpus", []) if g.get("vendor") == "nvidia"])
        env = {**gpu_env, **env}
        if note:
            notes.append(note)
    if settings["launch"].get("cpu_mask") and not _has(args, "-C", "--cpu-mask"):
        args += ["-C", settings["launch"]["cpu_mask"], "--cpu-strict", "1"]
    if profile.engine == "custom":
        binary, kind = fill(profile.binary), "custom"
    else:
        if not engine:
            which = "FLE" if profile.engine == "fle" else "upstream"
            raise PlanError(f"Install the {which} engine first (Engine page)")
        binary, kind = engine["path"], engine["kind"]
    draft = _arg_value(args, "-md", "--model-draft")
    plan = Plan(
        engine=kind,
        binary=binary,
        args=args,
        env=env,
        port=port,
        model=model,
        profile=profile.id,
        notes=notes,
        draft=draft,
    )
    # For a custom profile `engine` is the check of its own binary (None if that could not run): no -lv unless known.
    return _finish(plan, bool((engine or {}).get("log_verbosity", profile.engine != "custom")))

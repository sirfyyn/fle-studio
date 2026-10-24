"""Finding the llama.cpp command-line tools Forge drives, and asking each what it can do.

Forge does not carry its own quantiser. It uses the tools of the engine build: `llama-imatrix` (importance matrix),
`llama-quantize`, `llama-perplexity` (KL divergence) and, in the FLE builds, `llama-fle-reap` (router statistics for
expert pruning). They are searched in this order: the folder set in the settings, the folders of the engines that
Studio installed, the folder of a llama-server of your own, then PATH. Each hit is asked for its version and for the
options Forge relies on, once per file version.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .. import paths

TOOL_NAMES = ("llama-imatrix", "llama-quantize", "llama-perplexity", "llama-fle-reap")
REQUIRED = {
    "hybrid": (),
    "quantize": ("llama-quantize",),
    "imatrix": ("llama-imatrix",),
    "verify": ("llama-perplexity",),
}

# option text to look for in `--help` -> feature name
FEATURES: dict[str, dict[str, str]] = {
    "llama-quantize": {
        "--tensor-type": "tensor_type",
        "--imatrix": "imatrix",
        "--allow-requantize": "allow_requantize",
        "--output-tensor-type": "output_type",
        "--token-embedding-type": "embedding_type",
    },
    "llama-imatrix": {"--chunks": "chunks", "--in-file": "in_file", "-ngl": "gpu_layers"},
    "llama-perplexity": {
        "--kl-divergence": "kl_divergence",
        "--kl-divergence-base": "kl_base",
        "--chunks": "chunks",
        "-ngl": "gpu_layers",
    },
    "llama-fle-reap": {"--chunks": "chunks", "-cmoe": "cmoe", "-fit": "fit"},
}


@dataclass
class Tool:
    name: str
    path: str
    source: str  # where it was found: settings | engine | custom engine | PATH
    version: str = ""
    features: dict[str, bool] = field(default_factory=dict)
    problem: str = ""

    @property
    def ok(self) -> bool:
        return not self.problem

    def public(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "path": self.path,
            "source": self.source,
            "version": self.version,
            "features": self.features,
            "problem": self.problem,
            "ok": self.ok,
        }


def command_for(path: str) -> list[str]:
    """A script (the test stand-ins are scripts) is run by this Python; a real tool is run directly."""
    return [sys.executable, path] if path.lower().endswith(".py") else [path]


def environment(path: str) -> dict[str, str]:
    env = os.environ.copy()
    here = str(Path(path).parent)
    if paths.IS_LINUX:  # the libraries of an engine build sit next to its tools
        env["LD_LIBRARY_PATH"] = os.pathsep.join(x for x in (here, env.get("LD_LIBRARY_PATH")) if x)
    env.setdefault("PYTHONUTF8", "1")
    return env


def _run(path: str, *args: str) -> str:
    flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    try:
        done = subprocess.run(
            [*command_for(path), *args],
            capture_output=True,
            text=True,
            timeout=60,
            errors="replace",
            env=environment(path),
            cwd=str(Path(path).parent),
            creationflags=flags,
            stdin=subprocess.DEVNULL,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise RuntimeError(str(exc)) from None
    return (done.stdout or "") + "\n" + (done.stderr or "")


_VERSION = re.compile(r"version:\s*(\S+)(?:\s*\(([0-9a-f]+)\))?", re.I)
_cache: dict[tuple[str, int, int], Tool] = {}
_lock = threading.Lock()


def probe(path: str, name: str, source: str) -> Tool:
    p = Path(path)
    st = p.stat()
    key = (str(p), st.st_size, int(st.st_mtime))
    with _lock:
        hit = _cache.get(key)
    if hit:
        return Tool(name, hit.path, source, hit.version, dict(hit.features), hit.problem)
    tool = Tool(name, str(p), source)
    try:
        ver = _run(str(p), "--version")
        m = _VERSION.search(ver)
        tool.version = (m.group(1) + (f" ({m.group(2)})" if m.group(2) else "")) if m else ""
        hlp = _run(str(p), "--help")
        tool.features = {feat: opt in hlp for opt, feat in FEATURES.get(name, {}).items()}
        if not m and not hlp.strip():
            tool.problem = "does not answer --version or --help"
    except RuntimeError as exc:
        tool.problem = str(exc)[:300]
    with _lock:
        _cache[key] = tool
    return tool


def search_dirs(configured: str, engine_paths: list[tuple[str, str]]) -> list[tuple[str, Path]]:
    """(label, folder) in the order they are searched. `engine_paths` are (label, path of a llama-server)."""
    out: list[tuple[str, Path]] = []
    if configured.strip():
        out.append(("settings", Path(configured.strip()).expanduser()))
    for label, server in engine_paths:
        if server:
            out.append((label, Path(server).expanduser().parent))
    return out


def find(dirs: list[tuple[str, Path]], use_path: bool = True) -> dict[str, Tool]:
    """Every tool that can be found, probed. Missing tools are simply absent from the result."""
    found: dict[str, Tool] = {}
    for name in TOOL_NAMES:
        candidates: list[tuple[str, str]] = []
        for label, folder in dirs:
            for fname in (paths.exe(name), f"{name}.py"):
                p = folder / fname
                if p.is_file():
                    candidates.append((label, str(p)))
                    break
        if use_path:
            on_path = shutil.which(name)
            if on_path:
                candidates.append(("PATH", on_path))
        for label, cand in candidates:
            tool = probe(cand, name, label)
            if tool.ok:
                found[name] = tool
                break
            found.setdefault(name, tool)  # shown with its problem unless a later candidate works
    return found


def missing_for(tools: dict[str, Tool], needs: tuple[str, ...]) -> list[str]:
    return [n for n in needs if n not in tools or not tools[n].ok]

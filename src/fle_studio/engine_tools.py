"""The other programs that come with llama.cpp (llama-perplexity, llama-bench) and running them.

They sit next to llama-server in every release archive. FLE Studio finds them from the path of the engine in use and
runs them as plain subprocesses with the same library path the engine gets, so a build that needs its own libraries
works the same way. ``run_tool`` is blocking and cancellable, and hands partial output to a callback as it arrives (the
progress of llama-perplexity is printed without line breaks).
"""

from __future__ import annotations

import os
import subprocess
import sys
import threading
import time
from collections.abc import Callable
from pathlib import Path

from . import paths, procutil


def find_tool(engine_path: str, name: str) -> Path | None:
    """``name`` (e.g. llama-perplexity) from the folder of the engine, or None if the build does not have it.

    A development engine written in Python (the test stand-in ``fake_llama_server.py``) keeps its tools as
    ``fake_<name with underscores>.py`` next to it."""
    server = Path(engine_path)
    folder = server.parent
    if server.suffix == ".py":
        candidate = folder / f"fake_{name.replace('-', '_')}.py"
        return candidate if candidate.is_file() else None
    candidate = folder / paths.exe(name)
    return candidate if candidate.is_file() else None


def tool_command(binary: Path, args: list[str]) -> list[str]:
    return [sys.executable, str(binary), *args] if binary.suffix == ".py" else [str(binary), *args]


def tool_env(binary: Path, extra: dict[str, str] | None = None) -> dict[str, str]:
    env = os.environ.copy()
    if paths.IS_LINUX:
        env["LD_LIBRARY_PATH"] = os.pathsep.join(p for p in (str(binary.parent), env.get("LD_LIBRARY_PATH")) if p)
    elif paths.IS_WIN:
        env["PATH"] = str(binary.parent) + os.pathsep + env.get("PATH", "")
    env.update(extra or {})
    return env


class ToolError(RuntimeError):
    """A tool failed or was not found; the message says why in words a user can act on."""


def run_tool(
    binary: Path,
    args: list[str],
    *,
    env: dict[str, str] | None = None,
    cancel: threading.Event | None = None,
    on_text: Callable[[str], None] | None = None,
    timeout: float | None = None,
) -> tuple[int, str]:
    """Run a tool to the end and return (exit code, all output with stderr merged in).

    ``on_text`` gets the output in pieces as it comes. When ``cancel`` is set, or ``timeout`` seconds have passed,
    the tool is stopped and InterruptedError / TimeoutError is raised."""
    proc = subprocess.Popen(
        tool_command(binary, args),
        cwd=str(binary.parent),
        env=tool_env(binary, env),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        **procutil.popen_kwargs(),
    )
    chunks: list[str] = []
    lock = threading.Lock()

    def read() -> None:
        assert proc.stdout is not None
        while True:
            data = proc.stdout.read1(4096) if hasattr(proc.stdout, "read1") else proc.stdout.read(4096)
            if not data:
                break
            text = data.decode("utf-8", "replace")
            with lock:
                chunks.append(text)
            if on_text:
                on_text(text)

    reader = threading.Thread(target=read, name="tool-output", daemon=True)
    reader.start()
    started = time.monotonic()
    try:
        while proc.poll() is None:
            if cancel is not None and cancel.is_set():
                raise InterruptedError
            if timeout is not None and time.monotonic() - started > timeout:
                raise TimeoutError(f"{binary.name} did not finish within {timeout:.0f} s")
            time.sleep(0.1)
    except BaseException:
        procutil.ask_to_stop(proc)
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=10)
        raise
    finally:
        reader.join(timeout=5)
    with lock:
        return int(proc.returncode), "".join(chunks)

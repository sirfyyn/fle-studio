"""Shared helpers for the tests that run the stand-in engines (fake_llama_server.py and friends) as real processes."""

from __future__ import annotations

import json
import sys
import time
import urllib.request
from collections.abc import Callable
from pathlib import Path
from typing import Any

from fle_studio.engine import Engine
from fle_studio.launch import Plan, free_port

HERE = Path(__file__).resolve().parent
FAKE_SERVER = HERE / "fake_llama_server.py"
FAKE_PERPLEXITY = HERE / "fake_llama_perplexity.py"
FAKE_BENCH = HERE / "fake_llama_bench.py"


def stub_gguf(path: Path, arch: str = "qwen35moe", name: str = "stub") -> Path:
    """A GGUF file with a few metadata keys and nothing else: enough for the stand-in and for FLE Studio's reader."""
    import struct

    def text(value: str) -> bytes:
        raw = value.encode()
        return struct.pack("<Q", len(raw)) + raw

    def key_string(key: str, value: str) -> bytes:
        return text(key) + struct.pack("<I", 8) + text(value)

    def key_u32(key: str, value: int) -> bytes:
        return text(key) + struct.pack("<II", 4, value)

    items = [
        key_string("general.architecture", arch),
        key_string("general.name", name),
        key_u32(f"{arch}.context_length", 262144),
        key_u32(f"{arch}.block_count", 6),
        key_u32(f"{arch}.expert_count", 64),
        key_u32(f"{arch}.expert_used_count", 4),
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"GGUF" + struct.pack("<IQQ", 3, 0, len(items)) + b"".join(items))
    return path


def fake_plan(model: Path, *extra: str, env: dict[str, str] | None = None, engine: str = "custom") -> Plan:
    """A plan that starts the stand-in llama-server on a free port."""
    port = free_port()
    args = ["-m", str(model), "--host", "127.0.0.1", "--port", str(port), "--metrics", "-lv", "4", *extra]
    return Plan(
        engine=engine,
        binary=str(FAKE_SERVER),
        args=args,
        env={"FAKE_LOAD_SECONDS": "0.1", **(env or {})},
        port=port,
        model={"id": "m", "name": "stub", "path": str(model)},
    )


def wait_for(probe: Callable[[], Any], timeout: float = 30, what: str = "the condition") -> Any:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        value = probe()
        if value:
            return value
        time.sleep(0.05)
    raise AssertionError(f"timed out waiting for {what}")


def start_engine(engine: Engine, plan: Plan, timeout: float = 30) -> str:
    engine.start(plan)
    wait_for(lambda: engine.state in ("ready", "crashed", "stopped"), timeout, "the engine to start")
    return engine.state


def stop_engine(engine: Engine, timeout: float = 20) -> None:
    engine.stop()
    wait_for(lambda: engine.state in ("stopped", "crashed"), timeout, "the engine to stop")


def chat(base: str, text: str, max_tokens: int = 8) -> dict[str, Any]:
    body = {"messages": [{"role": "user", "content": text}], "max_tokens": max_tokens, "stream": False}
    req = urllib.request.Request(
        base + "/v1/chat/completions", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read().decode())


PYTHON = sys.executable

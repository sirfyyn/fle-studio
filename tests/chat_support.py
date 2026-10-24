"""Shared set-up for the chat tests: a real Studio server in front of the fake llama-server, and a small HTTP client."""

from __future__ import annotations

import http.client
import json
import os
import socket
import subprocess
import sys
import threading
import time
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from fle_studio.server import Studio, make_server

HERE = Path(__file__).parent
FAKE_LLAMA = HERE / "fake_llama_server.py"
FAKE_MCP = HERE / "fake_mcp_server.py"


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def clean_env(extra: dict[str, str] | None = None) -> dict[str, str]:
    env = {k: v for k, v in os.environ.items() if not k.startswith(("FAKE_", "FLE_", "LLAMA_ARG_"))}
    env.update(extra or {})
    return env


class FakeLlama:
    """The fake llama-server as a child process."""

    def __init__(self, tmp_path: Path, env: dict[str, str] | None = None) -> None:
        model = tmp_path / "Fake-Model-Q4_K_M.gguf"
        model.write_bytes(b"GGUF" + bytes(60))
        self.port = free_port()
        flags: dict[str, Any] = {}
        if os.name == "nt":
            flags["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
        self.proc = subprocess.Popen(
            [
                sys.executable,
                str(FAKE_LLAMA),
                "-m",
                str(model),
                "--host",
                "127.0.0.1",
                "--port",
                str(self.port),
                "-c",
                "8192",
                "--metrics",
            ],
            env=clean_env(
                {
                    "FAKE_LOAD_SECONDS": "0.2",
                    "FAKE_TOK_PER_SEC": "800",
                    "FAKE_PROMPT_TOK_PER_SEC": "50000",
                    **(env or {}),
                }
            ),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            **flags,
        )
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            try:
                conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=2)
                conn.request("GET", "/health")
                if conn.getresponse().status == 200:
                    conn.close()
                    return
            except OSError:
                pass
            time.sleep(0.05)
        self.close()
        raise AssertionError("the fake llama-server did not become ready")

    def close(self) -> None:
        if self.proc.poll() is None:
            self.proc.kill()
        self.proc.wait(timeout=10)


@dataclass
class Stack:
    studio: Studio
    port: int
    fake: FakeLlama | None

    def call(
        self,
        method: str,
        path: str,
        body: Any = None,
        raw: bytes | None = None,
        headers: dict[str, str] | None = None,
        token: bool = True,
    ) -> tuple[int, dict[str, str], bytes]:
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=60)
        try:
            send = dict(headers or {})
            if token:
                send["X-FLE-Token"] = self.studio.token
            payload = raw
            if payload is None and body is not None:
                payload = json.dumps(body).encode("utf-8")
                send.setdefault("Content-Type", "application/json")
            conn.request(method, path, body=payload, headers=send)
            resp = conn.getresponse()
            return resp.status, {k.lower(): v for k, v in resp.getheaders()}, resp.read()
        finally:
            conn.close()

    def json(self, method: str, path: str, body: Any = None, **kw: Any) -> tuple[int, Any]:
        status, _, data = self.call(method, path, body, **kw)
        return status, (json.loads(data) if data else None)

    def settings(self, patch: dict[str, Any]) -> dict[str, Any]:
        status, data = self.json("POST", "/api/settings", patch)
        assert status == 200, data
        return data  # type: ignore[no-any-return]

    def chat(self, body: dict[str, Any], stream: bool = False) -> tuple[int, Any]:
        """One chat completion through the Studio proxy."""
        status, _, data = self.call("POST", "/v1/chat/completions", {**body, "stream": stream})
        if not stream:
            return status, json.loads(data)
        events: list[dict[str, Any]] = []
        for line in data.decode("utf-8").splitlines():
            if line.startswith("data: ") and line != "data: [DONE]":
                events.append(json.loads(line[6:]))
        return status, events


def make_stack(tmp_path: Path, with_engine: bool = True, fake_env: dict[str, str] | None = None) -> Iterator[Stack]:
    fake = FakeLlama(tmp_path, fake_env) if with_engine else None
    studio = Studio(tmp_path / "home")
    server = make_server(studio, "127.0.0.1", 0)
    threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.1}, daemon=True).start()
    try:
        if fake is not None:
            studio.engine.attach(f"http://127.0.0.1:{fake.port}")
            assert studio.engine.props, "the fake llama-server gave no /props"
        yield Stack(studio, int(server.server_address[1]), fake)
    finally:
        studio.engine.shutdown()
        with_toolbox = studio.__dict__.get("toolbox")
        if with_toolbox is not None:
            with_toolbox.close()
        server.shutdown()
        server.server_close()
        if fake is not None:
            fake.close()


def mcp_config(*extra_args: str, name: str = "fake") -> str:
    """The `mcpServers` JSON for the stdio test server."""
    return json.dumps({"mcpServers": {name: {"command": sys.executable, "args": [str(FAKE_MCP), *extra_args]}}})

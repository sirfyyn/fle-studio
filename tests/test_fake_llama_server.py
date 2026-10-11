"""Tests for the fake llama-server in tests/fake_llama_server.py.

They start the fake as a real process, the way the launcher does, and talk to it over HTTP, so they also prove that
the double behaves well enough to stand in for the real engine in the launcher's own tests.
"""

from __future__ import annotations

import contextlib
import http.client
import importlib.util
import json
import os
import re
import signal
import socket
import struct
import subprocess
import sys
import threading
import time
from collections.abc import Generator, Iterator
from pathlib import Path
from typing import Any

import pytest

FAKE = Path(__file__).with_name("fake_llama_server.py")
STUB_NAME = "Fake-Model-IQ3_M-00001-of-00004.gguf"
LINE_RE = re.compile(r"^\d+\.\d\d\.\d\d\d\.\d\d\d [IWED] \S")
TIMING_KEYS = [
    "cache_n",
    "prompt_n",
    "prompt_ms",
    "prompt_per_token_ms",
    "prompt_per_second",
    "predicted_n",
    "predicted_ms",
    "predicted_per_token_ms",
    "predicted_per_second",
]
CHAT = {"messages": [{"role": "user", "content": "Say hello to the test suite."}]}
TOK_PER_SEC = 400


def load_fake_module() -> Any:
    spec = importlib.util.spec_from_file_location("fake_llama_server_under_test", FAKE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def clean_env(extra: dict[str, str] | None = None) -> dict[str, str]:
    """The current environment without anything that would change the fake's behaviour."""
    env = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith(("FAKE_", "FLE_", "LLAMA_ARG_")) and key != "LLAMA_API_KEY"
    }
    env.update(extra or {})
    return env


def make_stub(directory: Path, name: str = STUB_NAME) -> Path:
    path = directory / name
    path.write_bytes(b"GGUF" + bytes(60))
    return path


class Server:
    """A running fake llama-server process and everything it printed."""

    def __init__(self, proc: subprocess.Popen[str], port: int) -> None:
        self.proc = proc
        self.port = port
        self.first_health: tuple[int, Any] | None = None
        self._lines: list[str] = []
        self._lock = threading.Lock()
        self._reader = threading.Thread(target=self._read, daemon=True)
        self._reader.start()

    def _read(self) -> None:
        assert self.proc.stdout is not None
        for line in self.proc.stdout:
            with self._lock:
                self._lines.append(line.rstrip("\r\n"))

    def lines(self) -> list[str]:
        with self._lock:
            return list(self._lines)

    @property
    def log(self) -> str:
        return "\n".join(self.lines())

    def wait_for_log(self, needle: str, timeout: float = 15.0) -> bool:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if needle in self.log:
                return True
            time.sleep(0.02)
        return needle in self.log

    def request(
        self, method: str, path: str, body: Any = None, raw: bytes | None = None, timeout: float = 15.0
    ) -> tuple[int, dict[str, str], bytes]:
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=timeout)
        try:
            payload = raw if raw is not None else (json.dumps(body).encode() if body is not None else None)
            headers = {"Content-Type": "application/json"} if payload is not None else {}
            conn.request(method, path, body=payload, headers=headers)
            resp = conn.getresponse()
            return resp.status, {k.lower(): v for k, v in resp.getheaders()}, resp.read()
        finally:
            conn.close()

    def get_json(self, path: str) -> tuple[int, Any]:
        status, _, data = self.request("GET", path)
        return status, json.loads(data)

    def post_json(self, path: str, body: Any) -> tuple[int, Any]:
        status, _, data = self.request("POST", path, body)
        return status, json.loads(data)

    def stream(self, body: dict[str, Any]) -> tuple[list[dict[str, Any]], bool]:
        """POST a streaming chat request; returns the decoded events and whether [DONE] came last."""
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=30)
        try:
            conn.request(
                "POST",
                "/v1/chat/completions",
                body=json.dumps({**body, "stream": True}),
                headers={"Content-Type": "application/json"},
            )
            resp = conn.getresponse()
            assert resp.status == 200
            assert resp.getheader("Content-Type") == "text/event-stream"
            events: list[dict[str, Any]] = []
            while True:
                line = resp.readline()
                if not line:
                    return events, False
                line = line.strip()
                if not line.startswith(b"data: "):
                    continue
                if line == b"data: [DONE]":
                    return events, True
                events.append(json.loads(line[6:]))
        finally:
            conn.close()

    def wait_ready(self, timeout: float = 30.0) -> None:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self.proc.poll() is not None:
                raise AssertionError(f"the fake exited early with {self.proc.returncode}:\n{self.log}")
            try:
                status, _, data = self.request("GET", "/health", timeout=3.0)
            except (OSError, http.client.HTTPException):
                time.sleep(0.02)
                continue
            if self.first_health is None:
                self.first_health = (status, json.loads(data))
            if status == 200:
                return
            time.sleep(0.02)
        raise AssertionError(f"the fake did not become ready:\n{self.log}")

    def wait_exit(self, timeout: float = 20.0) -> int:
        code = self.proc.wait(timeout=timeout)
        self._reader.join(timeout=5)
        return code

    def terminate(self, timeout: float = 15.0) -> int:
        """What a launcher does to stop an engine: SIGTERM, or CTRL_BREAK on Windows."""
        if os.name != "nt":
            self.proc.terminate()
            return self.wait_exit(timeout)
        # A console control event needs a console shared with the child. A CI job without one cannot deliver it,
        # and that says nothing about the fake, so the test is skipped instead of failed there.
        try:
            self.proc.send_signal(signal.CTRL_BREAK_EVENT)
            return self.wait_exit(timeout)
        except (OSError, subprocess.TimeoutExpired) as exc:
            self.proc.kill()
            pytest.skip(f"CTRL_BREAK_EVENT did not reach the child process here: {exc!r}")

    def close(self) -> None:
        if self.proc.poll() is None:
            self.proc.kill()
        self.proc.wait(timeout=10)
        self._reader.join(timeout=5)
        if self.proc.stdout is not None:
            self.proc.stdout.close()


def start(
    tmp_path: Path,
    *args: str,
    env: dict[str, str] | None = None,
    wait: bool = True,
    model: str | None = None,
    port: int | None = None,
) -> Server:
    port = port if port is not None else free_port()
    model_path = model if model is not None else str(make_stub(tmp_path))
    cmd = [sys.executable, str(FAKE), "-m", model_path, "--host", "127.0.0.1", "--port", str(port), *args]
    environment = clean_env({"FAKE_LOAD_SECONDS": "0.3", "FAKE_TOK_PER_SEC": str(TOK_PER_SEC), **(env or {})})
    flags: dict[str, Any] = {}
    if os.name == "nt":
        flags["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env=environment,
        text=True,
        encoding="utf-8",
        errors="replace",
        **flags,
    )
    srv = Server(proc, port)
    if wait:
        try:
            srv.wait_ready()
        except BaseException:
            srv.close()
            raise
    return srv


@contextlib.contextmanager
def running(tmp_path: Path, *args: str, **kwargs: Any) -> Generator[Server, None, None]:
    srv = start(tmp_path, *args, **kwargs)
    try:
        yield srv
    finally:
        srv.close()


def scrape(srv: Server) -> dict[str, float]:
    status, _, data = srv.request("GET", "/metrics")
    assert status == 200
    values = {}
    for line in data.decode().splitlines():
        if line and not line.startswith("#"):
            name, _, value = line.rpartition(" ")
            values[name] = float(value)
    return values


@pytest.fixture(scope="module")
def server(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Server]:
    """One server with the usual launcher flags (several of them unknown to the fake) for most tests."""
    srv = start(
        tmp_path_factory.mktemp("fake"),
        "-c",
        "4096",
        "-a",
        "fake-model",
        "--metrics",
        "--jinja",
        "-ngl",
        "99",
        "--flash-attn",
        "on",
        "--cache-type-k",
        "q8_0",
        "--no-mmap",
        env={"FAKE_LOAD_SECONDS": "0.8"},
    )
    yield srv
    srv.close()


# ------------------------------------------------------------------------------------------------ startup and routes


def test_health_is_503_while_loading_then_200(server: Server) -> None:
    assert server.first_health is not None
    status, body = server.first_health
    assert status == 503
    assert body == {"error": {"message": "Loading model", "type": "unavailable_error", "code": 503}}
    assert server.get_json("/health") == (200, {"status": "ok"})
    log = server.log
    assert log.index("model loaded") < log.index(f"listening on http://127.0.0.1:{server.port}")


def test_default_log_has_the_real_shape_and_hides_library_lines(server: Server) -> None:
    log = server.log
    lines = [line for line in log.splitlines() if line.strip()]
    assert lines
    assert all(LINE_RE.match(line) for line in lines), lines
    assert " I srv  llama_server: initializing ..." in log
    assert " I cmn  common_param: common_params_print_info: verbosity = 3 (adjust with the `-lv N` CLI arg)" in log
    assert " W srv  llama_server: security: no API key is set and CORS allows all origins" in log
    assert " I srv    load_model: initializing, n_slots = 1, n_ctx_slot = 4096, kv_unified = 'false'" in log
    # these only exist at -lv 4 in the real tool, and so in the fake
    for hidden in ("llama_kv_cache", "print_info: file", "load_tensors", "MoE expert cache", "server is listening"):
        assert hidden not in log


def test_props_has_the_real_keys(server: Server) -> None:
    status, props = server.get_json("/props")
    assert status == 200
    assert list(props)[:6] == [
        "default_generation_settings",
        "total_slots",
        "model_alias",
        "model_ftype",
        "model_path",
        "modalities",
    ]
    assert props["default_generation_settings"]["n_ctx"] == 4096
    assert props["default_generation_settings"]["params"]["temperature"] == 0.8
    assert props["total_slots"] == 1
    assert props["model_alias"] == "fake-model"
    assert props["model_path"].endswith(STUB_NAME)
    assert props["model_ftype"] == "IQ3_S mix - 3.66 bpw"
    assert props["build_info"] == "b11540-1e6f04a75"
    assert props["modalities"] == {"vision": False, "video": False, "audio": False}
    assert props["endpoint_metrics"] is True and props["chat_template"] and props["eos_token"] == "<|im_end|>"


def test_models_and_slots(server: Server) -> None:
    status, models = server.get_json("/v1/models")
    assert status == 200 and models["object"] == "list"
    entry = models["data"][0]
    assert entry["id"] == "fake-model" and entry["aliases"] == ["fake-model"] and entry["owned_by"] == "llamacpp"
    assert entry["meta"]["n_ctx"] == 4096 and entry["meta"]["ftype"] == "IQ3_S mix - 3.66 bpw"
    status, slots = server.get_json("/slots")
    assert status == 200 and len(slots) == 1
    assert slots[0]["id"] == 0 and slots[0]["n_ctx"] == 4096 and slots[0]["is_processing"] is False


def test_metrics_endpoint_names_and_header(server: Server) -> None:
    status, headers, data = server.request("GET", "/metrics")
    assert status == 200
    assert headers["content-type"] == "text/plain; version=0.0.4"
    assert "process-start-time-unix" in headers
    text = data.decode()
    for name in (
        "prompt_tokens_total",
        "tokens_predicted_total",
        "tokens_predicted_seconds_total",
        "n_decode_total",
        "prompt_tokens_seconds",
        "predicted_tokens_seconds",
        "requests_processing",
        "requests_deferred",
    ):
        assert f"# TYPE llamacpp:{name} " in text
        assert re.search(rf"^llamacpp:{name} \S+$", text, re.MULTILINE)


def test_errors_use_the_real_bodies(server: Server) -> None:
    status, _, data = server.request("POST", "/v1/chat/completions", {"model": "x"})
    assert status == 400
    assert json.loads(data) == {
        "error": {"code": 400, "message": "'messages' is required", "type": "invalid_request_error"}
    }
    status, _, data = server.request("POST", "/v1/chat/completions", raw=b"{not json")
    assert status == 400 and json.loads(data)["error"]["type"] == "invalid_request_error"
    status, _, data = server.request("GET", "/nope")
    assert status == 404
    assert json.loads(data) == {"error": {"message": "File Not Found", "type": "not_found_error", "code": 404}}
    status, _, data = server.request(
        "POST", "/v1/chat/completions", {"messages": [{"role": "user", "content": "x" * 20000}]}
    )
    error = json.loads(data)["error"]
    assert status == 400 and error["type"] == "exceed_context_size_error" and error["n_ctx"] == 4096
    assert error["n_prompt_tokens"] > 4096 and "exceeds the available context size" in error["message"]
    assert server.get_json("/slots")[1][0]["is_processing"] is False


# ------------------------------------------------------------------------------------------------ chat


def test_streaming_chat_has_reasoning_content_usage_and_timings(server: Server) -> None:
    events, done = server.stream({**CHAT, "stream_options": {"include_usage": True}})
    assert done
    assert events[0]["choices"][0]["delta"] == {"role": "assistant", "content": None}
    for event in events:
        assert event["object"] == "chat.completion.chunk" and event["model"] == "fake-model"
        assert event["system_fingerprint"] == "b11540-1e6f04a75" and event["id"].startswith("chatcmpl-")
    kinds = []
    for event in events:
        if event["choices"]:
            delta = event["choices"][0]["delta"]
            if "reasoning_content" in delta:
                kinds.append("r")
            elif delta.get("content"):
                kinds.append("c")
    first_content = kinds.index("c")
    assert first_content >= 30 and kinds[:first_content] == ["r"] * first_content
    assert "r" not in kinds[first_content:]
    finishing = [e for e in events if e["choices"] and e["choices"][0]["finish_reason"]]
    assert len(finishing) == 1 and finishing[0]["choices"][0]["finish_reason"] == "stop"
    assert finishing[0]["choices"][0]["delta"] == {}
    last = events[-1]
    assert last["choices"] == []
    assert last["usage"]["completion_tokens"] > 0
    assert last["usage"]["total_tokens"] == last["usage"]["completion_tokens"] + last["usage"]["prompt_tokens"]
    assert list(last["timings"]) == TIMING_KEYS
    assert last["timings"]["predicted_n"] == last["usage"]["completion_tokens"]
    assert 0.75 * TOK_PER_SEC < last["timings"]["predicted_per_second"] < 1.25 * TOK_PER_SEC


def test_streaming_text_is_deterministic(server: Server) -> None:
    def answer() -> str:
        events, _ = server.stream(CHAT)
        return "".join(e["choices"][0]["delta"].get("content") or "" for e in events if e["choices"])

    first = answer()
    assert first.startswith('Fake reply to "Say hello to the test suite."')
    assert answer() == first


def test_non_streaming_chat_with_and_without_thinking(server: Server) -> None:
    status, body = server.post_json("/v1/chat/completions", CHAT)
    assert status == 200 and body["object"] == "chat.completion" and body["model"] == "fake-model"
    message = body["choices"][0]["message"]
    assert body["choices"][0]["finish_reason"] == "stop" and message["role"] == "assistant"
    assert message["reasoning_content"] and message["content"].startswith("Fake reply to")
    assert list(body["usage"]) == ["completion_tokens", "prompt_tokens", "total_tokens", "prompt_tokens_details"]
    assert list(body["timings"]) == TIMING_KEYS
    status, body = server.post_json(
        "/v1/chat/completions", {**CHAT, "chat_template_kwargs": {"enable_thinking": False}}
    )
    assert status == 200 and "reasoning_content" not in body["choices"][0]["message"]
    assert body["choices"][0]["message"]["content"].startswith("Fake reply to")


def test_max_tokens_and_ignore_eos(server: Server) -> None:
    _, body = server.post_json("/v1/chat/completions", {**CHAT, "max_tokens": 5})
    assert body["choices"][0]["finish_reason"] == "length" and body["usage"]["completion_tokens"] == 5
    assert body["timings"]["predicted_n"] == 5
    _, body = server.post_json("/v1/chat/completions", {**CHAT, "n_predict": 150, "ignore_eos": True})
    assert body["choices"][0]["finish_reason"] == "length" and body["usage"]["completion_tokens"] == 150


def test_prompt_cache_reuses_the_common_prefix(server: Server) -> None:
    first = [{"role": "user", "content": "A long enough first message so the cache has something to reuse."}]
    server.post_json("/v1/chat/completions", {"messages": first, "max_tokens": 4})
    follow = [*first, {"role": "assistant", "content": "ok"}, {"role": "user", "content": "And a second question."}]
    _, body = server.post_json("/v1/chat/completions", {"messages": follow, "max_tokens": 4})
    timings = body["timings"]
    assert timings["cache_n"] > 0 and timings["prompt_n"] > 0
    assert body["usage"]["prompt_tokens_details"]["cached_tokens"] == timings["cache_n"]
    assert body["usage"]["prompt_tokens"] == timings["cache_n"] + timings["prompt_n"]
    _, body = server.post_json("/v1/chat/completions", {"messages": follow, "max_tokens": 4, "cache_prompt": False})
    assert body["timings"]["cache_n"] == 0


def test_metrics_counters_follow_requests_and_log_has_print_timings(server: Server) -> None:
    before = scrape(server)
    mark = len(server.lines())
    _, body = server.post_json("/v1/chat/completions", {**CHAT, "max_tokens": 30})
    after = scrape(server)
    generated = body["usage"]["completion_tokens"]
    assert after["llamacpp:tokens_predicted_total"] - before["llamacpp:tokens_predicted_total"] == generated
    assert after["llamacpp:prompt_tokens_total"] > before["llamacpp:prompt_tokens_total"]
    assert after["llamacpp:tokens_predicted_seconds_total"] > before["llamacpp:tokens_predicted_seconds_total"]
    assert after["llamacpp:n_decode_total"] > before["llamacpp:n_decode_total"]
    assert 0.75 * TOK_PER_SEC < after["llamacpp:predicted_tokens_seconds"] < 1.25 * TOK_PER_SEC
    new = "\n".join(server.lines()[mark:])
    for expected in (
        "processing task, is_child = 0",
        "prompt eval time =",
        "       eval time =",
        "      total time =",
        "   graphs reused =",
        "stop processing: n_tokens = ",
    ):
        assert expected in new, new


def test_client_disconnect_stops_generation_and_frees_the_slot(server: Server) -> None:
    mark = len(server.lines())
    conn = http.client.HTTPConnection("127.0.0.1", server.port, timeout=10)
    conn.request(
        "POST",
        "/v1/chat/completions",
        body=json.dumps({**CHAT, "stream": True, "ignore_eos": True, "max_tokens": 100000}),
        headers={"Content-Type": "application/json"},
    )
    resp = conn.getresponse()
    assert resp.status == 200
    for _ in range(6):
        resp.readline()
    conn.close()
    deadline = time.monotonic() + 10
    while server.get_json("/slots")[1][0]["is_processing"]:
        assert time.monotonic() < deadline, "the slot stayed busy after the client left"
        time.sleep(0.05)
    new = "\n".join(server.lines()[mark:])
    assert "stop: cancel task, id_task = " in new and "stop processing:" in new
    assert "prompt eval time" not in new  # a cancelled task prints no timings


def test_one_slot_serves_one_request_at_a_time(server: Server) -> None:
    mark = len(server.lines())
    results: list[tuple[int, Any]] = []

    def worker(text: str) -> None:
        body = {
            "messages": [{"role": "user", "content": text}],
            "max_tokens": 40,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        results.append(server.post_json("/v1/chat/completions", body))

    threads = [threading.Thread(target=worker, args=(f"request number {i}",)) for i in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=30)
    assert [status for status, _ in results] == [200, 200]
    events = re.findall(r"slot\s+(launch_slot_|release): id\s+\d+ \| task (\d+)", "\n".join(server.lines()[mark:]))
    assert [name for name, _ in events] == ["launch_slot_", "release", "launch_slot_", "release"]
    assert events[0][1] == events[1][1] and events[2][1] == events[3][1] and events[0][1] != events[2][1]


# ------------------------------------------------------------------------------------------------ other processes


def test_metrics_and_slots_can_be_switched_off(tmp_path: Path) -> None:
    with running(tmp_path, "--no-slots") as srv:
        status, _, data = srv.request("GET", "/metrics")
        assert status == 501
        assert json.loads(data) == {
            "error": {
                "code": 501,
                "type": "not_supported_error",
                "message": "This server does not support metrics endpoint. Start it with `--metrics`",
            }
        }
        status, _, data = srv.request("GET", "/slots")
        assert status == 501 and "Start it with `--slots`" in json.loads(data)["error"]["message"]
        props = srv.get_json("/props")[1]
        assert props["endpoint_metrics"] is False and props["endpoint_slots"] is False


def test_fle_mode_speculative_decoding_and_verbose_log(tmp_path: Path) -> None:
    draft = make_stub(tmp_path, "draft-mtp-IQ4_NL.gguf")
    env = {
        "FLE_MOE_CACHE_BUDGET_MIB": "4096",
        "FLE_MOE_CACHE_BUDGET_FIT": "1",
        "FLE_SLOT_EARLY_SAVE": "1",
        "FAKE_TOK_PER_SEC": "4000",
    }
    args = (
        "-lv",
        "4",
        "-md",
        str(draft),
        "--spec-type",
        "draft-mtp",
        "--spec-draft-n-max",
        "2",
        "--metrics",
        "--cache-ram",
        "1024",
        "-fit",
        "on",
        "-a",
        "fle",
    )
    with running(tmp_path, *args, env=env) as srv:
        log = srv.log
        assert "common_init_result: MoE cache budget 4096 MiB added to the fit margin of device 0 (now 5120 MiB)" in log
        assert "common_speculative_init_result: loading draft model" in log
        assert "MoE expert cache enabled: 40 layers x " in log and "[fle-moe-budget] EVIDENCE allocated" in log
        assert "llama_kv_cache: size = " in log and "[fle-slot-early-save] ON" in log
        assert "NOTICE" not in log  # the default-port notice only appears on :8080
        props = srv.get_json("/props")[1]
        assert props["build_info"] == "b10825-1efd83707"
        assert props["default_generation_settings"]["params"]["speculative.types"] == "draft-mtp"
        body = {**CHAT, "max_tokens": 1500, "ignore_eos": True, "chat_template_kwargs": {"enable_thinking": False}}
        _, answer = srv.post_json("/v1/chat/completions", body)
        timings = answer["timings"]
        assert 0 < timings["draft_n_accepted"] <= timings["draft_n"]
        assert srv.wait_for_log("draft acceptance = ", 5) and srv.wait_for_log("moe-cache: steps=512 hits=", 5)
        assert srv.wait_for_log("[fle-slot-early-save] saved:", 5)
        metrics = scrape(srv)
        assert metrics["llamacpp:spec_decode_num_draft_tokens_total"] == timings["draft_n"]
        assert metrics["llamacpp:spec_decode_num_accepted_tokens_total"] == timings["draft_n_accepted"]
        assert 'llamacpp:spec_decode_num_accepted_tokens_per_pos_total{position="0"}' in metrics


def test_missing_model_fails_like_the_real_server(tmp_path: Path) -> None:
    with running(tmp_path, wait=False, model=str(tmp_path / "missing.gguf")) as srv:
        assert srv.wait_exit() == 1
        log = srv.log
    assert " E gguf_init_from_file: failed to open GGUF file " in log
    assert " E llama_model_load_from_file_impl: failed to load model" in log
    assert " E srv    load_model: failed to load model, " in log
    assert " E srv  llama_server: exiting due to model loading error" in log


def test_out_of_memory_fails_with_a_cuda_message(tmp_path: Path) -> None:
    with running(tmp_path, wait=False, env={"FAKE_FAIL": "oom", "FAKE_LOAD_SECONDS": "0.2"}) as srv:
        assert srv.wait_exit() == 1
        log = srv.log
    assert "cudaMalloc failed: out of memory" in log and "unable to allocate CUDA0 buffer" in log
    assert "exiting due to model loading error" in log


def test_port_in_use_fails_to_bind(tmp_path: Path) -> None:
    with socket.socket() as blocker:
        blocker.bind(("127.0.0.1", 0))
        blocker.listen(1)
        with running(tmp_path, wait=False, port=blocker.getsockname()[1]) as srv:
            assert srv.wait_exit() == 1
            log = srv.log
    assert "couldn't bind HTTP server socket" in log and "exiting due to HTTP server error" in log


def test_version_output_has_the_real_shape() -> None:
    expected = {
        "upstream": "version: 0.6.0-dev (build 11540, commit 1e6f04a75)",
        "fle": "version: 0.4.0-dev (build 10825, commit 1efd83707)",
    }
    for engine, first_line in expected.items():
        result = subprocess.run(
            [sys.executable, str(FAKE), "--version"],
            capture_output=True,
            text=True,
            env=clean_env({"FAKE_ENGINE": engine}),
            timeout=30,
            check=False,
        )
        assert result.returncode == 0
        lines = result.stderr.splitlines()
        assert lines[0] == first_line and lines[1].startswith("built with ")


def test_help_names_what_fle_studio_detects() -> None:
    # engines.detect: "--log-verbosity" means the engine takes -lv, "--moe-expert-cache-inserts" means FLE.
    for engine in ("upstream", "fle"):
        result = subprocess.run(
            [sys.executable, str(FAKE), "--help"],
            capture_output=True,
            text=True,
            env=clean_env({"FAKE_ENGINE": engine}),
            timeout=30,
            check=False,
        )
        assert result.returncode == 0
        assert "--log-verbosity" in result.stdout
        assert ("--moe-expert-cache-inserts" in result.stdout) == (engine == "fle")


def test_terminate_after_ready_exits_zero_with_the_real_cleanup_line(tmp_path: Path) -> None:
    with running(tmp_path, env={"FAKE_LOAD_SECONDS": "0.2"}) as srv:
        started = time.monotonic()
        assert srv.terminate() == 0
        assert time.monotonic() - started < 5
        assert " I srv    operator(): operator(): cleaning up before exit..." in srv.log


def test_terminate_during_the_load_exits_promptly(tmp_path: Path) -> None:
    with running(tmp_path, wait=False, env={"FAKE_LOAD_SECONDS": "60"}) as srv:
        assert srv.wait_for_log("loading model")
        started = time.monotonic()
        assert srv.terminate() == 0
        assert time.monotonic() - started < 5


# ------------------------------------------------------------------------------------------------ units


def test_parse_args_accepts_and_ignores_unknown_flags() -> None:
    fake = load_fake_module()
    argv = [
        "-m",
        "a.gguf",
        "--port=1234",
        "-c",
        "8192",
        "-a",
        "x,y",
        "--metrics",
        "--flash-attn",
        "on",
        "-ngl",
        "99",
        "--no-mmap",
        "--cache-type-k",
        "q8_0",
        "--seed",
        "-1",
        "--spec-draft-n-max",
        "2",
        "-np",
        "2",
        "-v",
    ]
    cfg = fake.parse_args(argv, {})
    assert (cfg.model, cfg.port, cfg.ctx, cfg.alias) == ("a.gguf", 1234, 8192, "x,y")
    assert cfg.metrics is True and cfg.parallel == 2
    assert cfg.draft_max == 2 and cfg.verbosity == 2**31 - 1
    assert cfg.unknown == ["--flash-attn", "-ngl", "--no-mmap", "--cache-type-k", "--seed"]
    assert cfg.argv == argv


def test_parse_args_environment_and_engine_defaults() -> None:
    fake = load_fake_module()
    assert fake.parse_args(["-m", "a"], {}).port == 9931 and fake.parse_args(["-m", "a"], {}).engine == "upstream"
    fle = fake.parse_args(["-m", "a"], {"FLE_MOE_CACHE_BUDGET_MIB": "100"})
    assert fle.engine == "fle" and fle.port == 8080
    assert fake.parse_args(["-m", "a"], {"LLAMA_ARG_PORT": "4321", "LLAMA_ARG_ENDPOINT_METRICS": "1"}).port == 4321
    assert fake.parse_args(["-m", "a", "--port", "55"], {"LLAMA_ARG_PORT": "4321"}).port == 55
    assert fake.parse_args(["-m", "a"], {"LLAMA_ARG_ENDPOINT_METRICS": "1"}).metrics is True


def test_log_prefix_and_verbosity_filter(capsys: pytest.CaptureFixture[str]) -> None:
    fake = load_fake_module()
    assert fake.slt("release", 0, 7) == "slot      release: id  0 | task 7 | "
    assert fake.srv("operator()") == "srv    operator(): "
    assert fake.cmn("common_params_print_info") == "cmn  common_param: "
    log = fake.Log(3)
    log.inf("srv  x: shown")
    log.trc("llama_kv_cache: hidden")
    err = capsys.readouterr().err
    assert re.fullmatch(r"\d+\.\d\d\.\d\d\d\.\d\d\d I srv  x: shown\n", err)
    verbose = fake.Log(4, prefix=False)
    verbose.trc("llama_kv_cache: shown")
    assert capsys.readouterr().err == "llama_kv_cache: shown\n"


def test_gguf_reader_skips_arrays_and_reads_what_follows(tmp_path: Path) -> None:
    fake = load_fake_module()

    def key(name: str) -> bytes:
        return struct.pack("<Q", len(name)) + name.encode()

    def text(value: str) -> bytes:
        return struct.pack("<I", 8) + struct.pack("<Q", len(value)) + value.encode()

    strings = b"".join(struct.pack("<Q", len(s)) + s.encode() for s in ("a", "bb", "ccc"))
    body = (
        key("general.architecture")
        + text("qwen35moe")
        + key("tokenizer.ggml.tokens")
        + struct.pack("<I", 9)
        + struct.pack("<IQ", 8, 3)
        + strings
        + key("qwen35moe.block_count")
        + struct.pack("<I", 4)
        + struct.pack("<I", 48)
        + key("general.name")
        + text("Fake Model")
    )
    path = tmp_path / "meta.gguf"
    path.write_bytes(b"GGUF" + struct.pack("<IQQ", 3, 0, 4) + body)
    assert fake.read_gguf_meta(str(path)) == {
        "general.architecture": "qwen35moe",
        "qwen35moe.block_count": 48,
        "general.name": "Fake Model",
    }
    assert fake.read_gguf_meta(str(tmp_path / "missing.gguf")) == {}
    assert fake.ModelInfo(str(path)).n_layer == 48

"""Browser end-to-end test of FLE Studio.

It starts a real FLE Studio (`python -m fle_studio`, the installed package) in a temporary data folder, points it at
`tests/fake_llama_server.py` as "my own build" of llama-server, adds a tiny GGUF file as the model and starts the engine
with the automatic plan. Then a headless Chromium (Playwright) does what a person would do:

* open the app, send one chat message and wait for the streamed answer, with its reasoning block and speed line;
* check that the live panel shows a decode speed;
* visit Models, Engine, Experiments (one run of the speed test), Forge (two synthetic quants, a calculated hybrid,
  pruning and a check built by stand-in llama.cpp tools, then published) and Settings (the theme switch), and save
  a full-page screenshot of every page in light and in dark mode;
* fail on any uncaught page error and any console error.

At the end it stops the engine and Studio and checks that no engine process is left behind. One line is printed per
step. The exit code is 0 only when every step passed. No GPU, no model and no internet are needed.

    python -m pip install . playwright
    python -m playwright install chromium
    python tests/e2e/run_e2e.py --out e2e-out [--channel chrome] [--headed]
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import re
import shutil
import signal
import socket
import ssl
import struct
import subprocess
import sys
import tempfile
import time
import traceback
import urllib.error
import urllib.request
from collections.abc import Callable, Generator
from pathlib import Path
from typing import TYPE_CHECKING, Any
from urllib.parse import urlparse

if TYPE_CHECKING:
    from playwright.sync_api import Page

FAKE_ENGINE = Path(__file__).resolve().parents[1] / "fake_llama_server.py"
FAKE_MCP = Path(__file__).resolve().parents[1] / "fake_mcp_server.py"
sys.path.insert(0, str(FAKE_ENGINE.parent))
from fake_github import FakeGitHub, release  # noqa: E402 - the stand-in for api.github.com (update check)

# What the fake release says. The scripts and the image must stay inert text in the page.
UPDATE_TAG = "v9.0.0"
UPDATE_NOTES = (
    "## What is new\n\n- A **fast** thing with `code`\n- A second thing\n\n"
    '<script>window.__pwned = 1</script>\n<img src=x onerror="window.__pwned = 1">\n'
)

MODEL_NAME = "FLE Studio E2E stub"
# Shorter than the 40 characters the stand-in engine echoes back in its answer.
QUESTION = "Hello from the browser end-to-end test"
EDITED = "Edited question for the versions step"
NEEDLE = "zebra-crossing-token"  # sits inside an attached file, to find it with the chat search
FORMAT_FIELDS = ("response_format", "json_schema", "grammar")
VIEWPORT = {"width": 1440, "height": 900}  # wide enough for the live panel to be open from the start
WAIT_MS = 30_000  # how long the browser waits for one thing to show up
MAX_SHOT_HEIGHT = 6000
PNG_MAGIC = b"\x89PNG\r\n\x1a\n"
# Makes the stand-in engine fast and gives the settings a sweet spot, so the tuner and the A/B/A test have something to
# find within a few seconds. Passed to the engine through the start settings ("Extra environment"), not to Studio.
FAST_ENV = "\n".join(
    [
        "FAKE_SPEED_MODEL=1",
        "FAKE_TOK_PER_SEC=3000",
        "FAKE_PROMPT_TOK_PER_SEC=200000",
        "FAKE_JITTER=0",
        "FAKE_RUN_NOISE=0.01",
        "FAKE_LOAD_SECONDS=0.3",
    ]
)
FLE_ENV = "\n".join([FAST_ENV, "FLE_MOE_CACHE_BUDGET_MIB=1024", "FAKE_ANSWER_TOKENS=1300"])

# Settings that would change how the stand-in or Studio behave must not leak in from the caller's shell.
SCRUBBED_PREFIXES = ("FAKE_", "FLE_", "LLAMA_ARG_")
SCRUBBED_NAMES = {"LLAMA_API_KEY", "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "NO_PROXY"}

# ------------------------------------------------------------------------------------------------ small helpers


class E2EError(Exception):
    """A check of the run failed; the message says what and why."""


def say(text: str = "") -> None:
    print(text, flush=True)


def first_line(exc: BaseException) -> str:
    text = str(exc).strip()
    line = text.splitlines()[0] if text else ""
    return line if isinstance(exc, E2EError) else f"{type(exc).__name__}: {line}"


class Step:
    def __init__(self, number: int, title: str) -> None:
        self.number, self.title, self.note = number, title, ""


class Report:
    """Prints one line per step: `[03] ok    title - note (1.2 s)`."""

    def __init__(self) -> None:
        self.number = 0
        self.screenshots: list[Path] = []

    @contextlib.contextmanager
    def step(self, title: str) -> Generator[Step, None, None]:
        self.number += 1
        step = Step(self.number, title)
        started = time.monotonic()
        try:
            yield step
        except Exception as exc:
            say(f"[{step.number:02d}] FAIL  {title} ({time.monotonic() - started:.1f} s): {first_line(exc)}")
            raise
        note = f" - {step.note}" if step.note else ""
        say(f"[{step.number:02d}] ok    {title}{note} ({time.monotonic() - started:.1f} s)")


def wait_until(probe: Callable[[], Any], timeout: float, what: str, interval: float = 0.25) -> Any:
    """Call `probe` until it returns something true; a server that does not answer yet counts as 'not yet'."""
    deadline = time.monotonic() + timeout
    last: Exception | None = None
    while True:
        try:
            value = probe()
            if value:
                return value
        except (OSError, ValueError) as exc:
            last = exc
        if time.monotonic() >= deadline:
            hint = f" (last error: {last})" if last else ""
            raise E2EError(f"timed out after {timeout:.0f} s waiting for {what}{hint}")
        time.sleep(interval)


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def port_open(port: int) -> bool:
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=1):
            return True
    except OSError:
        return False


def pid_alive(pid: int) -> bool:
    if sys.platform == "win32":
        # os.kill(pid, 0) would terminate the process on Windows, so ask the process API.
        import ctypes  # only needed on Windows

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)  # type: ignore[attr-defined]
        kernel32.OpenProcess.restype = ctypes.c_void_p
        kernel32.OpenProcess.argtypes = [ctypes.c_ulong, ctypes.c_int, ctypes.c_ulong]
        kernel32.GetExitCodeProcess.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_ulong)]
        kernel32.CloseHandle.argtypes = [ctypes.c_void_p]
        handle = kernel32.OpenProcess(0x1000, 0, pid)  # PROCESS_QUERY_LIMITED_INFORMATION
        if not handle:
            return False
        try:
            code = ctypes.c_ulong()
            return bool(kernel32.GetExitCodeProcess(handle, ctypes.byref(code))) and code.value == 259  # STILL_ACTIVE
        finally:
            kernel32.CloseHandle(handle)
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def kill_pid(pid: int) -> None:
    """SIGKILL where there is one; on Windows os.kill with SIGTERM is TerminateProcess."""
    with contextlib.suppress(OSError):
        os.kill(pid, getattr(signal, "SIGKILL", signal.SIGTERM))


_UNVERIFIED = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)  # a host Studio shows a certificate of its own making
_UNVERIFIED.check_hostname = False
_UNVERIFIED.verify_mode = ssl.CERT_NONE
_OPENER = urllib.request.build_opener(  # never through a proxy: it is all loopback
    urllib.request.ProxyHandler({}), urllib.request.HTTPSHandler(context=_UNVERIFIED)
)


def http(
    method: str, url: str, *, body: Any = None, headers: dict[str, str] | None = None, timeout: float = 30
) -> tuple[int, bytes]:
    data = None
    sent = dict(headers or {})
    if body is not None:
        data = json.dumps(body).encode("utf-8")
        sent["Content-Type"] = "application/json"
    request = urllib.request.Request(url, data=data, method=method, headers=sent)
    try:
        with _OPENER.open(request, timeout=timeout) as response:
            return response.status, response.read()
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read()


def number_in(text: str | None) -> float | None:
    """The first number in a piece of page text ('81.3tok/s' -> 81.3, '1,987' -> 1987)."""
    match = re.search(r"\d[\d,]*(?:\.\d+)?", text or "")
    return float(match.group(0).replace(",", "")) if match else None


# ------------------------------------------------------------------------------------------------ the model file


def gguf_stub() -> bytes:
    """A small but valid GGUF v3 file: the header of a tiny mixture-of-experts model with zeroed weights.

    It declares a 32k context so that the automatic plan can use the default context and the long prompt of the
    speed test (about 6,000 tokens) fits.
    """

    def text(value: str) -> bytes:
        raw = value.encode("utf-8")
        return struct.pack("<Q", len(raw)) + raw

    uint32, string, array = 4, 8, 9  # GGUF value types
    metadata = [
        ("general.architecture", string, text("llama")),
        ("general.name", string, text(MODEL_NAME)),
        ("llama.context_length", uint32, struct.pack("<I", 32768)),
        ("llama.block_count", uint32, struct.pack("<I", 2)),
        ("llama.expert_count", uint32, struct.pack("<I", 4)),
        ("llama.expert_used_count", uint32, struct.pack("<I", 2)),
        ("llama.attention.head_count", uint32, struct.pack("<I", 4)),
        ("llama.attention.head_count_kv", uint32, struct.pack("<I", 2)),
        ("llama.embedding_length", uint32, struct.pack("<I", 64)),
        ("tokenizer.chat_template", string, text("{{ messages }}")),
        ("tokenizer.ggml.tokens", array, struct.pack("<IQ", string, 100) + b"".join(text(f"t{i}") for i in range(100))),
    ]
    tensors = [
        ("token_embd.weight", (64, 100), 12800),
        ("blk.0.ffn_up_exps.weight", (64, 64, 4), 32768),
        ("blk.0.attn_q.weight", (64, 64), 8192),
    ]  # (name, dimensions, bytes as F16)
    out = b"GGUF" + struct.pack("<IQQ", 3, len(tensors), len(metadata))
    for key, kind, value in metadata:
        out += text(key) + struct.pack("<I", kind) + value
    offset = 0
    for name, dims, size in tensors:
        out += text(name) + struct.pack("<I", len(dims)) + b"".join(struct.pack("<Q", d) for d in dims)
        out += struct.pack("<IQ", 1, offset)  # type 1 = F16
        offset += size
    out += b"\0" * (-len(out) % 32)  # the data section starts on a 32-byte boundary
    return out + b"\0" * offset


# ------------------------------------------------------------------------------------------------ Studio


class StudioProcess:
    """FLE Studio as a child process, with its output in `studio.log` next to the screenshots."""

    def __init__(self, work: Path, out: Path, name: str = "studio", extra: tuple[str, ...] = ()) -> None:
        self.work, self.home = work, work / "home"
        self.log_path = out / f"{name}.log"
        self.extra = extra  # more command line, e.g. host mode
        self.scheme = "https" if "--tls" in extra else "http"
        self.sidekick: StudioProcess | None = None  # a second Studio (the host of the remote steps)
        self.proc: subprocess.Popen[bytes] | None = None
        self.port = 0
        self.base = ""
        self.token = ""
        self.version = ""
        self.engine_pid = 0
        self.engine_port = 0
        self.killed = False
        self.github: FakeGitHub | None = None
        self.env = {
            k: v
            for k, v in os.environ.items()
            if not k.upper().startswith(SCRUBBED_PREFIXES) and k.upper() not in SCRUBBED_NAMES
        }
        self.env.update({"PYTHONUNBUFFERED": "1", "PYTHONUTF8": "1", "NO_PROXY": "127.0.0.1,localhost"})

    def _ping(self) -> dict[str, Any] | None:
        try:
            status, raw = http("GET", self.base + "/api/ping", timeout=2)
            data = json.loads(raw.decode("utf-8"))
        except (OSError, ValueError):
            return None
        return data if status == 200 and data.get("app") == "fle-studio" else None

    def start(self) -> None:
        """Start the server on a free port and wait for it; the per-run token comes from the start page."""
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        exits: list[str] = []
        for _ in range(3):  # a port can be taken between looking for a free one and the server binding it
            self.port = free_port()
            self.base = f"{self.scheme}://127.0.0.1:{self.port}"
            cmd = [
                sys.executable,
                "-m",
                "fle_studio",
                "--home",
                str(self.home),
                "--port",
                str(self.port),
                "--no-browser",
                *self.extra,
            ]
            with self.log_path.open("ab") as log:
                log.write(f"$ {' '.join(cmd)}\n".encode())
                log.flush()
                self.proc = subprocess.Popen(
                    cmd, cwd=self.work, env=self.env, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT
                )
            deadline = time.monotonic() + 60
            ping = None
            while time.monotonic() < deadline and self.proc.poll() is None:
                ping = self._ping()
                if ping:
                    break
                time.sleep(0.2)
            if ping:
                self.version = str(ping.get("version"))
                break
            if self.proc.poll() is None:
                self.terminate()
                raise E2EError(f"Studio did not answer /api/ping within 60 s (see {self.log_path})")
            exits.append(f"exit code {self.proc.returncode} on port {self.port}")
        else:
            raise E2EError(f"Studio exited right after the start ({'; '.join(exits)}; see {self.log_path})")
        status, raw = http("GET", self.base + "/")
        match = re.search(rb'<meta name="fle-token" content="([^"]+)"', raw)
        if status != 200 or not match or match.group(1) == b"__FLE_TOKEN__":
            raise E2EError(f"The start page ({status}) carries no session token")
        self.token = match.group(1).decode()

    def api(self, method: str, path: str, body: Any = None, timeout: float = 30) -> Any:
        status, raw = http(method, self.base + path, body=body, headers={"X-FLE-Token": self.token}, timeout=timeout)
        try:
            data = json.loads(raw.decode("utf-8")) if raw else None
        except ValueError:
            data = None
        if status >= 400:
            message = data.get("error", {}).get("message") if isinstance(data, dict) else None
            raise E2EError(f"{method} {path} answered {status}: {message or raw[:200]!r}")
        return data

    def engine_state(self) -> dict[str, Any]:
        return self.api("GET", "/api/engine")  # type: ignore[no-any-return]

    def wait_engine(self, want: str, timeout: float) -> dict[str, Any]:
        def probe() -> dict[str, Any] | None:
            state = self.engine_state()
            if state["state"] == "crashed":
                lines = self.api("GET", "/api/engine/log?since=0")["lines"][-12:]
                tail = "\n".join(line["text"] for line in lines)
                raise E2EError(
                    f"the engine stopped with an error: {state.get('error')} "
                    f"(exit code {state.get('exit_code')})\n{tail}"
                )
            return state if state["state"] == want else None

        return wait_until(probe, timeout, f"the engine to be {want}")  # type: ignore[no-any-return]

    def terminate(self, timeout: float = 20) -> int | None:
        """Ask Studio to quit (SIGTERM; a hard kill on Windows) and kill it if it does not; returns its exit code."""
        proc = self.proc
        if proc is None:
            return None
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                self.killed = True
                proc.kill()
                proc.wait(timeout=10)
        return proc.returncode

    def cleanup(self) -> None:
        """At the very end, also after a failure: no Studio and no engine may stay behind."""
        if self.sidekick is not None:
            self.sidekick.cleanup()
        if self.proc is not None and self.proc.poll() is None:
            with contextlib.suppress(Exception):
                self.api("POST", "/api/engine/stop", timeout=10)
                wait_until(lambda: self.engine_state()["state"] in ("stopped", "crashed"), 15, "the engine to stop")
            self.terminate()
        if self.engine_pid and pid_alive(self.engine_pid):
            say(f"note: the engine process {self.engine_pid} was still running; killing it")
            kill_pid(self.engine_pid)
        if self.github is not None:
            with contextlib.suppress(Exception):
                self.github.__exit__(None, None, None)
            self.github = None


# ------------------------------------------------------------------------------------------------ browser helpers


class PageWatch:
    """Collects uncaught page errors and console errors; `check` raises if there were any since the last call."""

    def __init__(self, page: Page) -> None:
        self.errors: list[str] = []
        self.lines: list[str] = []
        self.tolerated: list[str] = []  # console texts a step expects (a host that refuses or is away: 424, 502)
        page.on("console", self._console)
        page.on("pageerror", self._page_error)

    def _console(self, msg: Any) -> None:
        loc = msg.location or {}
        where = f" ({loc.get('url')}:{loc.get('lineNumber')})" if loc.get("url") else ""
        self.lines.append(f"{msg.type}: {msg.text}{where}")
        if msg.type == "error" and not any(t in msg.text for t in self.tolerated):
            self.errors.append(f"console error: {msg.text}{where}")

    def _page_error(self, exc: Any) -> None:
        self.lines.append(f"pageerror: {exc}")
        self.errors.append(f"uncaught page error: {exc}")

    def check(self, where: str) -> None:
        if self.errors:
            found, self.errors = self.errors, []
            raise E2EError(f"{len(found)} browser error(s) during '{where}': " + " | ".join(found[:5]))


def background_luma(page: Page) -> float:
    """How bright the page background is, 0 (black) to 255 (white)."""
    color = page.evaluate("() => getComputedStyle(document.body).backgroundColor")
    numbers = re.findall(r"\d+(?:\.\d+)?", color)[:3]
    if len(numbers) < 3:
        raise E2EError(f"cannot read the page background colour: {color!r}")
    red, green, blue = (float(n) for n in numbers)
    return 0.2126 * red + 0.7152 * green + 0.0722 * blue


def wait_for_look(page: Page, want: str, why: str) -> None:
    def probe() -> bool:
        return ("dark" if background_luma(page) < 128 else "light") == want

    wait_until(probe, 10, f"the page to look {want} ({why})", interval=0.1)


# Height the page needs to show everything: the scrolling area of a page sits below the top bar.
_NEEDED_HEIGHT = """() => {
  const page = document.querySelector('.page');
  const below = page ? page.getBoundingClientRect().top + page.scrollHeight : 0;
  return Math.ceil(Math.max(window.innerHeight, below));
}"""


def full_page_screenshot(page: Page, path: Path) -> None:
    """FLE Studio scrolls inside .page, not the document, so make the window as tall as the page for the shot."""
    size = page.viewport_size or dict(VIEWPORT)
    needed = min(int(page.evaluate(_NEEDED_HEIGHT)), MAX_SHOT_HEIGHT)
    grown = needed > size["height"]
    if grown:
        page.set_viewport_size({"width": size["width"], "height": needed})
        page.wait_for_timeout(300)  # charts redraw on resize
    try:
        page.evaluate("() => document.fonts.ready.then(() => true)")
        page.screenshot(path=str(path), full_page=True, animations="disabled", caret="hide")
    finally:
        if grown:
            page.set_viewport_size(size)
            page.wait_for_timeout(200)


class Session:
    """The browser side of the run: one page whose errors are watched, and the steps that use it."""

    def __init__(
        self,
        report: Report,
        studio: StudioProcess,
        page: Page,
        watch: PageWatch,
        out: Path,
        expect: Any,
        model: dict[str, Any],
    ) -> None:
        self.report, self.studio, self.page, self.watch = report, studio, page, watch
        self.out, self.expect, self.model = out, expect, model
        self.chat_id = ""
        self.sent: list[dict[str, Any]] = []  # the bodies of the chat requests the page sent to the engine
        page.on("request", self._on_request)

    def _on_request(self, request: Any) -> None:
        if request.method == "POST" and request.url.split("?")[0].endswith("/v1/chat/completions"):
            try:
                self.sent.append(json.loads(request.post_data or "{}"))
            except ValueError:
                self.sent.append({})

    @contextlib.contextmanager
    def step(self, title: str) -> Generator[Step, None, None]:
        """A report step that also fails on browser errors seen while it ran."""
        with self.report.step(title) as step:
            yield step
            self.watch.check(title)

    # ---- small actions

    def go(self, label: str, path: str, heading: str) -> None:
        """Click a sidebar link and wait for the page."""
        self.page.locator("#nav").get_by_role("link", name=label, exact=True).click()
        self.expect(self.page).to_have_url(self.studio.base + path, timeout=WAIT_MS)
        self.expect(self.page.locator(".page-head h2")).to_have_text(heading, timeout=WAIT_MS)

    def shoot(self, name: str) -> str:
        """Screenshot the current page in light and in dark mode (the OS setting, with the theme on 'system')."""
        if self.page.evaluate("() => document.documentElement.hasAttribute('data-theme')"):
            raise E2EError(f"the theme is not 'like the system' before the {name} screenshots")
        digests = {}
        for scheme in ("light", "dark"):
            self.page.emulate_media(color_scheme=scheme)
            wait_for_look(self.page, scheme, f"{name} with a {scheme} system setting")
            if self.page.locator("iframe.mcp-app").count():  # an MCP App follows the setting a moment later
                self.expect(self.page.frame_locator("iframe.mcp-app").locator("#theme")).to_have_text(
                    scheme, timeout=WAIT_MS
                )
            path = self.out / f"{name}-{scheme}.png"
            full_page_screenshot(self.page, path)
            data = path.read_bytes()
            if not data.startswith(PNG_MAGIC) or len(data) < 10_000:
                raise E2EError(f"{path.name} is not a plausible screenshot ({len(data)} bytes)")
            digests[scheme] = hashlib.sha256(data).hexdigest()
            self.report.screenshots.append(path)
        self.page.emulate_media(color_scheme="light")
        if digests["light"] == digests["dark"]:
            raise E2EError(f"the light and dark screenshots of {name} are identical")
        return f"{name}-light.png, {name}-dark.png"

    # ---- steps

    def open_app(self) -> None:
        with self.step("Open the app") as step:
            self.page.goto(self.studio.base + "/", wait_until="domcontentloaded")
            self.expect(self.page).to_have_title("FLE Studio", timeout=WAIT_MS)
            self.expect(self.page.locator("#brand-ver")).to_have_text(f"v{self.studio.version}", timeout=WAIT_MS)
            self.expect(self.page.locator("#nav a")).to_have_count(7, timeout=WAIT_MS)
            self.expect(self.page.locator("#engine-card .pill")).to_have_text("Ready", timeout=WAIT_MS)
            self.expect(self.page.locator(".welcome h2")).to_contain_text(
                f"What should {self.model['name']} work on?", timeout=WAIT_MS
            )
            step.note = f"FLE Studio v{self.studio.version}, engine card says Ready"

    def chat(self) -> None:
        page, studio = self.page, self.studio
        with self.step("Send a chat message and read the streamed answer") as step:
            # Record how the answer grows, to tell a stream from one lump that arrives at the end.
            page.evaluate("""() => {
              window.__e2e = { lengths: [] };
              const grab = () => {
                const el = document.querySelector('.msg.assistant .body > .md');
                const n = el ? el.textContent.length : 0;
                const seen = window.__e2e.lengths;
                if (n && seen[seen.length - 1] !== n) seen.push(n);
              };
              new MutationObserver(grab).observe(document.getElementById('main'),
                { subtree: true, childList: true, characterData: true });
            }""")
            page.locator("textarea[aria-label='Message']").fill(QUESTION)
            page.locator("button.send").click()
            self.expect(page.locator(".msg.user .bubble")).to_have_text(QUESTION, timeout=WAIT_MS)
            self.expect(page.locator(".msg.assistant .stats")).to_be_visible(timeout=60_000)  # the answer is complete
            self.expect(page.locator("button.send:not(.stop)")).to_be_visible(timeout=WAIT_MS)
            answer = " ".join(page.locator(".msg.assistant .body > .md").inner_text().split())
            if f'Fake reply to "{QUESTION}".' not in answer:
                raise E2EError(f"unexpected answer text: {answer[:200]!r}")
            lengths = page.evaluate("() => window.__e2e.lengths")
            if len(lengths) < 3:
                raise E2EError(f"the answer did not stream: its length went through {lengths} only")
            label = page.locator(".msg.assistant details.think summary .lbl").inner_text().strip()
            if not label.startswith("Thought"):
                raise E2EError(f"no reasoning block above the answer (label {label!r})")
            speed = number_in(page.locator(".msg.assistant .stats .hot b").inner_text())
            if not speed or speed <= 0:
                raise E2EError("the answer carries no tok/s figure")
            # The chat is saved while and after it streams; wait until the final version is stored.
            page.wait_for_function("() => /^\\/c\\/[0-9a-f]{16}$/.test(location.pathname)", timeout=WAIT_MS)
            self.chat_id = page.evaluate("() => location.pathname.split('/').pop()")

            def stored() -> dict[str, Any] | None:
                chat = studio.api("GET", f"/api/chats/{self.chat_id}")
                messages = chat.get("messages") or []
                answer_saved = len(messages) == 2 and messages[1].get("content")
                return chat if answer_saved and (messages[1].get("stats") or {}).get("predicted_n") else None

            chat = wait_until(stored, 15, "the finished chat to be saved")
            roles = [m["role"] for m in chat["messages"]]
            if roles != ["user", "assistant"] or not chat["messages"][1].get("reasoning"):
                raise E2EError(
                    f"the saved chat looks wrong: roles {roles}, reasoning kept: "
                    f"{bool(chat['messages'][1].get('reasoning'))}"
                )
            listed = [c["id"] for c in studio.api("GET", "/api/chats")["chats"]]
            if listed != [self.chat_id]:
                raise E2EError(f"the chat list is {listed}, expected just {self.chat_id}")
            step.note = f"answer streamed in {len(lengths)} steps, {speed:g} tok/s, {label.lower()}"

    def live_panel(self) -> None:
        page = self.page
        with self.step("Live panel shows a decode speed") as step:
            panel = page.locator("aside.live-panel")
            self.expect(panel).to_be_visible(timeout=WAIT_MS)
            tile = panel.locator(".tile", has_text="Generation")
            page.wait_for_function(
                """() => {
                  const tile = [...document.querySelectorAll('.live-panel .tile')]
                    .find((t) => t.textContent.includes('Generation'));
                  const value = tile && tile.querySelector('.v');
                  return !!value && /\\d/.test(value.textContent);
                }""",
                timeout=WAIT_MS,
            )
            text = tile.locator(".v").text_content()
            speed = number_in(text)
            if not speed or not 1 <= speed <= 100_000 or "tok/s" not in (text or ""):
                raise E2EError(f"the Generation tile shows {text!r}, not a decode speed")
            prefill = number_in(panel.locator(".tile", has_text="Prompt reading").locator(".v").text_content())

            # The tile falls back on the last answer when the engine has no /metrics; the sampler must have seen it.
            def sampled() -> dict[str, Any] | None:
                live = self.studio.api("GET", "/api/live")
                seen = live["metrics_available"] and any((s.get("decode") or 0) > 0 for s in live["history"])
                return live if seen else None

            wait_until(sampled, 15, "the live sampler to record a decode speed from /metrics")
            step.note = f"Generation {speed:g} tok/s" + (f", prompt reading {prefill:g} tok/s" if prefill else "")

    def reload_chat(self) -> None:
        page = self.page
        with self.step("Reload the saved chat and screenshot it") as step:
            page.goto(f"{self.studio.base}/c/{self.chat_id}", wait_until="domcontentloaded")
            self.expect(page.locator(".msg.user .bubble")).to_have_text(QUESTION, timeout=WAIT_MS)
            self.expect(page.locator(".msg.assistant .body > .md")).to_contain_text("Fake reply to", timeout=WAIT_MS)
            self.expect(page.locator(".msg.assistant details.think")).to_have_count(1, timeout=WAIT_MS)
            self.expect(page.locator("#chat-list .chat-item.active")).to_have_count(1, timeout=WAIT_MS)
            self.expect(page.locator(".live-panel .tile", has_text="Generation").locator(".v")).to_contain_text(
                "tok/s", timeout=WAIT_MS
            )
            step.note = self.shoot("chat")

    # ---- chat features: attachments, versions, export and import, search, tools

    def new_chat(self) -> None:
        self.page.goto(self.studio.base + "/", wait_until="domcontentloaded")
        self.expect(self.page.locator(".welcome h2")).to_be_visible(timeout=WAIT_MS)
        self.expect(self.page.locator("button.send")).to_be_visible(timeout=WAIT_MS)

    def send(self, text: str) -> None:
        self.page.locator("textarea[aria-label='Message']").fill(text)
        self.page.locator("button.send").click()

    def idle(self) -> None:
        """The whole turn is over (streaming, tools and all)."""
        self.expect(self.page.locator("button.send:not(.stop)")).to_be_visible(timeout=60_000)

    def saved_chat(self, want: Callable[[dict[str, Any]], bool], what: str) -> dict[str, Any]:
        page, studio = self.page, self.studio
        page.wait_for_function("() => /^\\/c\\/[0-9a-f]{16}$/.test(location.pathname)", timeout=WAIT_MS)
        chat_id = page.evaluate("() => location.pathname.split('/').pop()")
        found = wait_until(
            lambda: (lambda c: c if want(c) else None)(studio.api("GET", f"/api/chats/{chat_id}")), 15, what
        )
        return found  # type: ignore[no-any-return]

    def attachments(self) -> None:
        page = self.page
        attach = page.locator("input[aria-label='Attach files']")
        with self.step("Attach a text file: it goes to the model in front of the question") as step:
            self.new_chat()
            notes = f"Project notes\n{NEEDLE}\n".encode() + b"plain words " * 40
            attach.set_input_files([{"name": "notes.txt", "mimeType": "text/plain", "buffer": notes}])
            chip = page.locator(".att-row .att-chip.text")
            self.expect(chip).to_contain_text("notes.txt", timeout=WAIT_MS)
            self.expect(chip).to_contain_text("tokens", timeout=WAIT_MS)
            self.send("What do the notes say?")
            self.expect(page.locator(".msg.user .att-file summary .n")).to_have_text("notes.txt", timeout=WAIT_MS)
            self.expect(page.locator(".msg.assistant .stats")).to_be_visible(timeout=60_000)
            self.idle()
            sent = self.sent[-1]["messages"][-1]["content"]
            if not isinstance(sent, str) or not (
                sent.startswith('<attachment name="notes.txt">')
                and NEEDLE in sent
                and sent.endswith("What do the notes say?")
            ):
                raise E2EError(f"the attachment did not reach the model in front of the question: {str(sent)[:200]!r}")
            self.expect(page.locator(".att-row .att-chip")).to_have_count(0, timeout=WAIT_MS)  # taken by the send
            saved = self.saved_chat(
                lambda c: len(c["messages"]) == 2 and c["messages"][1].get("content"),
                "the chat with the attachment to be saved",
            )
            kept = (saved["messages"][0].get("attachments") or [{}])[0]
            if kept.get("name") != "notes.txt" or NEEDLE not in kept.get("text", ""):
                raise E2EError(f"the saved chat lost the attachment: {kept}")
            step.note = "sent as an <attachment> block, kept with the chat; " + self.shoot("chat-attachment")

        with self.step("A picture is turned away politely when the model cannot see") as step:
            self.new_chat()
            png = b"\x89PNG\r\n\x1a\n" + b"\0" * 32
            attach.set_input_files([{"name": "cat.png", "mimeType": "image/png", "buffer": png}])
            self.expect(page.locator("#toasts .toast.bad")).to_contain_text("cannot look at pictures", timeout=WAIT_MS)
            self.expect(page.locator(".att-row .att-chip")).to_have_count(0, timeout=WAIT_MS)
            step.note = "the page says so before anything is uploaded"

        with self.step("A long attachment shows the context warning") as step:
            big = b"lorem ipsum dolor " * 5600  # about 28,000 tokens of the 32,768 the engine holds
            attach.set_input_files([{"name": "big.txt", "mimeType": "text/plain", "buffer": big}])
            self.expect(page.locator(".att-row .att-chip.text")).to_contain_text("big.txt", timeout=WAIT_MS)
            warn = page.locator(".ctx-warn")
            self.expect(warn).to_be_visible(timeout=WAIT_MS)
            self.expect(warn).to_contain_text("tokens", timeout=WAIT_MS)
            step.note = "warning above the message box; " + self.shoot("chat-context-warning")
            page.get_by_role("button", name="Remove big.txt").click()
            self.expect(warn).to_be_hidden(timeout=WAIT_MS)

    def versions(self) -> None:
        page = self.page
        user_switch = page.locator(".msg.user .switcher .num")
        with self.step("Edit a message: it becomes version 2, the first stays one click away") as step:
            page.goto(f"{self.studio.base}/c/{self.chat_id}", wait_until="domcontentloaded")
            self.expect(page.locator(".msg.user .bubble")).to_have_text(QUESTION, timeout=WAIT_MS)
            self.expect(page.locator(".msg .switcher")).to_have_count(0, timeout=WAIT_MS)
            user = page.locator(".msg.user")
            user.hover()
            user.get_by_role("button", name="Edit", exact=True).click()
            page.get_by_label("Edit your message").fill(EDITED)
            page.get_by_role("button", name="Send as new version").click()
            self.expect(page.locator(".msg.user .bubble")).to_have_text(EDITED, timeout=WAIT_MS)
            self.expect(page.locator(".msg.assistant .stats")).to_be_visible(timeout=60_000)
            self.idle()
            self.expect(user_switch).to_have_text("2 / 2", timeout=WAIT_MS)
            asked = [m["content"] for m in self.sent[-1]["messages"] if m["role"] == "user"]
            if asked != [EDITED]:
                raise E2EError(f"the engine was sent {asked} instead of only the edited question")
            self.saved_chat(lambda c: len(c["messages"]) == 4, "both versions to be saved")
            page.get_by_role("button", name="Previous version").click()
            self.expect(page.locator(".msg.user .bubble")).to_have_text(QUESTION, timeout=WAIT_MS)
            self.expect(user_switch).to_have_text("1 / 2", timeout=WAIT_MS)
            self.expect(page.locator(".msg.assistant .body > .md")).to_contain_text(QUESTION, timeout=WAIT_MS)
            step.note = "< 1 / 2 > shows the first question and its own answer; " + self.shoot("chat-versions")
            page.get_by_role("button", name="Next version").click()
            self.expect(page.locator(".msg.user .bubble")).to_have_text(EDITED, timeout=WAIT_MS)

        with self.step("Answer again: the first answer stays as another version") as step:
            last = page.locator(".msg.assistant").last
            last.hover()
            last.get_by_role("button", name="Regenerate").click()
            self.expect(page.locator(".msg.assistant .switcher .num")).to_have_text("2 / 2", timeout=WAIT_MS)
            self.expect(page.locator(".msg.assistant .stats")).to_be_visible(timeout=60_000)
            self.idle()
            self.saved_chat(lambda c: len(c["messages"]) == 5, "the second answer to be saved")
            answer_switch = page.locator(".msg.assistant .switcher")
            answer_switch.get_by_role("button", name="Previous version").click()
            self.expect(page.locator(".msg.assistant .switcher .num")).to_have_text("1 / 2", timeout=WAIT_MS)
            answer_switch.get_by_role("button", name="Next version").click()
            self.expect(page.locator(".msg.assistant .switcher .num")).to_have_text("2 / 2", timeout=WAIT_MS)
            step.note = "5 messages in the file, one question has 2 versions and one answer has 2"

    def export_import(self) -> None:
        page, studio = self.page, self.studio
        before = len(studio.api("GET", "/api/chats")["chats"])
        with self.step("Export the chat as Markdown and as JSON") as step:
            page.get_by_role("button", name="Chat menu").click()
            with page.expect_download() as download:
                page.get_by_role("menuitem", name="Export as Markdown").click()
            md_path = self.out / "export.md"
            download.value.save_as(str(md_path))
            markdown = md_path.read_text(encoding="utf-8")
            filename = download.value.suggested_filename
            if not filename.endswith(".md") or EDITED not in markdown or "## Assistant" not in markdown:
                raise E2EError(f"the Markdown export looks wrong ({filename}): {markdown[:200]!r}")
            if QUESTION in markdown.split("\n", 1)[1]:  # the title line still carries the first question
                raise E2EError("the Markdown export holds the other version of the question too")
            page.get_by_role("button", name="Chat menu").click()
            with page.expect_download() as download:
                page.get_by_role("menuitem", name="Export as JSON").click()
            json_path = self.out / "export.json"
            download.value.save_as(str(json_path))
            doc = json.loads(json_path.read_text(encoding="utf-8"))
            if doc.get("format") != "fle-studio-chat" or len(doc["chat"]["messages"]) != 5:
                raise E2EError(
                    f"the JSON export is not the whole tree: {doc.get('format')}, "
                    f"{len(doc.get('chat', {}).get('messages', []))} messages"
                )
            step.note = "Markdown has the version on screen, JSON has all 5 messages"

        with self.step("Import the JSON file as a new chat") as step:
            page.locator("input[aria-label='Import a chat file']").set_input_files(str(json_path))
            self.expect(page.locator("#toasts .toast.good")).to_contain_text("Imported", timeout=WAIT_MS)
            page.wait_for_function(
                "(old) => /^\\/c\\/[0-9a-f]{16}$/.test(location.pathname) && !location.pathname.endsWith(old)",
                arg=self.chat_id,
                timeout=WAIT_MS,
            )
            self.expect(page.locator(".msg.user .bubble")).to_have_text(EDITED, timeout=WAIT_MS)
            self.expect(page.locator(".msg.user .switcher .num")).to_have_text("2 / 2", timeout=WAIT_MS)
            after = len(studio.api("GET", "/api/chats")["chats"])
            if after != before + 1:
                raise E2EError(f"the chat list went from {before} to {after} chats")
            step.note = "opens as a copy with its versions"

    def search(self) -> None:
        page = self.page
        box = page.locator("#chat-search")
        hits = page.locator("#chat-list .chat-item.hit")
        with self.step("Search the chats: text inside an attachment and in an older version") as step:
            box.fill("zebra-crossing")
            self.expect(hits).to_have_count(1, timeout=WAIT_MS)
            self.expect(hits.first).to_contain_text("zebra-crossing", timeout=WAIT_MS)
            self.expect(hits.first.locator("mark")).to_have_text("zebra-crossing", timeout=WAIT_MS)
            box.fill("Hello from the browser")  # only in the first version of the question, in two chats
            self.expect(hits).to_have_count(2, timeout=WAIT_MS)
            box.fill("nothing like this anywhere")
            self.expect(page.locator("#chat-list .chat-empty")).to_be_visible(timeout=WAIT_MS)
            box.fill("zebra-crossing")
            self.expect(hits).to_have_count(1, timeout=WAIT_MS)
            step.note = "hits are highlighted; " + self.shoot("chat-search")
            box.fill("")
            self.expect(hits).to_have_count(0, timeout=WAIT_MS)
            self.expect(page.locator("#chat-list .chat-item")).not_to_have_count(0, timeout=WAIT_MS)

    def tools(self) -> None:
        page, studio = self.page, self.studio
        card = page.locator(".toolcard")
        with self.step("The model calls a built-in tool and the card shows call and result") as step:
            self.new_chat()
            self.expect(page.locator("button.chip", has_text="Tools ·")).to_be_visible(timeout=WAIT_MS)
            self.send('CALL_TOOL calculator {"expression": "6*7"}')
            self.expect(card).to_have_count(1, timeout=WAIT_MS)
            self.expect(card.locator(".tname")).to_have_text("calculator", timeout=WAIT_MS)
            self.expect(card.locator(".pill")).to_contain_text("Done", timeout=WAIT_MS)
            card.locator("summary").click()
            self.expect(card.locator(".tbody")).to_be_visible(timeout=WAIT_MS)
            self.expect(card.locator(".tbody pre").last).to_have_text("42", timeout=WAIT_MS)
            self.expect(page.locator(".msg.assistant .body > .md").last).to_contain_text("Tool said", timeout=60_000)
            self.idle()
            first, second = self.sent[-2], self.sent[-1]
            names = [t["function"]["name"] for t in first.get("tools") or []]
            answers = [m for m in second["messages"] if m["role"] == "tool"]
            if "calculator" not in names or not answers or answers[-1]["content"] != "42":
                raise E2EError(f"the loop did not carry the tool: tools {names}, tool messages {answers}")
            step.note = "request 1 offered the tools, request 2 carried the result; " + self.shoot("chat-tool")

        with self.step("An MCP server's tool waits for approval, then runs once allowed") as step:
            config = json.dumps({"mcpServers": {"fake": {"command": sys.executable, "args": [str(FAKE_MCP)]}}})
            studio.api("POST", "/api/settings", {"tools": {"mcp_enabled": True, "mcp_json": config}})
            studio.api("POST", "/api/tools/reload")
            wait_until(
                lambda: any(t["name"] == "fake__write_note" for t in studio.api("GET", "/api/tools")["tools"]),
                30,
                "the MCP server's tools to be listed",
            )
            self.new_chat()
            self.expect(page.locator("button.chip", has_text="Tools ·")).to_be_visible(timeout=WAIT_MS)
            self.send('CALL_TOOL fake__write_note {"text": "from the e2e test"}')
            self.expect(card.locator(".pill")).to_contain_text("Waiting for you", timeout=WAIT_MS)
            self.expect(card.get_by_role("button", name="Allow once")).to_be_visible(timeout=WAIT_MS)
            self.expect(card.get_by_role("button", name="Always allow")).to_be_visible(timeout=WAIT_MS)
            waiting = self.shoot("chat-mcp-approval")
            card.get_by_role("button", name="Allow once").click()
            self.expect(card.locator(".pill")).to_contain_text("Done", timeout=WAIT_MS)
            self.expect(card.locator(".tbody")).to_contain_text("saved note 1", timeout=WAIT_MS)
            self.expect(page.locator(".msg.assistant .body > .md").last).to_contain_text("Tool said", timeout=60_000)
            self.idle()
            step.note = "asked first, ran after 'Allow once'; " + waiting

        with self.step("Declining a tool call tells the model so") as step:
            self.send('CALL_TOOL fake__write_note {"text": "not this one"}')
            self.expect(card).to_have_count(2, timeout=WAIT_MS)
            self.expect(card.last.locator(".pill")).to_contain_text("Waiting for you", timeout=WAIT_MS)
            card.last.get_by_role("button", name="Decline").click()
            self.expect(card.last.locator(".pill")).to_contain_text("Declined", timeout=WAIT_MS)
            self.expect(page.locator(".msg.assistant .body > .md").last).to_contain_text("declined", timeout=60_000)
            self.idle()
            notes = studio.api(
                "POST",
                "/api/tools/call",
                {"name": "fake__write_note", "arguments": {"text": "probe"}, "approved": True},
            )["content"]
            if notes != "saved note 2":
                raise E2EError(f"the declined call still ran on the server (the next note is {notes!r}, expected 2)")
            step.note = "the server never ran it"

        with self.step("No request carried tools together with a forced answer format") as step:
            bad = [i for i, body in enumerate(self.sent) if body.get("tools") and any(k in body for k in FORMAT_FIELDS)]
            if bad:
                raise E2EError(f"requests {bad} carried tools and a forced answer format")
            offered = sum(1 for body in self.sent if body.get("tools"))
            step.note = f"{len(self.sent)} chat requests checked, {offered} offered tools"

    def mcp_remote(self) -> None:
        """A web MCP server that wants a sign-in (OAuth), and an MCP App in the chat."""
        page, studio = self.page, self.studio
        port = free_port()
        proc = subprocess.Popen(
            [sys.executable, str(FAKE_MCP), "--http", str(port), "--oauth"],
            env={k: v for k, v in os.environ.items() if not k.startswith("FAKE_")},
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
        try:
            assert proc.stderr is not None
            proc.stderr.readline()
            remote = f"http://127.0.0.1:{port}"
            card = page.locator(".toolcard")
            row = page.locator(".mcp-row[data-server='web']")

            with self.step("A web MCP server asks for a sign-in, and Sign in connects it") as step:
                config = json.dumps(
                    {
                        "mcpServers": {
                            "fake": {"command": sys.executable, "args": [str(FAKE_MCP)]},
                            "web": {"url": remote + "/mcp"},
                        }
                    }
                )
                studio.api(
                    "POST", "/api/settings", {"tools": {"mcp_enabled": True, "mcp_json": config, "mcp_trusted": []}}
                )
                studio.api("POST", "/api/tools/reload")
                wait_until(
                    lambda: any(
                        sv["name"] == "web" and sv["status"] == "auth_required"
                        for sv in studio.api("GET", "/api/tools")["servers"]
                    ),
                    30,
                    "the web server to ask for a sign-in",
                )
                self.go("Settings", "/settings", "Settings")
                self.expect(row.locator(".pill")).to_have_text("Sign-in needed", timeout=WAIT_MS)
                self.expect(row.locator("[data-act='signin']")).to_be_visible(timeout=WAIT_MS)
                needed = self.shoot("settings-mcp-signin-needed")
                with page.context.expect_page() as popup_info:
                    row.locator("[data-act='signin']").click()
                popup = popup_info.value
                popup.wait_for_url(re.compile(r"/oauth/callback\?"), timeout=WAIT_MS)
                self.expect(popup.locator("h1")).to_have_text("Signed in", timeout=WAIT_MS)
                popup.close()
                self.expect(row.locator(".pill")).to_contain_text("tool", timeout=WAIT_MS)
                self.expect(row.locator("[data-act='signout']")).to_be_visible(timeout=WAIT_MS)
                files = list((studio.home / "oauth").glob("*.json"))
                if len(files) != 1:
                    raise E2EError(f"expected one token file in the data folder, found {files}")
                if os.name != "nt" and (files[0].stat().st_mode & 0o777) != 0o600:
                    raise E2EError(f"the token file has mode {oct(files[0].stat().st_mode & 0o777)}, not 0600")
                page_text = page.content() + json.dumps(studio.api("GET", "/api/tools"))
                token = json.loads(files[0].read_text(encoding="utf-8"))["access_token"]
                if token in page_text or token in json.dumps(studio.api("GET", "/api/settings")):
                    raise E2EError("the access token reached the page or the settings")
                step.note = (
                    "the 401 led to discovery, registration and PKCE; token file mode 0600; "
                    + needed
                    + "; "
                    + self.shoot("settings-mcp-signed-in")
                )

            with self.step("The tool of an MCP App shows its app in the chat, inside a sandboxed frame") as step:
                wait_until(
                    lambda: any(t["name"] == "web__show_chart" for t in studio.api("GET", "/api/tools")["tools"]),
                    30,
                    "the web server's tools to be listed",
                )
                self.new_chat()
                self.expect(page.locator("button.chip", has_text="Tools ·")).to_be_visible(timeout=WAIT_MS)
                self.send('CALL_TOOL web__show_chart {"title": "E2E chart"}')
                self.expect(card.locator(".pill")).to_contain_text("Waiting for you", timeout=WAIT_MS)
                card.get_by_role("button", name="Allow once").click()
                frame_el = page.locator("iframe.mcp-app")
                self.expect(frame_el).to_have_count(1, timeout=WAIT_MS)
                self.expect(frame_el).to_have_attribute("sandbox", "allow-scripts", timeout=WAIT_MS)
                app = page.frame_locator("iframe.mcp-app")
                self.expect(app.locator("#title")).to_have_text("E2E chart", timeout=WAIT_MS)
                self.expect(app.locator("#bars i")).to_have_count(3, timeout=WAIT_MS)
                self.expect(app.locator("#input")).to_contain_text("E2E chart", timeout=WAIT_MS)
                inside = next(f for f in page.frames if "/app-frame/" in f.url)
                origin = inside.evaluate("() => window.origin")
                reach = inside.evaluate(
                    "() => { try { return typeof parent.document.title; } catch (e) { return 'blocked'; } }"
                )
                storage = inside.evaluate(
                    "() => { try { return typeof localStorage; } catch (e) { return 'blocked'; } }"
                )
                fetched = inside.evaluate("() => fetch('/api/tools').then(() => 'reached', () => 'blocked')")
                blocked = [e for e in self.watch.errors if "Content Security Policy" in e]
                if not blocked:
                    raise E2EError("the browser did not report that the app's fetch broke the frame's CSP")
                self.watch.errors = [e for e in self.watch.errors if "Content Security Policy" not in e]  # expected
                if origin != "null" or reach != "blocked" or storage != "blocked" or fetched != "blocked":
                    raise E2EError(
                        f"the app is not isolated: origin {origin}, parent {reach}, storage {storage}, fetch {fetched}"
                    )
                self.idle()
                step.note = "iframe sandbox=allow-scripts; origin null; no access to the page, storage or Studio's API"

            with self.step("The app follows the light and dark setting") as step:
                self.expect(app.locator("#theme")).to_have_text("light", timeout=WAIT_MS)
                page.emulate_media(color_scheme="dark")
                self.expect(app.locator("#theme")).to_have_text("dark", timeout=WAIT_MS)
                dark_bg = inside.evaluate("() => getComputedStyle(document.body).backgroundColor")
                page.emulate_media(color_scheme="light")
                self.expect(app.locator("#theme")).to_have_text("light", timeout=WAIT_MS)
                light_bg = inside.evaluate("() => getComputedStyle(document.body).backgroundColor")
                if dark_bg == light_bg:
                    raise E2EError(f"the app looks the same in both themes ({dark_bg})")
                step.note = f"dark {dark_bg}, light {light_bg}; " + self.shoot("chat-mcp-app")

            with self.step("The app calls a tool of its own server, and the confirmation rule applies") as step:
                dialog = page.locator(".modal")
                app.locator("#b1").click()
                self.expect(dialog).to_contain_text("web", timeout=WAIT_MS)
                dialog.get_by_role("button", name="Cancel").click()
                self.expect(app.locator("#bump")).to_contain_text("declined", timeout=WAIT_MS)
                app.locator("#b1").click()
                dialog.get_by_role("button", name="Allow once").click()
                self.expect(app.locator("#bump")).to_have_text("1", timeout=WAIT_MS)
                app.locator("#b4").click()  # a writing tool, from the app
                dialog.get_by_role("button", name="Allow once").click()
                self.expect(app.locator("#reply")).to_contain_text("saved note", timeout=WAIT_MS)
                step.note = (
                    "declined once (nothing ran), allowed once (count 1), writing call asked too; "
                    + self.shoot("chat-mcp-app-call")
                )

            with self.step("The app can put a message into the chat only after you agree; links ask too") as step:
                app.locator("#b3").click()
                self.expect(dialog).to_contain_text("example.com", timeout=WAIT_MS)
                dialog.get_by_role("button", name="Cancel").click()
                self.expect(app.locator("#reply")).to_have_text("refused", timeout=WAIT_MS)
                app.locator("#b2").click()
                self.expect(dialog).to_contain_text("Hello from the app", timeout=WAIT_MS)
                dialog.get_by_role("button", name="Send").click()
                self.expect(page.locator(".msg.user .bubble").last).to_have_text("Hello from the app", timeout=WAIT_MS)
                self.idle()
                step.note = "the app's text became a user message after 'Send'; the link was declined"

            with self.step("After a reload the app starts with a click, and signing out ends the access") as step:
                page.reload(wait_until="domcontentloaded")
                self.expect(page.locator(".toolcard").first).to_be_visible(timeout=WAIT_MS)
                self.expect(page.locator("iframe.mcp-app")).to_have_count(0, timeout=WAIT_MS)
                page.locator(".app-open").first.click()
                self.expect(page.frame_locator("iframe.mcp-app").locator("#title")).to_have_text(
                    "E2E chart", timeout=WAIT_MS
                )
                self.go("Settings", "/settings", "Settings")
                self.expect(row.locator("[data-act='signout']")).to_be_visible(timeout=WAIT_MS)
                row.locator("[data-act='signout']").click()
                self.expect(row.locator(".pill")).to_have_text("Sign-in needed", timeout=WAIT_MS)
                left = list((studio.home / "oauth").glob("*.json"))
                if left:
                    raise E2EError(f"signing out left token files behind: {left}")
                step.note = "reloaded app shown on request; token file gone after Sign out"

            with self.step("Put the settings back for the next steps") as step:
                config = json.dumps({"mcpServers": {"fake": {"command": sys.executable, "args": [str(FAKE_MCP)]}}})
                studio.api("POST", "/api/settings", {"tools": {"mcp_json": config}})
                studio.api("POST", "/api/tools/reload")
                wait_until(
                    lambda: any(t["name"] == "fake__write_note" for t in studio.api("GET", "/api/tools")["tools"]),
                    30,
                    "the stdio server to be back",
                )
                step.note = "only the stdio server is left"
        finally:
            proc.kill()
            proc.wait(timeout=10)
            if proc.stderr:
                proc.stderr.close()

    def settings_tools(self) -> None:
        page = self.page
        with self.step("Settings lists the tools and the connected MCP server") as step:
            self.go("Settings", "/settings", "Settings")
            card = page.locator(".card", has=page.get_by_role("heading", name="Tools", exact=True))
            self.expect(card).to_have_count(1, timeout=WAIT_MS)
            self.expect(card).to_contain_text("Calculator", timeout=WAIT_MS)
            self.expect(card).to_contain_text("fake", timeout=WAIT_MS)
            servers_box = page.get_by_label("MCP servers (JSON)")
            self.expect(servers_box).to_have_value(re.compile("mcpServers"), timeout=WAIT_MS)
            step.note = self.shoot("settings-tools")

    def models(self) -> None:
        page = self.page
        with self.step("Models page") as step:
            self.go("Models", "/models", "Models")
            catalog = self.studio.api("GET", "/api/state")["catalog"]
            self.expect(page.locator(".hw .tile")).to_have_count(4, timeout=WAIT_MS)
            self.expect(page.locator(".cat-grid .model-card")).to_have_count(len(catalog), timeout=WAIT_MS)
            row = page.locator(".lib-row", has_text=self.model["name"])
            self.expect(row).to_have_count(1, timeout=WAIT_MS)
            self.expect(row.locator(".pill.good")).to_have_text("loaded", timeout=WAIT_MS)
            step.note = f"{len(catalog)} catalog models, the stub is in the library; " + self.shoot("models")

    def engine(self) -> None:
        page = self.page
        with self.step("Engine page") as step:
            self.go("Engine", "/engine", "Engine")
            self.expect(page.locator(".status-hero .pill")).to_have_text("Ready", timeout=WAIT_MS)
            self.expect(page.locator(".status-hero")).to_contain_text(self.model["name"], timeout=WAIT_MS)
            self.expect(page.get_by_text(f"{self.studio.base}/v1", exact=True)).to_be_visible(timeout=WAIT_MS)
            command = page.locator(".card", has=page.get_by_role("heading", name="Command", exact=True)).locator(".cmd")
            self.expect(command).to_contain_text(Path(self.model["path"]).name, timeout=WAIT_MS)
            page.wait_for_function(
                "() => /listening/i.test(document.querySelector('.log')?.innerText || '')", timeout=WAIT_MS
            )
            step.note = "status, command and engine log; " + self.shoot("engine")

    def experiments(self) -> None:
        page, studio = self.page, self.studio
        with self.step("Experiments page: run the speed test once") as step:
            self.go("Experiments", "/lab", "Experiments")
            label = "e2e run"
            speed_card = page.locator(".card", has=page.get_by_role("heading", name="Speed test", exact=True))
            speed_card.locator("select.select").select_option("1")
            page.get_by_label("Label", exact=True).fill(label)
            page.get_by_role("button", name="Run the test").click()

            def finished() -> dict[str, Any] | None:
                bench = studio.api("GET", "/api/bench")
                current = bench.get("current") or {}
                if current.get("state") in ("failed", "cancelled"):
                    raise E2EError(f"the speed test {current.get('state')}: {current.get('error')}")
                return bench if current.get("state") == "done" and bench.get("results") else None

            bench = wait_until(finished, 180, "the speed test to finish")
            summary = bench["results"][0]["summary"]
            if not (summary.get("decode") or 0) > 0 or not (summary.get("prefill") or 0) > 0:
                raise E2EError(f"the speed test has no decode or prefill speed: {summary}")
            row = page.locator("tr.click", has_text=label)
            self.expect(row).to_be_visible(timeout=WAIT_MS)
            decode = number_in(row.locator("td.r").nth(0).inner_text())
            prefill = number_in(row.locator("td.r").nth(1).inner_text())
            if not decode or not prefill:
                raise E2EError(f"the result row shows decode {decode} and prefill {prefill}")
            self.expect(page.locator(".bench-bar").first).to_be_visible(timeout=WAIT_MS)
            step.note = f"decode {decode:g} tok/s, prefill {prefill:g} tok/s; " + self.shoot("experiments")

    def forge(self) -> None:
        """Forge: tools, two tiny synthetic quants, a hybrid with pruning and a check, built by the stand-in tools."""
        page, studio = self.page, self.studio
        tests = Path(__file__).resolve().parents[1]
        sys.path.insert(0, str(tests))
        from forge_synth import make_model  # the synthetic mixture-of-experts files of the unit tests

        folder = studio.work / "forge-files"
        folder.mkdir(exist_ok=True)
        high = make_model(folder / "high-Q8_0.gguf", seed=1, layers=4, types=[("Q8_0",) * 3] * 4)
        low = make_model(folder / "low-Q4_0.gguf", seed=2, layers=4, types=[("Q4_0",) * 3] * 4)
        (folder / "calibration.txt").write_text("calibration text\n" * 60, encoding="utf-8")
        (folder / "evaluation.txt").write_text("evaluation text\n" * 60, encoding="utf-8")
        tools = tests / "fake_tools"

        def next_step() -> None:
            page.get_by_role("button", name="Next", exact=True).click()

        with self.step("Forge page: tools and the two source quants") as step:
            self.go("Forge", "/forge", "Forge")
            page.locator("#fg-name").fill("e2e-forge")
            page.locator("#fg-tools-dir").fill(str(tools))
            page.get_by_role("button", name="Search again").click()
            self.expect(page.locator(".pill.good", has_text="Found")).to_have_count(4, timeout=WAIT_MS)
            next_step()
            for sid, path in (("B", high), ("K", low)):
                row = page.locator(f".fg-source[data-source='{sid}']")
                row.locator("input").fill(str(path))
                row.get_by_role("button", name="Read it").click()
                self.expect(row.locator(".fg-info")).to_contain_text("4 layers", timeout=WAIT_MS)
                self.expect(row.locator(".fg-info")).to_contain_text("8 experts, 2 used", timeout=WAIT_MS)
            step.note = "4 tools found, both files read (4 layers, 8 experts)"

        with self.step("Forge page: the calculator picks layers within a size budget") as step:
            next_step()
            limits = studio.api(
                "POST",
                "/api/forge/calculate",
                {"mode": "hybrid", "base": "B", "budget_gb": 1, "sources": {"B": str(high), "K": str(low)}},
            )
            selection = limits["selection"]
            middle = (selection["min_bytes"] + selection["max_bytes"]) / 2 / 1e9
            page.locator("#fg-budget").fill(f"{middle:.9f}")
            page.locator("#fg-ram").fill("64")
            page.locator("#fg-vram").fill("24")
            page.locator("#fg-calc").click()
            self.expect(page.locator(".fg-cell")).to_have_count(4, timeout=WAIT_MS)
            self.expect(page.locator(".fg-grid + .stack .fact").first).to_be_visible(timeout=WAIT_MS)
            mixed = {c.strip() for c in page.locator(".fg-cell b").all_inner_texts()}
            if mixed != {"B", "K"}:
                raise E2EError(f"a budget between the smallest and the largest result should mix both sources: {mixed}")
            first = page.locator(".fg-cell").first
            before = first.locator("b").inner_text()
            first.click()
            self.expect(first.locator("b")).not_to_have_text(before, timeout=WAIT_MS)  # a click takes the next source
            first.click()
            self.expect(first.locator("b")).to_have_text(before, timeout=WAIT_MS)
            step.note = f"layers {sorted(mixed)} within {middle * 1000:.3f} MB; " + self.shoot("forge-recipe")

        with self.step("Forge page: prune, check, build and publish") as step:
            next_step()
            page.locator("label.check", has_text="Prune experts after the build").click()
            page.locator("#fg-keep").fill("4")
            next_step()
            page.locator("#fg-corpus").fill(str(folder / "calibration.txt"))
            page.locator("#fg-eval").fill(str(folder / "evaluation.txt"))
            next_step()
            page.locator("#fg-check").click()
            self.expect(page.locator("#fg-check-box")).to_contain_text("Everything is in place", timeout=WAIT_MS)
            for title in ("Build the model", "Prune experts", "KL divergence"):
                self.expect(page.locator("#fg-check-box .fg-plan")).to_contain_text(title, timeout=WAIT_MS)
            with page.expect_download() as download:
                page.locator("#fg-save").click()
            shared = json.loads(Path(download.value.path()).read_text(encoding="utf-8"))
            if any(s["path"] for s in shared["sources"].values()) or str(folder) in json.dumps(shared):
                raise E2EError("the saved recipe still contains folders of this computer")
            page.locator("#fg-start").click()
            self.expect(page.locator("#fg-job-state")).to_have_text("done", timeout=120_000)
            self.expect(page.locator("#fg-result")).to_contain_text("KL divergence", timeout=WAIT_MS)
            self.expect(page.locator("#fg-result")).to_contain_text("0.0123", timeout=WAIT_MS)
            self.expect(page.locator(".fg-job-step.done")).to_have_count(6, timeout=WAIT_MS)
            self.expect(page.locator("#fg-log")).to_contain_text("Done", timeout=WAIT_MS)
            page.locator("#fg-publish").click()

            def published() -> bool:
                names = [p["name"] for p in studio.api("GET", "/api/state")["profiles"]]
                return "Forge: e2e-forge" in names

            wait_until(published, 15, "the profile 'Forge: e2e-forge' to be saved")
            built = studio.home / "models" / "forge" / "e2e-forge.gguf"
            if not built.is_file() or not (built.parent / "e2e-forge.recipe.json").is_file():
                raise E2EError(f"the result or its recipe is missing in {built.parent}")
            step.note = "hybrid + 4 of 8 experts + check, published; " + self.shoot("forge")

    # ---- 0.2: simple mode, quality, A/B/A, tuner, expert cache

    def restart_engine(self, extra_env: str, profile: str = "auto", *, start: bool = True) -> None:
        """Stop the engine, set the 'Extra environment' of the start settings, and (by default) start it again."""
        studio = self.studio
        studio.api("POST", "/api/engine/stop")
        studio.wait_engine("stopped", 30)
        wait_until(lambda: not pid_alive(studio.engine_pid), 15, f"engine process {studio.engine_pid} to exit")
        studio.api("POST", "/api/settings", {"launch": {"extra_env": extra_env}})
        if not start:
            return
        started = studio.api("POST", "/api/engine/start", {"model": self.model["id"], "profile": profile})
        if started["state"] not in ("starting", "ready"):
            raise E2EError(f"the engine did not start: {started}")
        ready = studio.wait_engine("ready", 90)
        studio.engine_pid = int(ready["pid"])
        studio.engine_port = urlparse(ready["base_url"]).port or 0

    def engine_modes(self) -> None:
        page = self.page
        with self.step("Engine page: simple view, Best for my machine, what each flag does") as step:
            self.go("Engine", "/engine", "Engine")
            self.expect(page.locator(".mode-switch button.on")).to_have_text("Simple", timeout=WAIT_MS)
            heading = page.get_by_role("heading", name="Best for my machine", exact=True)
            self.expect(heading).to_be_visible(timeout=WAIT_MS)
            self.expect(page.get_by_role("heading", name="Launch profiles", exact=True)).to_be_hidden(timeout=WAIT_MS)
            rows = page.locator(".explain-row")
            self.expect(rows.first).to_be_visible(timeout=WAIT_MS)
            known = page.locator(".explain-row:not(.unknown)")
            if known.count() < 3:
                raise E2EError(f"only {known.count()} flags of the command are explained")
            command = page.locator(".card", has=page.get_by_role("heading", name="Command", exact=True)).locator(".cmd")
            self.expect(command).to_contain_text(Path(self.model["path"]).name, timeout=WAIT_MS)
            tips = page.locator(".cmd .flag[title]").count()
            if not tips:
                raise E2EError("no flag in the command has an explanation as a tooltip")
            n_rows = rows.count()
            shot = self.shoot("engine-simple")
            page.locator(".mode-switch button", has_text="Advanced").click()
            self.expect(page.get_by_role("heading", name="Launch profiles", exact=True)).to_be_visible(timeout=WAIT_MS)
            self.expect(page.get_by_role("heading", name="Start settings", exact=True)).to_be_visible(timeout=WAIT_MS)
            page.locator(".mode-switch button", has_text="Simple").click()
            self.expect(page.get_by_role("heading", name="Launch profiles", exact=True)).to_be_hidden(timeout=WAIT_MS)
            step.note = f"{n_rows} explained parts of the command, {tips} tooltips; " + shot

    def quality(self) -> None:
        page, studio = self.page, self.studio
        with self.step("Experiments page: knowledge probe, KL divergence, speed with a score") as step:
            self.restart_engine(FAST_ENV)
            reference = Path(self.model["path"]).with_name("e2e-stub-Q8_0.gguf")
            reference.write_bytes(Path(self.model["path"]).read_bytes())
            ref = studio.api("POST", "/api/models/add", {"path": str(reference)})
            page.reload(wait_until="domcontentloaded")  # the page learns about the new file
            self.go("Experiments", "/lab", "Experiments")
            card = page.locator(".card", has=page.get_by_role("heading", name="Quality", exact=True))
            self.expect(card).to_be_visible(timeout=WAIT_MS)
            card.locator("select").nth(0).select_option(value=ref["id"])
            card.locator("select").nth(1).select_option(value=self.model["id"])
            card.get_by_role("button", name="Measure KL divergence").click()

            def kld_done() -> dict[str, Any] | None:
                state = studio.api("GET", "/api/quality")
                cur = state.get("current") or {}
                if cur.get("state") in ("failed", "cancelled"):
                    raise E2EError(f"the KL divergence run {cur.get('state')}: {cur.get('error')}")
                done = [r for r in state["results"] if r.get("kind") == "kld" and r.get("state") == "done"]
                return done[0] if done else None

            kld = wait_until(kld_done, 120, "the KL divergence run to finish")
            mean = kld["result"]["mean_kld"]
            if not mean > 0:
                raise E2EError(f"the mean KL divergence is {mean}")
            card.get_by_role("button", name="Ask the running model").click()

            def probe_done() -> dict[str, Any] | None:
                state = studio.api("GET", "/api/quality")
                cur = state.get("current") or {}
                if cur.get("state") in ("failed", "cancelled"):
                    raise E2EError(f"the knowledge probe {cur.get('state')}: {cur.get('error')}")
                done = [r for r in state["results"] if r.get("kind") == "probe" and r.get("state") == "done"]
                return done[0] if done else None

            probe = wait_until(probe_done, 180, "the knowledge probe to finish")["result"]
            if probe["n"] < 20 or not probe["ci_lo"] <= probe["accuracy"] <= probe["ci_hi"]:
                raise E2EError(f"the knowledge probe looks wrong: {probe['n']} questions, accuracy {probe['accuracy']}")
            self.expect(card.locator("tbody tr")).to_have_count(2, timeout=WAIT_MS)
            self.expect(card).to_contain_text("Knowledge probe", timeout=WAIT_MS)
            # the speed test with the knowledge questions: the score ends up next to the speed
            speed = page.locator(".card", has=page.get_by_role("heading", name="Speed test", exact=True))
            speed.locator("select.select").select_option("3")
            page.get_by_label("Label", exact=True).fill("with knowledge")
            speed.get_by_label("Also ask the knowledge questions").check()
            speed.get_by_role("button", name="Run the test").click()

            def speed_done() -> dict[str, Any] | None:
                bench = studio.api("GET", "/api/bench")
                cur = bench.get("current") or {}
                if cur.get("state") in ("failed", "cancelled"):
                    raise E2EError(f"the speed test {cur.get('state')}: {cur.get('error')}")
                hit = [r for r in bench["results"] if r.get("label") == "with knowledge" and r.get("state") == "done"]
                return hit[0] if hit else None

            run = wait_until(speed_done, 240, "the speed test with questions to finish")
            if not run.get("quality", {}).get("n"):
                raise E2EError("the speed test result has no knowledge score")
            ci = run["summary"]["decode_ci"]
            if ci["n"] != 3 or ci["lo"] is None or not ci["lo"] <= ci["median"] <= ci["hi"]:
                raise E2EError(f"three runs should give an interval around the median: {ci}")
            row = page.locator("tr.click", has_text="with knowledge")
            self.expect(row).to_contain_text("KLD", timeout=WAIT_MS)
            self.expect(row.locator(".ci").first).to_be_visible(timeout=WAIT_MS)
            step.note = (
                f"KL divergence {mean:.4f}, knowledge {probe['correct']} of {probe['n']}, decode "
                f"{ci['median']:g} tok/s [{ci['lo']:g} to {ci['hi']:g}]; " + self.shoot("experiments-quality")
            )

    def cache_page(self) -> None:
        page, studio = self.page, self.studio
        with self.step("Expert cache page: before and after the engine reports its cache") as step:
            self.go("Expert cache", "/cache", "Expert cache")
            self.expect(page.get_by_text("What this engine does not tell us yet")).to_be_visible(timeout=WAIT_MS)
            empty = self.shoot("cache-empty")
            self.restart_engine(FLE_ENV)
            base = studio.engine_state()["base_url"]
            body = {"messages": [{"role": "user", "content": "Count slowly."}], "max_tokens": 1400, "stream": False}
            status, raw = http("POST", base + "/v1/chat/completions", body=body, timeout=120)
            if status != 200:
                raise E2EError(f"the engine answered {status} to the long request: {raw[:200]!r}")
            snap = wait_until(
                lambda: (lambda d: d if d["available"]["layers"] and d["available"]["hot"] else None)(
                    studio.api("GET", "/api/cache")
                ),
                30,
                "the cache lines to arrive",
            )
            if len(snap["layers"]) != 2 or len(snap["overall"]["series"]) < 2:
                raise E2EError(
                    f"unexpected cache data: {len(snap['layers'])} layers, {len(snap['overall']['series'])} points"
                )
            self.expect(page.locator(".layer-cell")).to_have_count(2, timeout=WAIT_MS)
            self.expect(page.get_by_text("What this engine does not tell us yet")).to_have_count(0, timeout=WAIT_MS)
            page.locator(".layer-cell").first.click()
            self.expect(page.locator(".hot-cell").first).to_be_visible(timeout=WAIT_MS)
            self.expect(page.locator(".cache-top .fact").first).to_contain_text("%", timeout=WAIT_MS)
            step.note = (
                f"hit rate {snap['overall']['hit_rate']} % over {len(snap['overall']['series'])} points, "
                f"{len(snap['layers'])} layers; " + empty + ", " + self.shoot("cache")
            )

    def tuning(self) -> None:
        page, studio = self.page, self.studio
        with self.step("Engine page: Best for my machine finds settings and saves a profile") as step:
            self.restart_engine(FAST_ENV, start=False)
            self.go("Engine", "/engine", "Engine")
            best = page.locator(".card", has=page.get_by_role("heading", name="Best for my machine", exact=True))
            best.locator("summary").click()
            best.get_by_label("Batch size").uncheck()
            best.get_by_role("button", name="Find the best settings").click()

            def finished() -> dict[str, Any] | None:
                state = studio.api("GET", "/api/tuning")
                cur = state.get("current") or {}
                if cur.get("state") in ("failed", "cancelled"):
                    raise E2EError(f"the tuner {cur.get('state')}: {cur.get('error')}")
                return cur if cur.get("state") == "done" else None

            run = wait_until(finished, 240, "the tuner to finish")
            profile = run["profile"]["id"]
            saved = [p for p in studio.api("GET", "/api/profiles")["profiles"] if p["id"] == profile]
            if not saved or not saved[0].get("meta", {}).get("tuned"):
                raise E2EError(f"the tuned profile {profile} was not saved with its measurements")
            self.expect(best.get_by_role("button", name="Start with the tuned settings")).to_be_visible(timeout=WAIT_MS)
            shot = self.shoot("engine-tuned")
            tried = len(run["trials"])
            best.get_by_role("button", name="Start with the tuned settings").click()
            ready = studio.wait_engine("ready", 90)
            if (ready.get("plan") or {}).get("profile") != profile:
                started_with = (ready.get("plan") or {}).get("profile")
                raise E2EError(f"the engine did not start with the tuned profile: {started_with}")
            studio.engine_pid = int(ready["pid"])
            studio.engine_port = urlparse(ready["base_url"]).port or 0
            step.note = f"{tried} settings tried, profile {saved[0]['name']!r} saved and started; " + shot

        with self.step("Experiments page: A/B/A comparison of two profiles") as step:
            slow = dict(saved[0])
            args = list(slow["args"])
            if "-t" in args:
                args[args.index("-t") + 1] = "1"
            else:
                args += ["-t", "1"]
            studio.api(
                "POST",
                "/api/profiles",
                {
                    "name": "E2E one thread",
                    "engine": slow["engine"],
                    "binary": slow["binary"],
                    "args": args,
                    "env": slow["env"],
                    "notes": "slow on purpose",
                },
            )
            studio.api("POST", "/api/engine/stop")
            studio.wait_engine("stopped", 30)
            page.reload(wait_until="domcontentloaded")  # the page learns about the new profile
            self.go("Experiments", "/lab", "Experiments")
            card = page.locator(".card", has=page.get_by_role("heading", name="A/B/A comparison", exact=True))
            card.locator("select").nth(0).select_option(value="auto")
            card.locator("select").nth(1).select_option(label="E2E one thread")
            card.get_by_role("button", name="Compare").click()

            def aba_done() -> dict[str, Any] | None:
                cur = studio.api("GET", "/api/tuning").get("current") or {}
                if cur.get("state") in ("failed", "cancelled"):
                    raise E2EError(f"the comparison {cur.get('state')}: {cur.get('error')}")
                return cur if cur.get("state") == "done" and cur.get("kind") == "aba" else None

            aba = wait_until(aba_done, 240, "the A/B/A comparison to finish")
            verdict = aba["result"]["decode"]["verdict"]
            if verdict not in ("better", "worse", "within noise"):
                raise E2EError(f"unexpected verdict {verdict!r}")
            self.expect(card.locator(".verdict").first).to_be_visible(timeout=WAIT_MS)
            self.expect(card.locator("tbody tr")).to_have_count(2, timeout=WAIT_MS)
            pct = aba["result"]["decode"]["pct"]
            step.note = f"B against A (generation): {verdict}, {pct} %; " + self.shoot("experiments-aba")
            self.restart_engine("")  # the other steps and the final checks expect the automatic plan, running

    def settings(self) -> None:
        page, studio = self.page, self.studio

        def theme(label: str, attribute: str | None) -> None:
            radio = page.get_by_role("radio", name=label, exact=True)
            radio.click()
            self.expect(radio).to_have_attribute("aria-checked", "true", timeout=WAIT_MS)
            if attribute is None:
                page.wait_for_function("() => !document.documentElement.hasAttribute('data-theme')", timeout=WAIT_MS)
            else:
                self.expect(page.locator("html")).to_have_attribute("data-theme", attribute, timeout=WAIT_MS)

        with self.step("Settings page: theme switch and system setting") as step:
            self.go("Settings", "/settings", "Settings")
            about = page.locator(".card", has=page.get_by_role("heading", name="About", exact=True))
            self.expect(about).to_contain_text(studio.version, timeout=WAIT_MS)
            page.emulate_media(color_scheme="light")
            wait_for_look(page, "light", "like the system, light")
            page.emulate_media(color_scheme="dark")
            wait_for_look(page, "dark", "like the system, dark")
            theme("Light", "light")  # an explicit choice beats the dark system setting
            wait_for_look(page, "light", "Light chosen under a dark system setting")
            page.emulate_media(color_scheme="light")
            theme("Dark", "dark")  # and the other way round
            wait_for_look(page, "dark", "Dark chosen under a light system setting")
            saved = studio.api("GET", "/api/state")["settings"]["theme"]
            if saved != "dark":
                raise E2EError(f"the theme choice was not saved (settings say {saved!r})")
            page.reload(wait_until="domcontentloaded")  # the choice survives a reload
            self.expect(page.locator("html")).to_have_attribute("data-theme", "dark", timeout=WAIT_MS)
            wait_for_look(page, "dark", "Dark after a reload")
            theme("Like the system", None)
            wait_for_look(page, "light", "like the system, light again")
            saved = studio.api("GET", "/api/state")["settings"]["theme"]
            if saved != "system":
                raise E2EError(f"'Like the system' was not saved (settings say {saved!r})")
            step.note = "Light and Dark override the system, saved, kept after a reload; " + self.shoot("settings")

    def updates(self) -> None:
        page, studio = self.page, self.studio
        gh = studio.github
        assert gh is not None
        with self.step("Updates: check now, notice, notes as text, Later, switch off") as step:
            self.go("Settings", "/settings", "Settings")
            card = page.locator(".card", has=page.get_by_role("heading", name="Updates", exact=True))
            self.expect(card).to_be_visible(timeout=WAIT_MS)
            page.locator("#update-check-now").click()
            notice = page.locator("#update-card")
            self.expect(notice).to_contain_text(f"FLE Studio {UPDATE_TAG[1:]} is available", timeout=WAIT_MS)
            self.expect(card).to_contain_text("is available", timeout=WAIT_MS)
            notice.click()
            dialog = page.locator(".upd-modal")
            self.expect(dialog).to_be_visible(timeout=WAIT_MS)
            notes = dialog.locator(".upd-notes")
            self.expect(notes.locator("h4")).to_have_text("What is new", timeout=WAIT_MS)
            self.expect(notes.locator("li")).to_have_count(2, timeout=WAIT_MS)
            self.expect(notes.locator("code")).to_have_text("code", timeout=WAIT_MS)
            self.expect(notes).to_contain_text("<script>window.__pwned = 1</script>", timeout=WAIT_MS)
            if notes.locator("script, img").count() or page.evaluate("() => window.__pwned") is not None:
                raise E2EError("HTML from the release notes became part of the page")
            link = dialog.get_by_role("link", name="Release page on GitHub")
            want = f"https://github.com/sirfyyn/fle-studio/releases/tag/{UPDATE_TAG}"
            self.expect(link).to_have_attribute("href", want, timeout=WAIT_MS)
            # This run is not a uv tool install: no one-click button, the command to copy instead.
            self.expect(dialog.get_by_role("button", name="Update now")).to_have_count(0, timeout=WAIT_MS)
            # (an editable install from a source folder gets "git pull", a pip install the pip command)
            command = dialog.locator(".upd-cmd code")
            self.expect(command).to_contain_text(re.compile(r"git pull|pip install"), timeout=WAIT_MS)
            shots = self.shoot("update")
            dialog.get_by_role("button", name="Later").click()
            self.expect(dialog).to_have_count(0, timeout=WAIT_MS)
            self.expect(notice).to_be_hidden(timeout=WAIT_MS)
            page.reload(wait_until="domcontentloaded")
            self.expect(page.locator(".page-head h2")).to_have_text("Settings", timeout=WAIT_MS)
            self.expect(page.locator("#update-card")).to_be_hidden(timeout=WAIT_MS)
            seen = len(gh.requests)
            card = page.locator(".card", has=page.get_by_role("heading", name="Updates", exact=True))
            card.locator("label", has_text="Check for updates").locator("i").click()
            self.expect(page.locator("#update-check-now")).to_be_disabled(timeout=WAIT_MS)
            if studio.api("GET", "/api/state")["settings"]["updates"]["check"] is not False:
                raise E2EError("switching the update check off was not saved")
            if len(gh.requests) != seen:
                raise E2EError("FLE Studio asked GitHub after the update check was switched off")
            if any({"authorization", "cookie"} & set(headers) for _, headers in gh.requests):
                raise E2EError("the update check sent identifying headers")
            step.note = f"{len(gh.requests)} request(s) to the fake GitHub, no identifiers; " + shots

    def remote(self) -> None:
        """A second Studio is the host (TLS, its own engine); this page adds it, works on it and comes back."""
        page, studio = self.page, self.studio
        host = StudioProcess(studio.work / "host", self.out, name="host", extra=("--host", "127.0.0.1", "--tls"))
        studio.sidekick = host
        host.work.mkdir(parents=True, exist_ok=True)
        label = page.locator("#host-name")

        def pairing_code() -> str:
            return str(host.api("POST", "/api/remote/code", {})["pretty"])

        with self.step("Remote: start a host Studio (TLS) with its own engine") as step:
            host.start()
            stub = host.work / "elsewhere" / "e2e-stub-Q4_K_M.gguf"
            stub.parent.mkdir(parents=True, exist_ok=True)
            stub.write_bytes(gguf_stub())
            host.api("POST", "/api/engines/check", {"path": str(FAKE_ENGINE)})
            host.api("POST", "/api/settings", {"engine": {"choice": "custom", "custom_path": str(FAKE_ENGINE)}})
            model = host.api("POST", "/api/models/add", {"path": str(stub)})
            host.api("POST", "/api/engine/start", {"model": model["id"], "profile": "auto"})
            ready = host.wait_engine("ready", 90)
            host.engine_pid = int(ready["pid"])
            host.engine_port = urlparse(ready["base_url"]).port or 0
            info = host.api("GET", "/api/remote")
            if not info["listening"]["tls"] or len(info["fingerprint"]) != 64:
                raise E2EError(f"the host does not show a certificate fingerprint: {info}")
            step.note = f"host on {host.base}, fingerprint {info['fingerprint_pretty'][:23]}…"

        with self.step("Remote: add the host with its pairing code and switch to it") as step:
            self.expect(label).to_have_text("This machine", timeout=WAIT_MS)
            page.locator("#host-switch").click()
            page.locator(".menu").get_by_role("menuitem", name="Add host…").click()
            page.locator("#host-address").fill(f"127.0.0.1:{host.port}")
            page.locator("#host-name-in").fill("GPU box (e2e)")
            page.locator("#host-code").fill(pairing_code())
            shots = self.shoot("host-add")
            page.locator("#host-add-go").click()
            self.expect(label).to_have_text("GPU box (e2e)", timeout=WAIT_MS)
            self.expect(page.locator("body")).to_have_class(re.compile(r"host-remote"), timeout=WAIT_MS)
            self.expect(page).to_have_title("GPU box (e2e) · FLE Studio", timeout=WAIT_MS)
            data_dir = page.evaluate("async () => (await (await fetch('/api/state')).json()).data_dir")
            if data_dir != str(host.home) and Path(data_dir).resolve() != host.home.resolve():
                raise E2EError(f"the page still talks to this machine: data folder {data_dir}")
            self.expect(page.locator("#engine-card .pill")).to_have_text("Ready", timeout=WAIT_MS)
            step.note = f"the page shows the host's data folder, the badge and the colour; {shots}"

        with self.step("Remote: chat on the host, the answer streams, the chat stays on the host") as step:
            before = len(studio.api("GET", "/api/chats")["chats"])
            page.locator("textarea[aria-label='Message']").fill("Hello host")
            page.locator("button.send").click()
            self.expect(page.locator(".msg.assistant .stats")).to_be_visible(timeout=60_000)
            reply = page.locator(".msg.assistant .body > .md")
            self.expect(reply).to_contain_text('Fake reply to "Hello host"', timeout=WAIT_MS)
            page.wait_for_function("() => /^\\/c\\/[0-9a-f]{16}$/.test(location.pathname)", timeout=WAIT_MS)
            chat_id = page.evaluate("() => location.pathname.split('/').pop()")
            stored = wait_until(lambda: self._chat_on(host, chat_id), 15, "the chat to be saved on the host")
            if studio.api("GET", "/api/chats")["chats"] and len(studio.api("GET", "/api/chats")["chats"]) != before:
                raise E2EError("a chat was saved on this machine while working on the host")
            step.note = f"{len(stored['messages'])} messages stored on the host only; " + self.shoot("host-chat")

        host_page = page.context.new_page()
        self.page = host_page
        try:
            with self.step("Remote: the host's own Settings lists the paired computer and makes codes") as step:
                host_page.goto(host.base + "/settings", wait_until="domcontentloaded")
                heading = host_page.get_by_role("heading", name="Remote access", exact=True)
                card = host_page.locator(".card", has=heading)
                self.expect(card).to_contain_text("Certificate fingerprint", timeout=WAIT_MS)
                self.expect(card.locator("#host-fp")).to_have_text(
                    re.compile(r"^([0-9A-F]{2}:){31}[0-9A-F]{2}$"), timeout=WAIT_MS
                )
                self.expect(card.locator("[data-client]")).to_have_count(1, timeout=WAIT_MS)
                card.locator("#pair-new").click()
                code_re = re.compile(r"^[A-Z2-9]{4}-[A-Z2-9]{4}$")
                self.expect(card.locator("#pair-code")).to_have_text(code_re, timeout=WAIT_MS)
                step.note = self.shoot("host-access")

            with self.step("Remote: revoking on the host is noticed on the client, pairing again fixes it") as step:
                self.watch.tolerated = ["status of 424"]  # the host refuses a revoked client: expected, and shown
                card.locator("[data-client]").get_by_role("button", name="Revoke").click()
                host_page.locator(".modal").get_by_role("button", name="Revoke").click()
                self.expect(card.locator("[data-client]")).to_have_count(0, timeout=WAIT_MS)
                self.page = page
                banner = page.locator("#host-banner")
                page.evaluate("() => fetch('/api/state')")  # the next request the page makes finds out
                self.expect(banner).to_contain_text("no longer accepts this computer", timeout=WAIT_MS)
                shots = self.shoot("host-revoked")
                banner.get_by_role("button", name="Pair again…").click()
                page.locator("#repair-code").fill(pairing_code())
                page.locator(".host-modal").get_by_role("button", name="Pair again").click()
                self.expect(label).to_have_text("GPU box (e2e)", timeout=WAIT_MS)
                self.expect(page.locator("#host-banner")).to_have_count(0, timeout=WAIT_MS)
                self.expect(page.locator("#engine-card .pill")).to_have_text("Ready", timeout=WAIT_MS)
                step.note = f"banner and re-pairing work; {shots}"
        finally:
            self.page = page
            self.watch.tolerated = []
            host_page.close()

        with self.step("Remote: the host goes away, the page says so and goes back to this machine") as step:
            self.watch.tolerated = ["status of 502", "status of 424"]
            code = host.terminate()
            page.evaluate("() => fetch('/api/state')")
            banner = page.locator("#host-banner")
            self.expect(banner).to_contain_text("is not reachable", timeout=WAIT_MS)
            shots = self.shoot("host-away")
            banner.get_by_role("button", name="Use this machine").click()
            self.expect(label).to_have_text("This machine", timeout=WAIT_MS)
            self.expect(page.locator("body")).not_to_have_class(re.compile(r"host-remote"), timeout=WAIT_MS)
            self.expect(page).to_have_title("FLE Studio", timeout=WAIT_MS)
            local_dir = page.evaluate("async () => (await (await fetch('/api/state')).json()).data_dir")
            if Path(local_dir).resolve() != studio.home.resolve():
                raise E2EError(f"back on this machine the page shows {local_dir}")
            step.note = f"host ended with code {code}; {shots}"
        self.watch.tolerated = []

    @staticmethod
    def _chat_on(host: StudioProcess, chat_id: str) -> dict[str, Any] | None:
        chat = host.api("GET", f"/api/chats/{chat_id}")
        messages = chat.get("messages") or []
        return chat if len(messages) == 2 and messages[1].get("content") else None


# ------------------------------------------------------------------------------------------------ the run


def run_browser(
    args: argparse.Namespace, report: Report, studio: StudioProcess, out: Path, model: dict[str, Any]
) -> None:
    try:
        from playwright.sync_api import expect, sync_playwright  # imported here to give a clear message when missing
    except ImportError:
        raise E2EError(
            "Playwright is not installed: python -m pip install playwright && python -m playwright install chromium"
        ) from None

    with sync_playwright() as pw:
        with report.step("Launch the browser") as step:
            browser = pw.chromium.launch(headless=not args.headed, channel=args.channel)
            step.note = (
                f"{args.channel or 'Playwright Chromium'} {browser.version}, {'headed' if args.headed else 'headless'}"
            )
        page = None
        watch = None
        try:
            context = browser.new_context(
                viewport=VIEWPORT,
                device_scale_factor=1,
                locale="en-US",
                color_scheme="light",
                reduced_motion="reduce",
                ignore_https_errors=True,
            )
            context.set_default_timeout(WAIT_MS)
            page = context.new_page()
            watch = PageWatch(page)
            session = Session(report, studio, page, watch, out, expect, model)
            session.open_app()
            session.chat()
            session.live_panel()
            session.reload_chat()
            session.attachments()
            session.versions()
            session.export_import()
            session.search()
            session.tools()
            session.mcp_remote()
            session.settings_tools()
            session.models()
            session.engine()
            session.experiments()
            session.engine_modes()
            session.quality()
            session.cache_page()
            session.tuning()
            session.forge()
            session.settings()
            session.updates()
            session.remote()
            with report.step("No uncaught page errors or console errors in the whole run") as step:
                watch.check("the whole run")
                step.note = f"{len(watch.lines)} console message(s) in browser-console.log"
        except BaseException:
            if page is not None:
                with contextlib.suppress(Exception):
                    page.screenshot(path=str(out / "failure.png"))
            raise
        finally:
            if watch is not None:
                (out / "browser-console.log").write_text("\n".join(watch.lines) + "\n", encoding="utf-8")
            with contextlib.suppress(Exception):
                browser.close()


def run(args: argparse.Namespace, report: Report, studio: StudioProcess, out: Path, work: Path) -> None:
    studio.github = FakeGitHub([release(UPDATE_TAG, UPDATE_NOTES)]).__enter__()
    studio.env["FLE_STUDIO_GITHUB_API"] = studio.github.url  # the update check never reaches the real GitHub
    with report.step("Start FLE Studio on a free port") as step:
        studio.start()
        step.note = f"v{studio.version} at {studio.base}, data in a temporary folder"

    stub = work / "elsewhere" / "e2e-stub-Q4_K_M.gguf"
    stub.parent.mkdir(parents=True)
    stub.write_bytes(gguf_stub())
    with report.step("Use the stand-in engine and add the stub model") as step:
        check = studio.api("POST", "/api/engines/check", {"path": str(FAKE_ENGINE)})
        if not check.get("ok"):
            raise E2EError(f"Studio does not accept the stand-in engine: {check.get('problem')}")
        engine_setting = {"choice": "custom", "custom_path": str(FAKE_ENGINE)}
        settings = studio.api("POST", "/api/settings", {"engine": engine_setting})
        if settings["engine"]["choice"] != "custom":
            raise E2EError(f"the engine choice was not saved: {settings['engine']}")
        model = studio.api("POST", "/api/models/add", {"path": str(stub)})
        if not model.get("complete") or model.get("kind") != "model" or model.get("error"):
            raise E2EError(f"Studio cannot use the stub model: {model.get('error') or model}")
        arch = (model.get("info") or {}).get("arch")
        step.note = f"engine: {check.get('flavor')} flavour, model: {model['name']} ({arch})"

    with report.step("Start the engine with the automatic plan") as step:
        started = studio.api("POST", "/api/engine/start", {"model": model["id"], "profile": "auto"})
        if started["state"] not in ("starting", "ready"):
            raise E2EError(f"the engine did not start: {started}")
        ready = studio.wait_engine("ready", 90)
        n_ctx = ready.get("n_ctx") or 0
        if n_ctx < 8192:
            raise E2EError(f"the engine runs with a context of {n_ctx} tokens; the speed test needs 8192 or more")
        studio.engine_pid = int(ready["pid"])
        studio.engine_port = urlparse(ready["base_url"]).port or 0
        step.note = f"ready, {n_ctx} tokens of context, engine process {studio.engine_pid} on port {studio.engine_port}"

    run_browser(args, report, studio, out, model)

    with report.step("Stop the engine") as step:
        studio.api("POST", "/api/engine/stop")
        studio.wait_engine("stopped", 30)
        wait_until(lambda: not pid_alive(studio.engine_pid), 15, f"engine process {studio.engine_pid} to exit")
        wait_until(lambda: not port_open(studio.engine_port), 15, f"engine port {studio.engine_port} to close")
        step.note = "state stopped, process gone, port closed"

    with report.step("Stop FLE Studio") as step:
        code = studio.terminate()
        if studio.killed:
            raise E2EError("Studio did not quit within 20 s of the stop signal and had to be killed")
        if sys.platform != "win32" and code != 0:  # on Windows terminate() is a hard kill, so any code is fine
            raise E2EError(f"Studio quit with exit code {code}; see {studio.log_path}")
        # A hard-killed Windows process can take a moment to release its sockets, so give the checks a few seconds.
        wait_until(lambda: not port_open(studio.port), 5, f"port {studio.port} to close after Studio quit")
        try:
            wait_until(lambda: not pid_alive(studio.engine_pid) and not port_open(studio.engine_port), 5, "engine gone")
        except E2EError:
            raise E2EError(
                f"an engine process is left behind (process {studio.engine_pid}, port {studio.engine_port})"
            ) from None
        log = studio.log_path.read_text(encoding="utf-8", errors="replace")
        if "Traceback (most recent call last)" in log:
            raise E2EError(f"Studio logged a traceback; see {studio.log_path}")
        step.note = f"exit code {code}, no traceback in studio.log"


def collect_logs(work: Path, out: Path) -> None:
    """Keep the engine logs next to the screenshots: they are what one needs when a run fails."""
    logs = work / "home" / "logs"
    if logs.is_dir():
        shutil.copytree(logs, out / "engine-logs", dirs_exist_ok=True)


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Browser end-to-end test of FLE Studio against the stand-in engine.")
    parser.add_argument("--out", default="e2e-out", help="folder for the screenshots and logs (default: e2e-out)")
    parser.add_argument(
        "--channel", help="use an installed browser instead of Playwright's Chromium, e.g. chrome or msedge"
    )
    parser.add_argument("--headed", action="store_true", help="show the browser window (default: headless)")
    return parser.parse_args(argv)


def interrupt(*_: Any) -> None:
    raise KeyboardInterrupt


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    for stream in (sys.stdout, sys.stderr):
        with contextlib.suppress(AttributeError, ValueError):
            stream.reconfigure(errors="backslashreplace")  # type: ignore[union-attr]
    with contextlib.suppress(ValueError, OSError):  # a CI cancel (SIGTERM) should still clean up
        signal.signal(signal.SIGTERM, interrupt)
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    for stale in ("failure.png", "studio.log", "host.log", "browser-console.log"):  # what an earlier run left behind
        (out / stale).unlink(missing_ok=True)
    shutil.rmtree(out / "engine-logs", ignore_errors=True)
    work = Path(tempfile.mkdtemp(prefix="fle-studio-e2e-"))
    report = Report()
    studio = StudioProcess(work, out)
    started = time.monotonic()
    status = 0
    try:
        run(args, report, studio, out, work)
    except Exception as exc:
        status = 1
        if not isinstance(exc, E2EError):
            traceback.print_exc()
        if "\n" in str(exc).strip():  # the rest of a long message (call log, engine log tail)
            print(f"\n{exc}\n", file=sys.stderr)
        say(f"E2E FAILED at step {report.number}: {first_line(exc)}")
    finally:
        studio.cleanup()
        with contextlib.suppress(OSError):
            collect_logs(work, out)
        shutil.rmtree(work, ignore_errors=True)
    if status == 0:
        say(
            f"E2E PASSED: {report.number} steps, {len(report.screenshots)} screenshots in {out} "
            f"({time.monotonic() - started:.0f} s)"
        )
    return status


if __name__ == "__main__":
    sys.exit(main())

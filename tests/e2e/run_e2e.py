"""Browser end-to-end test of FLE Studio.

It starts a real FLE Studio (`python -m fle_studio`, the installed package) in a temporary data folder, points it at
`tests/fake_llama_server.py` as "my own build" of llama-server, adds a tiny GGUF file as the model and starts the engine
with the automatic plan. Then a headless Chromium (Playwright) does what a person would do:

* open the app, send one chat message and wait for the streamed answer, with its reasoning block and speed line;
* check that the live panel shows a decode speed;
* visit Models, Engine, Experiments (one run of the speed test) and Settings (the theme switch), and save a
  full-page screenshot of every page in light and in dark mode;
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

MODEL_NAME = "FLE Studio E2E stub"
# Shorter than the 40 characters the stand-in engine echoes back in its answer.
QUESTION = "Hello from the browser end-to-end test"
VIEWPORT = {"width": 1440, "height": 900}  # wide enough for the live panel to be open from the start
WAIT_MS = 30_000  # how long the browser waits for one thing to show up
MAX_SHOT_HEIGHT = 6000
PNG_MAGIC = b"\x89PNG\r\n\x1a\n"
# Makes the stand-in engine fast and gives the settings a sweet spot, so the tuner and the A/B/A test have something to
# find within a few seconds. Passed to the engine through the start settings ("Extra environment"), not to Studio.
FAST_ENV = "\n".join(["FAKE_SPEED_MODEL=1", "FAKE_TOK_PER_SEC=3000", "FAKE_PROMPT_TOK_PER_SEC=200000",
                      "FAKE_JITTER=0", "FAKE_RUN_NOISE=0.01", "FAKE_LOAD_SECONDS=0.3"])
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


_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))  # never through a proxy: it is all loopback


def http(method: str, url: str, *, body: Any = None, headers: dict[str, str] | None = None,
         timeout: float = 30) -> tuple[int, bytes]:
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
    tensors = [("token_embd.weight", (64, 100), 12800), ("blk.0.ffn_up_exps.weight", (64, 64, 4), 32768),
               ("blk.0.attn_q.weight", (64, 64), 8192)]  # (name, dimensions, bytes as F16)
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

    def __init__(self, work: Path, out: Path) -> None:
        self.work, self.home = work, work / "home"
        self.log_path = out / "studio.log"
        self.proc: subprocess.Popen[bytes] | None = None
        self.port = 0
        self.base = ""
        self.token = ""
        self.version = ""
        self.engine_pid = 0
        self.engine_port = 0
        self.killed = False
        self.env = {k: v for k, v in os.environ.items()
                    if not k.upper().startswith(SCRUBBED_PREFIXES) and k.upper() not in SCRUBBED_NAMES}
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
            self.base = f"http://127.0.0.1:{self.port}"
            cmd = [sys.executable, "-m", "fle_studio", "--home", str(self.home), "--port", str(self.port),
                   "--no-browser"]
            with self.log_path.open("ab") as log:
                log.write(f"$ {' '.join(cmd)}\n".encode())
                log.flush()
                self.proc = subprocess.Popen(cmd, cwd=self.work, env=self.env, stdin=subprocess.DEVNULL, stdout=log,
                                             stderr=subprocess.STDOUT)
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
                raise E2EError(f"the engine stopped with an error: {state.get('error')} "
                               f"(exit code {state.get('exit_code')})\n{tail}")
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
        if self.proc is not None and self.proc.poll() is None:
            with contextlib.suppress(Exception):
                self.api("POST", "/api/engine/stop", timeout=10)
                wait_until(lambda: self.engine_state()["state"] in ("stopped", "crashed"), 15, "the engine to stop")
            self.terminate()
        if self.engine_pid and pid_alive(self.engine_pid):
            say(f"note: the engine process {self.engine_pid} was still running; killing it")
            kill_pid(self.engine_pid)


# ------------------------------------------------------------------------------------------------ browser helpers


class PageWatch:
    """Collects uncaught page errors and console errors; `check` raises if there were any since the last call."""

    def __init__(self, page: Page) -> None:
        self.errors: list[str] = []
        self.lines: list[str] = []
        page.on("console", self._console)
        page.on("pageerror", self._page_error)

    def _console(self, msg: Any) -> None:
        loc = msg.location or {}
        where = f" ({loc.get('url')}:{loc.get('lineNumber')})" if loc.get("url") else ""
        self.lines.append(f"{msg.type}: {msg.text}{where}")
        if msg.type == "error":
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

    def __init__(self, report: Report, studio: StudioProcess, page: Page, watch: PageWatch, out: Path, expect: Any,
                 model: dict[str, Any]) -> None:
        self.report, self.studio, self.page, self.watch = report, studio, page, watch
        self.out, self.expect, self.model = out, expect, model
        self.chat_id = ""

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
            self.expect(self.page.locator("#nav a")).to_have_count(6, timeout=WAIT_MS)
            self.expect(self.page.locator("#engine-card .pill")).to_have_text("Ready", timeout=WAIT_MS)
            self.expect(self.page.locator(".welcome h2")).to_contain_text(f"What should {self.model['name']} work on?",
                                                                          timeout=WAIT_MS)
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
                raise E2EError(f"the saved chat looks wrong: roles {roles}, reasoning kept: "
                               f"{bool(chat['messages'][1].get('reasoning'))}")
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
                }""", timeout=WAIT_MS)
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
                "tok/s", timeout=WAIT_MS)
            step.note = self.shoot("chat")

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
            page.wait_for_function("() => /listening/i.test(document.querySelector('.log')?.innerText || '')",
                                   timeout=WAIT_MS)
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
            step.note = (f"KL divergence {mean:.4f}, knowledge {probe['correct']} of {probe['n']}, decode "
                         f"{ci['median']:g} tok/s [{ci['lo']:g} to {ci['hi']:g}]; " + self.shoot("experiments-quality"))

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
            snap = wait_until(lambda: (lambda d: d if d["available"]["layers"] and d["available"]["hot"] else None)(
                studio.api("GET", "/api/cache")), 30, "the cache lines to arrive")
            if len(snap["layers"]) != 2 or len(snap["overall"]["series"]) < 2:
                raise E2EError(f"unexpected cache data: {len(snap['layers'])} layers, "
                               f"{len(snap['overall']['series'])} points")
            self.expect(page.locator(".layer-cell")).to_have_count(2, timeout=WAIT_MS)
            self.expect(page.get_by_text("What this engine does not tell us yet")).to_have_count(0, timeout=WAIT_MS)
            page.locator(".layer-cell").first.click()
            self.expect(page.locator(".hot-cell").first).to_be_visible(timeout=WAIT_MS)
            self.expect(page.locator(".cache-top .fact").first).to_contain_text("%", timeout=WAIT_MS)
            step.note = (f"hit rate {snap['overall']['hit_rate']} % over {len(snap['overall']['series'])} points, "
                         f"{len(snap['layers'])} layers; " + empty + ", " + self.shoot("cache"))

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
            studio.api("POST", "/api/profiles", {"name": "E2E one thread", "engine": slow["engine"],
                                                 "binary": slow["binary"],
                                                 "args": args, "env": slow["env"], "notes": "slow on purpose"})
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
            self.expect(page.locator(".card", has_text="About")).to_contain_text(studio.version, timeout=WAIT_MS)
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


# ------------------------------------------------------------------------------------------------ the run


def run_browser(args: argparse.Namespace, report: Report, studio: StudioProcess, out: Path,
                model: dict[str, Any]) -> None:
    try:
        from playwright.sync_api import expect, sync_playwright  # imported here to give a clear message when missing
    except ImportError:
        raise E2EError("Playwright is not installed: python -m pip install playwright && "
                       "python -m playwright install chromium") from None

    with sync_playwright() as pw:
        with report.step("Launch the browser") as step:
            browser = pw.chromium.launch(headless=not args.headed, channel=args.channel)
            step.note = (f"{args.channel or 'Playwright Chromium'} {browser.version}, "
                         f"{'headed' if args.headed else 'headless'}")
        page = None
        watch = None
        try:
            context = browser.new_context(viewport=VIEWPORT, device_scale_factor=1, locale="en-US",
                                          color_scheme="light", reduced_motion="reduce")
            context.set_default_timeout(WAIT_MS)
            page = context.new_page()
            watch = PageWatch(page)
            session = Session(report, studio, page, watch, out, expect, model)
            session.open_app()
            session.chat()
            session.live_panel()
            session.reload_chat()
            session.models()
            session.engine()
            session.experiments()
            session.engine_modes()
            session.quality()
            session.cache_page()
            session.tuning()
            session.settings()
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
    parser.add_argument("--channel", help="use an installed browser instead of Playwright's Chromium, "
                        "e.g. chrome or msedge")
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
    for stale in ("failure.png", "studio.log", "browser-console.log"):  # what an earlier run left behind
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
        say(f"E2E PASSED: {report.number} steps, {len(report.screenshots)} screenshots in {out} "
            f"({time.monotonic() - started:.0f} s)")
    return status


if __name__ == "__main__":
    sys.exit(main())

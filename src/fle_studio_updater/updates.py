"""Update check and one-click update.

The check asks GitHub (api.github.com, no login, nothing but the usual request headers) for the latest release of
FLE Studio, and for the latest FLE engine when one is installed. It runs at most once a day, can be turned off in the
Settings, keeps its answer in <data>/update-check.json and stays silent when there is no network.

The one-click update works only for a FLE Studio that was installed by the installer (a uv tool from the GitHub
release) and does not run as a service. Every other kind of install gets the command that fits it, to copy.
Nothing from the network is ever rendered as HTML or used as a link: the notes are plain text, the release link is
built from a tag that passed a strict pattern.
"""

from __future__ import annotations

import contextlib
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
from collections.abc import Callable
from importlib import resources
from pathlib import Path
from typing import Any
from urllib.error import HTTPError

from fle_studio import __version__, net, paths, selfupdate
from fle_studio.versions import SAFE_TAG as _SAFE_TAG
from fle_studio.versions import Version, is_newer, parse_version  # noqa: F401 - the tests use them from here

REPO = "sirfyyn/fle-studio"
REPO_URL = f"https://github.com/{REPO}"
ENGINE_REPO = "sirfyyn/FastLocalExperts"
CHECK_EVERY = 24 * 3600  # seconds between two automatic checks
RETRY_OFFLINE = 3600  # when the last try found no network at all
MANUAL_GAP = 5  # seconds between two presses of "Check now"
NOTES_MAX = 4000
UV_PYTHON = "3.12"  # what the installers use


def api_base() -> str:
    """GitHub's API address. FLE_STUDIO_GITHUB_API points it elsewhere, which the tests use."""
    return os.environ.get("FLE_STUDIO_GITHUB_API", "https://api.github.com").rstrip("/")


class UpdateError(Exception):
    """An update request that cannot be done; the message says why."""

    def __init__(self, message: str, status: int = 409) -> None:
        super().__init__(message)
        self.status = status


# ------------------------------------------------------------------------------------------------ releases


def _clean_notes(body: Any) -> str:
    text = str(body or "").replace("\r\n", "\n").replace("\r", "\n")
    text = "".join(ch for ch in text if ch == "\n" or ch == "\t" or ord(ch) >= 32).strip()
    if len(text) > NOTES_MAX:
        cut = text[:NOTES_MAX]
        text = cut[: cut.rfind("\n\n")] if "\n\n" in cut[NOTES_MAX // 2 :] else cut
        text = text.rstrip() + "\n…"
    return text


def pick_release(releases: Any, include_pre: bool) -> dict[str, Any] | None:
    """The newest usable release out of one release object or a list of them.

    Drafts are skipped. Pre-releases are skipped unless `include_pre`. A release whose tag is not a version is
    skipped. Returns {"version", "tag", "url", "notes", "published", "prerelease"}; the url is built from the tag.
    """
    items = releases if isinstance(releases, list) else [releases]
    best: tuple[Version, dict[str, Any]] | None = None
    for rel in items:
        if not isinstance(rel, dict) or rel.get("draft"):
            continue
        tag = str(rel.get("tag_name") or "")
        version = parse_version(tag)
        if version is None or not _SAFE_TAG.match(tag):
            continue
        pre = bool(rel.get("prerelease")) or version.prerelease
        if pre and not include_pre:
            continue
        if best is None or version > best[0]:
            best = (
                version,
                {
                    "version": version.text,
                    "tag": tag,
                    "url": f"{REPO_URL}/releases/tag/{tag}",
                    "notes": _clean_notes(rel.get("body")),
                    "published": str(rel.get("published_at") or "")[:25],
                    "prerelease": pre,
                },
            )
    return best[1] if best else None


def _get(path: str) -> Any:
    data, _ = net.get_json(f"{api_base()}{path}", timeout=15, headers={"Accept": "application/vnd.github+json"})
    return data


def fetch_app_release(include_pre: bool) -> dict[str, Any] | None:
    if include_pre:
        return pick_release(_get(f"/repos/{REPO}/releases?per_page=15"), True)
    return pick_release(_get(f"/repos/{REPO}/releases/latest"), False)


def engine_updates(installed: list[dict[str, Any]], release: dict[str, Any]) -> list[dict[str, Any]]:
    """FLE engines (not llama.cpp) that have a newer build in `release`, one entry per installed variant."""
    from fle_studio import engines  # noqa: PLC0415

    latest = str(release.get("tag_name") or "")
    if not latest or not _SAFE_TAG_ENGINE.match(latest):
        return []
    osname, arch = engines._os_arch()
    names = [str(a.get("name") or "") for a in release.get("assets") or [] if isinstance(a, dict)]
    out: list[dict[str, Any]] = []
    for variant in sorted({str(e.get("variant")) for e in installed if e.get("kind") == "fle"}):
        have = [e for e in installed if e.get("variant") == variant]
        if any(e.get("tag") == latest for e in have):
            continue
        try:
            pattern = engines._asset_patterns(variant, osname, arch)[0]
        except (ValueError, KeyError):
            continue
        if not any(re.match(pattern, n) for n in names):
            continue  # this release has no build for this machine
        newest = max(have, key=lambda e: float(e.get("installed") or 0))
        out.append(
            {
                "id": f"{variant}@{latest}",
                "variant": variant,
                "label": f"FLE engine ({newest.get('backend') or variant})",
                "installed": str(newest.get("tag") or ""),
                "latest": latest,
                "url": f"https://github.com/{ENGINE_REPO}/releases/tag/{latest}",
            }
        )
    return out


_SAFE_TAG_ENGINE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._+-]{0,60}$")


# ------------------------------------------------------------------------------------------------ how it is installed


def read_receipt(prefix: Path | None = None) -> str | None:
    """The text of uv's receipt when this Python is the environment of a uv tool."""
    try:
        return ((prefix or Path(sys.prefix)) / "uv-receipt.toml").read_text(encoding="utf-8")
    except OSError:
        return None


def parse_receipt(text: str) -> dict[str, str]:
    """What a uv receipt says about where fle-studio came from: {"source": git|folder|wheel|index, "launcher": path}.
    Read with patterns, not a TOML parser, so it works on Python 3.10 too."""
    req = re.search(r"requirements\s*=\s*\[(.*?)\]\s*(?:\n\w|\Z)", text, re.S)
    block = req.group(1) if req else text
    source = "index"
    if re.search(r"\bgit\s*=", block):
        source = "git"
    elif re.search(r"\b(?:directory|editable)\s*=", block):
        source = "folder"
    elif re.search(r"\b(?:path|url)\s*=", block):
        source = "wheel"
    out = {"source": source, "launcher": ""}
    for m in re.finditer(r"\{[^{}]*\}", text):
        entry = m.group(0)
        name = re.search(r'name\s*=\s*"([^"]+)"', entry)
        where = re.search(r'install-path\s*=\s*"([^"]+)"', entry)
        if name and where and name.group(1) == "fle-studio":
            out["launcher"] = where.group(1).replace("\\\\", "\\")
    return out


def read_direct_url() -> dict[str, Any] | None:
    """pip's record of where the installed package came from (PEP 610), if there is one."""
    try:
        from importlib import metadata  # noqa: PLC0415

        raw = metadata.distribution("fle-studio").read_text("direct_url.json")
        data = json.loads(raw) if raw else None
        return data if isinstance(data, dict) else None
    except Exception:  # not installed as a package, or a damaged record
        return None


def in_service(env: dict[str, str] | None = None, tty: bool | None = None, engine_unit: bool = False) -> bool:
    """Does FLE Studio itself run as a service (systemd), or next to one it controls?"""
    env = os.environ if env is None else env
    if engine_unit or env.get("FLE_STUDIO_SERVICE"):
        return True
    if tty is None:
        tty = bool(getattr(sys.stdout, "isatty", lambda: False)())
    return bool(env.get("INVOCATION_ID") or env.get("JOURNAL_STREAM")) and not tty


def detect_install(
    engine_unit: bool = False,
    env: dict[str, str] | None = None,
    prefix: Path | None = None,
    direct_url: dict[str, Any] | None | bool = False,
    tty: bool | None = None,
) -> dict[str, Any]:
    """How this FLE Studio was installed: {"kind": uv-tool | pip | source, "source": git | folder | wheel | index,
    "service": bool, "launcher": str}. The keyword arguments let tests say what the machine looks like."""
    service = in_service(env, tty, engine_unit)
    receipt = read_receipt(prefix)
    if receipt is not None:
        info = parse_receipt(receipt)
        return {"kind": "uv-tool", "source": info["source"], "service": service, "launcher": info["launcher"]}
    url = read_direct_url() if direct_url is False else direct_url
    source, kind = "index", "pip"
    if isinstance(url, dict):
        if (url.get("dir_info") or {}).get("editable"):
            source, kind = "folder", "source"
        elif url.get("vcs_info"):
            source = "git"
        elif str(url.get("url", "")).endswith((".whl", ".tar.gz", ".zip")):
            source = "wheel"
        elif url.get("dir_info") is not None:
            source = "folder"
    return {"kind": kind, "source": source, "service": service, "launcher": ""}


def update_command(install: dict[str, Any], tag: str) -> str:
    """The command that updates this kind of install to `tag`, for a person to copy. "" when no command fits."""
    if not _SAFE_TAG.match(tag):
        return ""
    git = f"git+{REPO_URL}@{tag}"
    if install["kind"] == "uv-tool":
        # `fle-studio update` keeps the packages installed next to Studio; a bare `uv tool install --force` drops them
        return f"fle-studio update --version {tag}" if install["source"] == "git" else ""
    if install["kind"] == "pip":
        return f'"{sys.executable}" -m pip install --upgrade "{git}"'
    return ""


def advice(install: dict[str, Any], tag: str, uv: bool, git: bool) -> dict[str, Any]:
    """Whether the one-click update is possible and, when it is not, what to do instead."""
    command = update_command(install, tag)
    reason = ""
    if install["kind"] != "uv-tool":
        reason = {
            "pip": "This copy was installed with pip, so pip updates it.",
            "source": "This copy runs from a source folder. Update the checkout with git, then restart FLE Studio.",
        }[install["kind"]]
    elif install["source"] != "git":
        reason = (
            "This copy was installed from a folder or a wheel file. Install the new version the same way "
            "(uv tool install --force <folder or wheel>), then restart FLE Studio."
        )
    elif not uv:
        reason = "uv was not found next to FLE Studio, so it cannot update itself."
    elif not git:
        reason = "git is needed to fetch the release and was not found."
    can_auto = not reason
    if not can_auto and not command and install["kind"] == "source":
        command = "git pull"
    return {"can_auto": can_auto, "reason": reason, "command": command}


# ------------------------------------------------------------------------------------------------ finding tools


def _exe(name: str) -> str:
    return f"{name}.exe" if paths.IS_WIN else name


def find_uv() -> list[str] | None:
    """uv as a command, also when it is not on this process's PATH (it is installed in the home folder)."""
    return selfupdate.find_uv()


def find_git() -> bool:
    return shutil.which("git") is not None


def launcher_command(install: dict[str, Any], uv: list[str] | None) -> list[str]:
    """The `fle-studio` program of this install, to start the new version and to ask it for its version."""
    if install.get("launcher") and Path(install["launcher"]).is_file():
        return [install["launcher"]]
    if uv:
        with contextlib.suppress(OSError, subprocess.SubprocessError):
            out = subprocess.run([*uv, "tool", "dir", "--bin"], capture_output=True, text=True, timeout=30)
            folder = out.stdout.strip()
            if out.returncode == 0 and folder and (Path(folder) / _exe("fle-studio")).is_file():
                return [str(Path(folder) / _exe("fle-studio"))]
    return [sys.executable, "-m", "fle_studio"]


def user_args() -> list[str]:
    """The arguments the user gave `fle-studio`, without opening a second browser tab on the restart."""
    args = list(sys.argv[1:])
    if "--no-browser" not in args:
        args.append("--no-browser")
    return args


def restart_command() -> list[str]:
    """The command line this FLE Studio was started with, without opening a second browser tab."""
    argv = list(getattr(sys, "orig_argv", None) or [sys.executable, "-m", "fle_studio", *sys.argv[1:]])
    if "--no-browser" not in argv:
        argv.append("--no-browser")
    return argv


# ------------------------------------------------------------------------------------------------ the update itself


def windows_helper_text() -> str:
    """The PowerShell script that does the update after FLE Studio has ended (see update_helper.ps1)."""
    return resources.files(__package__).joinpath("update_helper.ps1").read_text(encoding="utf-8")


class Updates:
    """The update check and the update run of one FLE Studio."""

    def __init__(self, data_dir: Path, config: Any, engines: Any) -> None:
        self.dir = data_dir
        self.config = config
        self.engines = engines
        self.file = data_dir / "update-check.json"
        self.log = data_dir / "update.log"
        self.result_file = data_dir / "update-result.json"
        self._lock = threading.Lock()
        self._checking = False
        self._last_manual = 0.0
        self.on_restart: Callable[[list[str] | None], None] | None = None  # set by the launcher: stop and re-run
        self.run: dict[str, Any] = {"state": "idle", "tag": "", "message": "", "tail": []}
        self.engine_unit: Callable[[], bool] = lambda: False
        self.state: dict[str, Any] = self._load()

    # ---- the stored answer

    def _load(self) -> dict[str, Any]:
        try:
            data = json.loads(self.file.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}
        return data if isinstance(data, dict) and data.get("v") == 1 else {}

    def _save(self) -> None:
        tmp = self.file.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.state, indent=1), encoding="utf-8")
        tmp.replace(self.file)

    # ---- settings

    def enabled(self) -> bool:
        return bool(self.config.get("updates.check"))

    def prereleases(self) -> bool:
        return bool(self.config.get("updates.prereleases"))

    # ---- checking

    def due(self, now: float | None = None) -> bool:
        if not self.enabled():
            return False
        now = time.time() if now is None else now
        last = float(self.state.get("attempt") or 0)
        wait = RETRY_OFFLINE if self.state.get("offline") else CHECK_EVERY
        return now - last >= wait

    def check(self, force: bool = False) -> dict[str, Any]:
        """Ask GitHub unless an answer is fresh enough (or force). Never raises for a network problem."""
        if not self.enabled():
            raise UpdateError("Update checks are turned off in Settings.")
        with self._lock:
            if self._checking:
                return self.view()
            if force:
                if time.time() - self._last_manual < MANUAL_GAP:
                    return self.view()
                self._last_manual = time.time()
            elif not self.due():
                return self.view()
            self._checking = True
        try:
            self._ask()
        finally:
            self._checking = False
        return self.view()

    def _ask(self) -> None:
        state: dict[str, Any] = {
            "v": 1,
            "attempt": time.time(),
            "pre": self.prereleases(),
            "offline": False,
            "error": "",
            "checked": self.state.get("checked"),
            "app": self.state.get("app"),
            "engines": self.state.get("engines") or [],
        }
        try:
            state["app"] = fetch_app_release(self.prereleases())
            state["checked"] = time.time()
        except HTTPError as exc:
            state["error"] = f"GitHub answered {exc.code}"
        except OSError as exc:  # no network, DNS, timeout: stay quiet, try again in an hour
            state["offline"] = True
            state["error"] = f"No connection ({getattr(exc, 'reason', None) or exc})"
        except ValueError:
            state["error"] = "GitHub's answer could not be read"
        installed = [e for e in self.engines.installed() if e.get("kind") == "fle"]
        if not state["error"] and installed and self.config.get("engine.fle_tag") == "latest":
            try:
                state["engines"] = engine_updates(installed, _get(f"/repos/{ENGINE_REPO}/releases/latest"))
            except (OSError, ValueError):
                state["engines"] = []  # the engine notice is a bonus; a failure is not shown
        elif not installed:
            state["engines"] = []
        with self._lock:
            self.state = state
            with contextlib.suppress(OSError):
                self._save()

    # ---- what the page shows

    def dismiss(self, version: str = "", engines: bool = False) -> None:
        patch: dict[str, Any] = {}
        if version:
            patch["dismissed"] = version
        if engines:
            patch["dismissed_engines"] = ",".join(e["id"] for e in self.state.get("engines") or [])
        if patch:
            self.config.update({"updates": patch})

    def view(self) -> dict[str, Any]:
        app = self.state.get("app")
        newer = bool(app and is_newer(app["version"], __version__))
        dismissed = self.config.get("updates.dismissed")
        hidden_engines = {x for x in str(self.config.get("updates.dismissed_engines")).split(",") if x}
        install = detect_install(self.engine_unit())
        uv, git = find_uv(), find_git()
        out: dict[str, Any] = {
            "enabled": self.enabled(),
            "prereleases": self.prereleases(),
            "current": __version__,
            "checked": self.state.get("checked"),
            "checking": self._checking,
            "error": self.state.get("error") or "",
            "app": None,
            "engines": [{**e, "dismissed": e["id"] in hidden_engines} for e in self.state.get("engines") or []],
            "install": {"kind": install["kind"], "source": install["source"], "service": install["service"]},
            "run": dict(self.run),
            "result": self._read_result(),
        }
        if newer and app:
            out["app"] = {
                **app,
                "dismissed": dismissed == app["version"],
                **advice(install, app["tag"], uv is not None, git),
            }
        return out

    # ---- running the update

    def apply(self, tag: str) -> dict[str, Any]:
        """Start the one-click update to `tag`, which must be what the last check found."""
        app = self.state.get("app") or {}
        if not self.enabled() or not app or app.get("tag") != tag or not is_newer(app["version"], __version__):
            raise UpdateError("That version is not the one the last check found. Check again first.")
        if self.run["state"] in ("running", "restarting"):
            raise UpdateError("An update is already running.")
        install = detect_install(self.engine_unit())
        uv = find_uv()
        verdict = advice(install, tag, uv is not None, find_git())
        if not verdict["can_auto"] or uv is None:
            raise UpdateError(verdict["reason"] or "This install cannot update itself.")
        self.run = {
            "state": "running",
            "tag": tag,
            "message": "Installing the new version …",
            "tail": [],
            "log": str(self.log),
        }
        if paths.IS_WIN:
            self._start_windows_helper(tag, uv, install)
        else:
            threading.Thread(target=self._run_unix, args=(tag, uv, install), name="update", daemon=True).start()
        return dict(self.run)

    def _spec(self, tag: str) -> str:
        return f"git+{REPO_URL}@{tag}"

    def _log_line(self, text: str) -> None:
        with contextlib.suppress(OSError), self.log.open("a", encoding="utf-8") as fh:
            fh.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {text}\n")

    def _fail(self, message: str) -> None:
        tail: list[str] = []
        with contextlib.suppress(OSError):
            tail = self.log.read_text(encoding="utf-8", errors="replace").splitlines()[-14:]
        self.run = {
            "state": "failed",
            "tag": self.run.get("tag", ""),
            "message": message,
            "tail": tail,
            "log": str(self.log),
        }

    def _run_unix(self, tag: str, uv: list[str], install: dict[str, Any]) -> None:
        # The same steps as `fle-studio update`: the packages next to Studio are installed again with the new version.
        host = selfupdate.Host(run=lambda cmd, stream: selfupdate.real_run(cmd, False))
        try:
            receipt, env, bin_dir = selfupdate.read_receipt(host, uv)
            main = receipt.main
            assert main is not None
            selfupdate.install(
                host,
                uv,
                self.dir,
                receipt,
                env,
                selfupdate.launcher_of(receipt, bin_dir),
                main,
                receipt.extras,
                receipt.python or UV_PYTHON,
                tag,
                selfupdate.Log(self.dir, lambda _text: None),  # into update.log only
                options=receipt.options,
            )
        except selfupdate.Problem as problem:
            self._log_line(problem.text())
            self._fail(f"{problem.text()}\nThe old version keeps running.")
            return
        launcher = launcher_command(install, uv)
        try:
            shown = subprocess.run([*launcher, "--version"], capture_output=True, text=True, timeout=60).stdout
        except (OSError, subprocess.SubprocessError) as exc:
            shown = ""
            self._log_line(f"could not start the new version: {exc}")
        want = parse_version(tag)
        got = parse_version((re.findall(r"\d[\w.+-]*", shown) or [""])[-1])
        if want is None or got is None or want != got:
            self._log_line(f"the new version answered {shown.strip()!r}, expected {tag}")
            self._fail(
                f"The new version was installed but does not start as {tag} ({shown.strip() or 'no answer'}). "
                "Restart FLE Studio by hand."
            )
            return
        self._log_line(f"installed {tag}; restarting")
        self.run = {**self.run, "state": "restarting", "message": "Restarting FLE Studio …"}
        if self.on_restart is not None:
            self.on_restart(restart_command())
        else:
            self.run = {**self.run, "state": "done", "message": "Installed. Restart FLE Studio to use it."}

    def _start_windows_helper(self, tag: str, uv: list[str], install: dict[str, Any]) -> None:
        """Windows locks the running program, so a small PowerShell script does the install after FLE Studio ended."""
        script = self.dir / "update-helper.ps1"
        job = self.dir / "update-job.json"
        start = [*launcher_command(install, uv), *user_args()]
        spec, with_args = self._spec(tag), []
        with contextlib.suppress(selfupdate.Problem):  # the packages next to Studio come along, as with the command
            receipt, _, _ = selfupdate.read_receipt(selfupdate.Host(), uv)
            for extra in receipt.extras:
                with_args += selfupdate.with_args(extra)
            if receipt.main is not None:
                spec = selfupdate.main_spec(receipt.main, tag)
        job.write_text(
            json.dumps(
                {
                    "pid": os.getpid(),
                    "uv": uv,
                    "spec": spec,
                    "with": with_args,
                    "python": UV_PYTHON,
                    "tag": tag,
                    "log": str(self.log),
                    "result": str(self.result_file),
                    "start": start,
                }
            ),
            encoding="utf-8",
        )
        script.write_text(windows_helper_text(), encoding="utf-8")
        self._log_line(f"starting the helper for {tag}")
        flags = (
            getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
            | getattr(subprocess, "DETACHED_PROCESS", 0)
            | getattr(subprocess, "CREATE_NO_WINDOW", 0)
        )
        try:
            subprocess.Popen(
                ["powershell.exe", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(script), "-Job", str(job)],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                creationflags=flags,
                close_fds=True,
            )
        except OSError as exc:
            self._log_line(f"could not start the helper: {exc}")
            self._fail(f"Could not start the update helper: {exc}. The old version keeps running.")
            return
        self.run = {**self.run, "state": "restarting", "message": "FLE Studio closes, updates and starts again …"}
        if self.on_restart is not None:
            self.on_restart(None)  # just end; the helper starts it again

    def _read_result(self) -> dict[str, Any] | None:
        try:
            data = json.loads(self.result_file.read_text(encoding="utf-8-sig"))
        except (OSError, ValueError):
            return None
        return data if isinstance(data, dict) else None

    def ack(self) -> None:
        for name in (self.result_file, self.dir / "update-job.json"):
            with contextlib.suppress(OSError):
                name.unlink()
        if self.run["state"] in ("failed", "done"):
            self.run = {"state": "idle", "tag": "", "message": "", "tail": []}


# ------------------------------------------------------------------------------------------------ server side


def service(studio: Any) -> Updates:
    """The Updates of a Studio, made on first use."""
    found = studio.__dict__.get("updates")
    if found is None:
        found = studio.__dict__["updates"] = Updates(studio.data_dir, studio.config, studio.engines)
        found.engine_unit = lambda: studio.engine.unit is not None
    return found  # type: ignore[no-any-return]


def start_background(studio: Any) -> Callable[[], None]:
    """Check once a day while FLE Studio runs: the first look a few seconds after the start.

    Returns the function that ends the loop (the plugin is turned off)."""
    updates = service(studio)
    stop = threading.Event()

    def loop() -> None:
        if stop.wait(8):
            return
        while True:
            with contextlib.suppress(Exception):  # a failed check must never end the loop
                if updates.due():
                    updates.check()
            if stop.wait(1800):
                return

    threading.Thread(target=loop, name="update-check", daemon=True).start()
    return stop.set


ROUTES: list[tuple[str, str, str]] = [
    ("GET", r"^/api/update$", "update"),
    ("POST", r"^/api/update/check$", "update_check"),
    ("POST", r"^/api/update/dismiss$", "update_dismiss"),
    ("POST", r"^/api/update/apply$", "update_apply"),
    ("POST", r"^/api/update/ack$", "update_ack"),
]

_ApiError: Any = None


def _fail(exc: UpdateError) -> Exception:
    return _ApiError(exc.status, str(exc))  # type: ignore[no-any-return]


def api_update(self: Any, query: dict[str, str]) -> None:
    self.send_json(200, service(self.studio).view())


def api_update_check(self: Any, query: dict[str, str]) -> None:
    try:
        self.send_json(200, service(self.studio).check(force=True))
    except UpdateError as exc:
        raise _fail(exc) from None


def api_update_dismiss(self: Any, query: dict[str, str]) -> None:
    body = self.read_json()
    updates = service(self.studio)
    updates.dismiss(str(body.get("version") or ""), body.get("engines") is True)
    self.send_json(200, updates.view())


def api_update_apply(self: Any, query: dict[str, str]) -> None:
    body = self.read_json()
    updates = service(self.studio)
    try:
        updates.apply(str(body.get("tag") or ""))
    except UpdateError as exc:
        raise _fail(exc) from None
    self.send_json(200, updates.view())


def api_update_ack(self: Any, query: dict[str, str]) -> None:
    updates = service(self.studio)
    updates.ack()
    self.send_json(200, updates.view())


def install(add: Callable[[str, str, Callable[..., None]], None], api_error: type) -> None:
    """Give the plugin context's `api` function the routes of this module."""
    global _ApiError
    _ApiError = api_error
    for method, pattern, name in ROUTES:
        add(method, pattern, globals()[f"api_{name}"])

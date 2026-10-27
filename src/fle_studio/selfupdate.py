"""`fle-studio update`: bring FLE Studio to a new version in one command, and keep everything the user has.

What the command does, in this order, and what it says when a step does not work:

1. Reads uv's receipt of the installed tool, so it knows the packages that were installed *next to* Studio
   (a plugin such as `fle-studio-ginny`). `uv tool install --force` forgets them; this command gives them back.
2. Installs the new version with the same packages, and checks that the launcher now reports it.
3. Refreshes the Dock starter on macOS (see dock_starter.py).
4. Restarts the service that runs Studio (launchd on macOS, a systemd user unit on Linux), and asks the running
   server which version it has. "Installed" and "running" are both printed, so a stale service cannot hide.

`--rollback` goes back to the version before the last update, with the packages it had then.

The module uses only the standard library and imports everything it needs at the top. The command replaces the very
environment it runs in, so nothing may be imported from FLE Studio's own files after step 2.
Everything that touches the machine (running programs, asking the server, sleeping) sits in `Host`, which the tests
replace.
"""

from __future__ import annotations

import argparse
import getpass
import http.client
import json
import os
import plistlib
import re
import shlex
import shutil
import ssl
import subprocess
import sys
import time
import urllib.parse
import urllib.request
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, NamedTuple

from . import __version__, dock_starter, net, paths
from .versions import SAFE_TAG, same_version

REPO = "sirfyyn/fle-studio"
REPO_URL = f"https://github.com/{REPO}"
PACKAGE = "fle-studio"
DEFAULT_PYTHON = "3.12"  # what the installers use
DEFAULT_PORT = 8642
SERVICE_WAIT = 60  # seconds the restarted server gets to answer with the new version
UV_TIMEOUT = 1800


class Problem(Exception):
    """Something that stops the update. `cause` says what is wrong, `remedy` what to do about it."""

    def __init__(self, cause: str, remedy: str = "") -> None:
        super().__init__(cause)
        self.cause = cause
        self.remedy = remedy

    def text(self) -> str:
        return self.cause + (f"\n  What to do: {self.remedy}" if self.remedy else "")


# ------------------------------------------------------------------------------------------------ the machine


class Result(NamedTuple):
    code: int
    out: str


def real_run(cmd: Sequence[str], stream: bool = False) -> Result:
    """Run a program and give back its exit code and everything it printed (stdout and stderr together)."""
    try:
        proc = subprocess.Popen(
            list(cmd),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            errors="replace",
        )
    except OSError as exc:
        return Result(127, f"could not run {cmd[0]}: {exc}")
    lines: list[str] = []
    assert proc.stdout is not None
    for line in proc.stdout:
        lines.append(line)
        if stream:
            print(f"    {line.rstrip()}", flush=True)
    return Result(proc.wait(), "".join(lines))


def real_ping(port: int) -> str | None:
    """The version of the FLE Studio that answers on this port of this machine, or None."""
    insecure = ssl.create_default_context()
    insecure.check_hostname = False  # a self-made certificate on the loopback address; only the version is read
    insecure.verify_mode = ssl.CERT_NONE
    for scheme, context in (("http", None), ("https", insecure)):
        try:
            with urllib.request.urlopen(f"{scheme}://127.0.0.1:{port}/api/ping", timeout=2, context=context) as r:
                data = json.loads(r.read().decode("utf-8"))
        except (OSError, ValueError, http.client.HTTPException):
            continue
        if isinstance(data, dict) and data.get("app") == "fle-studio":
            return str(data.get("version") or "")
    return None


def _apps_dirs() -> tuple[Path, ...]:
    override = os.environ.get("FLE_STUDIO_APPS_DIR")
    return (Path(override),) if override else (Path.home() / "Applications", Path("/Applications"))


@dataclass
class Host:
    """What the update does to the machine. The tests hand in a stand-in with the same fields."""

    run: Callable[[Sequence[str], bool], Result] = real_run
    ping: Callable[[int], str | None] = real_ping
    sleep: Callable[[float], None] = time.sleep
    platform: str = sys.platform
    home: Path = field(default_factory=Path.home)
    uid: int = field(default_factory=lambda: os.getuid() if hasattr(os, "getuid") else 0)
    user: str = field(default_factory=getpass.getuser)
    apps_dirs: tuple[Path, ...] = field(default_factory=_apps_dirs)
    which: Callable[[str], str | None] = shutil.which


# ------------------------------------------------------------------------------------------------ the receipt


@dataclass(frozen=True)
class Requirement:
    """One package of the tool environment, as uv's receipt lists it."""

    name: str
    kind: str  # git | directory | editable | path | url | index
    where: str = ""  # git: repository address; folder and file kinds: the path; url: the address; index: specifier
    extras: tuple[str, ...] = ()
    rev: str = ""  # git only: tag, branch or commit
    subdirectory: str = ""  # git only

    def to_json(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "kind": self.kind,
            "where": self.where,
            "extras": list(self.extras),
            "rev": self.rev,
            "subdirectory": self.subdirectory,
        }

    @staticmethod
    def from_json(data: Any) -> Requirement:
        if not isinstance(data, dict) or not data.get("name") or not data.get("kind"):
            raise ValueError("not a requirement")
        return Requirement(
            str(data["name"]),
            str(data["kind"]),
            str(data.get("where") or ""),
            tuple(str(x) for x in data.get("extras") or []),
            str(data.get("rev") or ""),
            str(data.get("subdirectory") or ""),
        )


@dataclass(frozen=True)
class Receipt:
    requirements: tuple[Requirement, ...]
    python: str = ""
    launcher: str = ""
    options: tuple[str, ...] = ()  # uv arguments the receipt keeps in [tool.options]: --index-url URL, ...

    @property
    def main(self) -> Requirement | None:
        return next((r for r in self.requirements if r.name == PACKAGE), None)

    @property
    def extras(self) -> tuple[Requirement, ...]:
        """Everything installed next to FLE Studio."""
        return tuple(r for r in self.requirements if r.name != PACKAGE)


_STRING = r'"((?:[^"\\]|\\.)*)"'


def _unescape(text: str) -> str:
    return text.replace("\\\\", "\\").replace('\\"', '"')


def _requirements_block(text: str) -> str:
    """The text between the brackets of `requirements = [ ... ]` (strings may contain brackets)."""
    start = re.search(r"^\s*requirements\s*=\s*\[", text, re.M)
    if not start:
        return ""
    depth, quoted, i = 1, False, start.end()
    while i < len(text):
        ch = text[i]
        if quoted:
            if ch == "\\":
                i += 1
            elif ch == '"':
                quoted = False
        elif ch == '"':
            quoted = True
        elif ch == "[":
            depth += 1
        elif ch == "]":
            depth -= 1
            if depth == 0:
                return text[start.end() : i]
        i += 1
    return text[start.end() :]


def _requirement(entry: str) -> Requirement | None:
    fields = {k: _unescape(v) for k, v in re.findall(rf"([\w-]+)\s*=\s*{_STRING}", entry)}
    name = fields.get("name")
    if not name:
        return None
    extras_match = re.search(r"\bextras\s*=\s*\[([^\]]*)\]", entry)
    extras = tuple(re.findall(r'"([^"]+)"', extras_match.group(1))) if extras_match else ()
    for kind in ("git", "editable", "directory", "path", "url"):
        if kind not in fields:
            continue
        where = fields[kind]
        if kind != "git":
            return Requirement(name, kind, where, extras)
        base, _, query = where.partition("?")
        params = urllib.parse.parse_qs(query)
        rev = next((params[k][0] for k in ("rev", "tag", "branch") if k in params), "")
        rev = rev or next((fields[k] for k in ("rev", "tag", "branch") if k in fields), "")
        return Requirement(name, "git", base, extras, rev, (params.get("subdirectory") or [""])[0])
    return Requirement(name, "index", fields.get("specifier", ""), extras)


def parse_receipt(text: str) -> Receipt:
    """Read uv's `uv-receipt.toml`. Patterns instead of a TOML parser, so it works on Python 3.10 as well."""
    requirements = tuple(
        r for r in (_requirement(e) for e in re.findall(r"\{[^{}]*\}", _requirements_block(text))) if r is not None
    )
    python = re.search(rf"^\s*python\s*=\s*{_STRING}", text, re.M)
    launcher = ""
    for entry in re.findall(r"\{[^{}]*\}", text.split("entrypoints", 1)[-1] if "entrypoints" in text else ""):
        name = re.search(rf'name\s*=\s*"{PACKAGE}"', entry)
        where = re.search(rf"install-path\s*=\s*{_STRING}", entry)
        if name and where:
            launcher = _unescape(where.group(1))
    return Receipt(requirements, python.group(1) if python else "", launcher, _options(text))


def _options(text: str) -> tuple[str, ...]:
    """The package index addresses uv stored in `[tool.options]`; a plugin from a private index needs them again."""
    section = re.search(r"^\[tool\.options\]\s*\n(.*?)(?=^\[|\Z)", text, re.M | re.S)
    out: list[str] = []
    for key, flag in (("index-url", "--index-url"), ("extra-index-url", "--extra-index-url")):
        line = re.search(rf"^{key}\s*=\s*(.+)$", section.group(1), re.M) if section else None
        for url in re.findall(r'"([^"]+)"', line.group(1)) if line else []:
            out += [flag, url]
    return tuple(out)


# ------------------------------------------------------------------------------------------------ install command


def git_spec(r: Requirement, rev: str | None = None) -> str:
    """`git+URL@REV`, the form `uv tool install` takes, for a git requirement (with another revision when given)."""
    rev = r.rev if rev is None else rev
    spec = f"git+{r.where}" + (f"@{rev}" if rev else "")
    return spec + (f"#subdirectory={r.subdirectory}" if r.subdirectory else "")


def with_args(r: Requirement) -> list[str]:
    """The `--with` argument that installs this package again next to Studio."""
    label = r.name + (f"[{','.join(r.extras)}]" if r.extras else "")
    if r.kind == "editable":
        return ["--with-editable", r.where]
    if r.kind in ("directory", "path"):
        return ["--with", f"{label} @ {Path(os.path.abspath(r.where)).as_uri()}" if r.extras else r.where]
    if r.kind == "git":
        return ["--with", f"{label} @ {git_spec(r)}"]
    if r.kind == "url":
        return ["--with", f"{label} @ {r.where}"]
    return ["--with", f"{label}{r.where}"]


def main_spec(main: Requirement, tag: str | None = None) -> str:
    """What to install as FLE Studio itself: the same repository as before, at `tag`."""
    if main.kind != "git":
        raise Problem(
            f"FLE Studio was installed from {'a folder' if main.kind in ('directory', 'editable') else 'a file'} "
            f"({main.where}), not from GitHub, so there is no release to move to.",
            "Install the new version from the same place: "
            + shlex.join(["uv", "tool", "install", "--force", main.where]),
        )
    spec = git_spec(main, tag)
    return f"{PACKAGE}[{','.join(main.extras)}] @ {spec}" if main.extras else spec


def install_command(
    uv: Sequence[str],
    main: Requirement,
    extras: Sequence[Requirement],
    python: str,
    tag: str | None = None,
    options: Sequence[str] = (),
) -> list[str]:
    cmd = [*uv, "tool", "install", "--python", python or DEFAULT_PYTHON, "--force", *options]
    for r in extras:
        cmd += with_args(r)
    cmd.append(main_spec(main, tag))
    return cmd


def missing_extras(extras: Sequence[Requirement]) -> list[Requirement]:
    """Packages that come from a folder or file that is not there any more."""
    return [r for r in extras if r.kind in ("directory", "editable", "path") and not Path(r.where).exists()]


# ------------------------------------------------------------------------------------------------ finding things


def find_uv(host: Host | None = None) -> list[str] | None:
    """uv as a command, also when it is not on PATH (it is installed in the home folder)."""
    host = host or Host()
    found = host.which("uv")
    if found:
        return [found]
    for folder in (
        os.environ.get("UV_INSTALL_DIR"),
        os.environ.get("XDG_BIN_HOME"),
        str(host.home / ".local" / "bin"),
        str(host.home / ".cargo" / "bin"),
    ):
        if folder and (Path(folder) / paths.exe("uv")).is_file():
            return [str(Path(folder) / paths.exe("uv"))]
    return None


def tool_dirs(host: Host, uv: Sequence[str]) -> tuple[Path | None, Path | None]:
    """(uv's folder of tool environments, uv's folder of tool launchers), as uv itself reports them."""
    base = host.run([*uv, "tool", "dir"], False)
    bins = host.run([*uv, "tool", "dir", "--bin"], False)
    return (
        Path(base.out.strip()) if base.code == 0 and base.out.strip() else None,
        Path(bins.out.strip()) if bins.code == 0 and bins.out.strip() else None,
    )


def read_receipt(host: Host, uv: Sequence[str]) -> tuple[Receipt, Path, Path | None]:
    """(receipt, the tool environment, the launcher folder). Raises Problem when this is no uv tool install."""
    base, bin_dir = tool_dirs(host, uv)
    candidates = [base / PACKAGE] if base else []
    candidates.append(Path(sys.prefix))
    for env in candidates:
        try:
            text = (env / "uv-receipt.toml").read_text(encoding="utf-8")
        except OSError:
            continue
        receipt = parse_receipt(text)
        if receipt.main is not None:
            return receipt, env, bin_dir
    raise Problem(
        "FLE Studio was not installed as a uv tool (uv has no receipt for it), so this command does not know "
        "how to update it.",
        f"Install it the supported way once: curl -LsSf https://raw.githubusercontent.com/{REPO}/main/install.sh | sh"
        f' (or, installed with pip: python -m pip install --upgrade "git+{REPO_URL}")',
    )


def launcher_of(receipt: Receipt, bin_dir: Path | None) -> str:
    if receipt.launcher and Path(receipt.launcher).is_file():
        return receipt.launcher
    if bin_dir and (bin_dir / paths.exe(PACKAGE)).is_file():
        return str(bin_dir / paths.exe(PACKAGE))
    return ""


def configured_port(data_dir: Path | None = None) -> int:
    """The port FLE Studio listens on according to its settings (8642 unless changed)."""
    try:
        settings = json.loads(((data_dir or paths.data_dir()) / "settings.json").read_text(encoding="utf-8"))
        return int(settings["server"]["port"])
    except (OSError, ValueError, KeyError, TypeError):
        return DEFAULT_PORT


@dataclass(frozen=True)
class Service:
    manager: str  # launchd | systemd
    name: str  # launchd label, or systemd unit name
    port: int
    definition: Path  # the plist or the unit file

    def describe(self) -> str:
        return f"{self.manager} {self.name}"


def _port_in(args: Sequence[str], default: int) -> int:
    for i, arg in enumerate(args):
        value = (
            arg.split("=", 1)[1]
            if arg.startswith("--port=")
            else args[i + 1]
            if arg == "--port" and i + 1 < len(args)
            else ""
        )
        if value.isdigit():
            return int(value)
    return default


def find_services(host: Host, default_port: int) -> list[Service]:
    """The services of this user that start FLE Studio: launchd agents (macOS) or systemd user units (Linux)."""
    found: list[Service] = []
    if host.platform == "darwin":
        for plist in sorted((host.home / "Library" / "LaunchAgents").glob("*.plist")):
            try:
                data = plistlib.loads(plist.read_bytes())
            except (OSError, ValueError, plistlib.InvalidFileException):
                continue
            args = [str(a) for a in data.get("ProgramArguments") or []]
            if args and Path(args[0]).name == PACKAGE and data.get("Label"):
                found.append(Service("launchd", str(data["Label"]), _port_in(args, default_port), plist))
    elif host.platform.startswith("linux"):
        for unit in sorted((host.home / ".config" / "systemd" / "user").glob("*.service")):
            try:
                lines = unit.read_text(encoding="utf-8", errors="replace").splitlines()
            except OSError:
                continue
            for line in lines:
                if not line.startswith("ExecStart="):
                    continue
                try:
                    args = shlex.split(line.split("=", 1)[1].replace("%h", str(host.home)))
                except ValueError:
                    break
                if args and Path(args[0].lstrip("-@+!:")).name == PACKAGE:
                    found.append(Service("systemd", unit.name, _port_in(args, default_port), unit))
                break
    return found


def restart_service(host: Host, service: Service) -> tuple[bool, str]:
    """Restart the service. Returns (restarted, a sentence about it); raises Problem when it cannot be restarted.
    A service that is not loaded or not running is left alone: the user stopped it on purpose."""
    if service.manager == "launchd":
        target = f"gui/{host.uid}/{service.name}"
        if host.run(["launchctl", "print", target], False).code != 0:
            load = f"launchctl bootstrap gui/{host.uid} '{service.definition}'"
            return False, f"{service.describe()} is not loaded, so it was left alone (to load it: {load})"
        done = host.run(["launchctl", "kickstart", "-k", target], False)
        if done.code != 0:
            raise Problem(
                f"launchd would not restart {service.name}: {done.out.strip() or 'no message'}",
                f"Restart it yourself: launchctl kickstart -k {target}",
            )
        return True, f"Restarted {service.describe()}."
    if host.run(["systemctl", "--user", "is-active", service.name], False).code != 0:
        start = f"systemctl --user start {service.name}"
        return False, f"{service.describe()} is not running, so it was left alone (to start it: {start})"
    done = host.run(["systemctl", "--user", "restart", service.name], False)
    if done.code != 0:
        no_bus = "bus" in done.out.lower() or "XDG_RUNTIME_DIR" in done.out
        raise Problem(
            f"systemd would not restart {service.name}: {done.out.strip() or 'no message'}",
            "Log in on the machine itself (or: export XDG_RUNTIME_DIR=/run/user/$(id -u)) and run "
            f"systemctl --user restart {service.name}"
            if no_bus
            else f"Restart it yourself: systemctl --user restart {service.name}",
        )
    return True, f"Restarted {service.describe()}."


def wait_for_version(host: Host, port: int, want: str, seconds: int = SERVICE_WAIT) -> str | None:
    """Poll the server on `port` until it reports `want`. Returns the last version seen (None: nothing answered)."""
    seen: str | None = None
    for _ in range(max(1, seconds)):
        seen = host.ping(port)
        if seen is not None and same_version(seen, want):
            return seen
        host.sleep(1)
    return seen


# ------------------------------------------------------------------------------------------------ problems, explained


def foreign_owned(root: Path, uid: int, limit: int = 100000) -> Path | None:
    """The first file or folder below `root` (or `root` itself) that another user owns, e.g. root after a sudo run."""
    seen = 0
    try:
        if root.lstat().st_uid != uid:
            return root
    except OSError:
        return None
    for folder, dirs, files in os.walk(root):
        for name in (*dirs, *files):
            seen += 1
            if seen > limit:
                return None
            try:
                if os.lstat(os.path.join(folder, name)).st_uid != uid:
                    return Path(folder) / name
            except OSError:
                continue
    return None


def ownership_problem(host: Host, env: Path, launcher: str) -> Problem | None:
    if host.platform == "win32":
        return None
    wrong = [p for p in (env, Path(launcher) if launcher else None) if p and foreign_owned(p, host.uid)]
    if not wrong:
        return None
    command = shlex.join(["sudo", "chown", "-R", host.user, *map(str, wrong)])
    return Problem(
        "uv's folder for FLE Studio belongs to another user (usually root, after an update that ran with sudo), "
        "so uv cannot change it: " + ", ".join(map(str, wrong)),
        f"Give it back to you, once: {command}   Then run fle-studio update again.",
    )


def explain_uv_failure(output: str, tag: str, host: Host, env: Path, launcher: str, code: int = 1) -> Problem:
    """A plain sentence for what uv complained about. The last lines of uv's own text follow."""
    low = output.lower()
    tail = "\n".join(output.strip().splitlines()[-6:])
    wrong_owner = ownership_problem(host, env, launcher)
    if "permission denied" in low or "operation not permitted" in low:
        if wrong_owner:
            return wrong_owner
        return Problem(
            "uv was not allowed to write its tool folder (permission denied).",
            f"Check who owns {env} (ls -ld) and make it yours: sudo chown -R {host.user} {shlex.quote(str(env))}",
        )
    if any(
        s in low
        for s in (
            "could not resolve host",
            "dns error",
            "error sending request",
            "connection refused",
            "timed out",
            "network is unreachable",
            "failed to resolve",
            "tcp connect error",
        )
    ):
        return Problem(
            "No connection to GitHub, so the new version could not be downloaded. Nothing was changed.",
            "Check the internet connection and run the same command again.",
        )
    if any(
        s in low
        for s in (
            "did not match any",
            "couldn't find remote ref",
            "not found in",
            "unknown revision",
            "pathspec",
            "reference is not a tree",
            "couldn't find",
        )
    ):
        return Problem(
            f"There is no release {tag} on GitHub.",
            f"See the releases: {REPO_URL}/releases, or run fle-studio update without --version for the latest.",
        )
    if "no such file or directory" in low and "git" in low:
        return Problem("git was not found; uv needs it to fetch the release.", "macOS: xcode-select --install")
    return Problem(
        f"uv could not install the new version (exit code {code}).\n" + tail,
        "The messages above are uv's own. The full text is in update.log in the data folder. "
        "Going back to the version before: fle-studio update --rollback",
    )


# ------------------------------------------------------------------------------------------------ remembered version


def previous_file(data_dir: Path) -> Path:
    return data_dir / "update-previous.json"


def save_previous(data_dir: Path, receipt: Receipt, version: str) -> None:
    """Remember what is installed now, for `fle-studio update --rollback`."""
    record = {
        "v": 1,
        "version": version,
        "python": receipt.python,
        "options": list(receipt.options),
        "saved": time.time(),
        "requirements": [r.to_json() for r in receipt.requirements],
    }
    data_dir.mkdir(parents=True, exist_ok=True)
    tmp = previous_file(data_dir).with_suffix(".tmp")
    tmp.write_text(json.dumps(record, indent=1), encoding="utf-8")
    tmp.replace(previous_file(data_dir))


def load_previous(data_dir: Path) -> tuple[str, Receipt]:
    try:
        record = json.loads(previous_file(data_dir).read_text(encoding="utf-8"))
        reqs = tuple(Requirement.from_json(x) for x in record["requirements"])
        if record.get("v") != 1 or not any(r.name == PACKAGE for r in reqs):
            raise ValueError("unknown record")
        options = tuple(str(x) for x in record.get("options") or [])
        return str(record["version"]), Receipt(reqs, str(record.get("python") or ""), "", options)
    except (OSError, ValueError, KeyError, TypeError):
        raise Problem(
            "There is no earlier version to go back to (an update has not run on this machine yet, or its record "
            "is damaged).",
            "Install a version by name: fle-studio update --version vX.Y.Z",
        ) from None


# ------------------------------------------------------------------------------------------------ latest release


def latest_tag() -> str:
    base = os.environ.get("FLE_STUDIO_GITHUB_API", "https://api.github.com").rstrip("/")
    try:
        data, _ = net.get_json(
            f"{base}/repos/{REPO}/releases/latest", timeout=15, headers={"Accept": "application/vnd.github+json"}
        )
        tag = str(data.get("tag_name") or "")
    except (OSError, ValueError, AttributeError) as exc:
        raise Problem(
            f"Could not ask GitHub for the latest release ({getattr(exc, 'reason', None) or exc}).",
            "Check the internet connection, or name the version yourself: fle-studio update --version vX.Y.Z",
        ) from None
    if not SAFE_TAG.match(tag):
        raise Problem(f"GitHub's latest release has an unusable name ({tag!r}).", "Name the version: --version vX.Y.Z")
    return tag


def normalise_tag(text: str) -> str:
    tag = text.strip()
    if not SAFE_TAG.match(tag):
        raise Problem(
            f"{text!r} is not a release name.",
            "Use a release such as v0.4.1 (see the list at " + REPO_URL + "/releases).",
        )
    return tag if tag.startswith("v") else f"v{tag}"


# ------------------------------------------------------------------------------------------------ the update


@dataclass
class Options:
    version: str = ""
    rollback: bool = False
    reinstall: bool = False
    dry_run: bool = False
    no_restart: bool = False
    drop: tuple[str, ...] = ()


class Log:
    """What the command says, on the screen and in update.log of the data folder."""

    def __init__(self, data_dir: Path, out: Callable[[str], None] = print) -> None:
        self.file = data_dir / "update.log"
        self.out = out

    def __call__(self, text: str = "") -> None:
        self.out(text)
        self.keep(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {text}")

    def keep(self, text: str) -> None:
        """Only into the log (uv's own output was already shown)."""
        try:
            self.file.parent.mkdir(parents=True, exist_ok=True)
            with self.file.open("a", encoding="utf-8") as fh:
                fh.write(text.rstrip("\n") + "\n")
        except OSError:
            pass


def _installed_version(host: Host, launcher: str) -> str:
    done = host.run([launcher, "--version"], False)
    found = re.findall(r"\d[\w.+!-]*", done.out)
    return found[-1] if done.code == 0 and found else ""


def install(
    host: Host,
    uv: Sequence[str],
    data_dir: Path,
    receipt: Receipt,
    env: Path,
    launcher: str,
    main: Requirement,
    extras: Sequence[Requirement],
    python: str,
    tag: str | None,
    say: Log,
    current: str = __version__,
    options: Sequence[str] = (),
) -> tuple[Receipt, str]:
    """Check what can be checked beforehand, remember the current state (`current` is the installed version), run uv
    and look at the result.

    Returns the new receipt and the new launcher. The command that is run is the same for the command line and for the
    button in the Updates card, so both keep the same packages."""
    if not host.which("git"):
        raise Problem("git was not found; uv needs it to fetch the release.", "macOS: xcode-select --install")
    gone = missing_extras(extras)
    if gone:
        raise Problem(
            "These packages came from a folder that is not there any more: "
            + ", ".join(f"{r.name} ({r.where})" for r in gone)
            + ".",
            "Put the folder back, or leave the package out: fle-studio update "
            + " ".join(f"--drop {r.name}" for r in gone),
        )
    problem = ownership_problem(host, env, launcher)
    if problem:
        raise problem
    command = install_command(uv, main, extras, python, tag, options)
    save_previous(data_dir, receipt, current)  # also for a rollback: going back twice goes forward again
    say("Installing …")
    say("$ " + shlex.join(command))
    done = host.run(command, True)
    say.keep(done.out)
    if done.code != 0:
        raise explain_uv_failure(done.out, tag or "that release", host, env, launcher, done.code)
    new, _, bin_dir = read_receipt(host, uv)
    lost = [r.name for r in extras if r.name not in {x.name for x in new.extras}]
    if lost:
        raise Problem(
            f"The update installed, but these packages are missing from the new environment: {', '.join(lost)}.",
            "Go back to the version before: fle-studio update --rollback",
        )
    return new, launcher_of(new, bin_dir)


def settle(host: Host, services: Sequence[Service], port: int, want: str, say: Log, restart: bool = True) -> int:
    """Make the running Studio the installed version, and say which version runs. Returns 0, or 3 when a Studio that
    was started by hand (no service) still runs another version."""
    if not services:
        running = host.ping(port)
        if running is None:
            say("No service runs FLE Studio here and no Studio is running. Start it with: fle-studio")
        elif same_version(running, want):
            say(f"Running: {running} on port {port}.")
        else:
            say(
                f"Installed: {want}. Running: {running}. This Studio was started by hand, not by a service, so it "
                "cannot be restarted from here. Stop it (Ctrl+C in its terminal) and start fle-studio again."
            )
            return 3
        return 0
    for service in services:
        running = host.ping(service.port)
        if running is not None and same_version(running, want):
            say(f"Running: {running} on port {service.port} ({service.describe()}); nothing to restart.")
            continue
        if not restart:
            say(f"{service.describe()} runs {running or 'nothing'}; not restarted (--no-restart).")
            continue
        restarted, sentence = restart_service(host, service)
        say(sentence)
        if not restarted:
            continue
        seen = wait_for_version(host, service.port, want)
        if seen is None:
            raise Problem(
                f"{service.describe()} was restarted, but nothing answers on port {service.port} "
                f"within {SERVICE_WAIT} seconds.",
                _where_to_look(service),
            )
        if not same_version(seen, want):
            raise Problem(
                f"{service.describe()} answers with {seen}, expected {want}.",
                f"Another program may hold port {service.port} (lsof -nP -iTCP:{service.port} -sTCP:LISTEN).",
            )
        say(f"Running: {seen} on port {service.port}.")
    return 0


def update(opts: Options, host: Host, data_dir: Path, out: Callable[[str], None] = print) -> int:
    """Run the update. Returns the exit code: 0 done, 3 installed but a Studio started by hand still runs the old
    version. A Problem is raised for everything that stops the update."""
    say = Log(data_dir, out)
    if host.platform == "win32":
        raise Problem(
            "On Windows the running program cannot replace itself from a command line.",
            "Use Settings → Updates in FLE Studio, or run install.ps1 again.",
        )
    uv = find_uv(host)
    if uv is None:
        raise Problem(
            "uv was not found, and FLE Studio is updated with it.",
            f"Run the installer again: curl -LsSf https://raw.githubusercontent.com/{REPO}/main/install.sh | sh",
        )
    receipt, env, bin_dir = read_receipt(host, uv)
    launcher = launcher_of(receipt, bin_dir)
    # What is installed is what the installed program says, not what the code running this command is
    current = (_installed_version(host, launcher) if launcher else "") or __version__
    if opts.rollback:
        want_version, target = load_previous(data_dir)
        if same_version(want_version, current):
            raise Problem(f"FLE Studio {current} is already the version before the last update.")
        tag = None
        say(f"Going back from FLE Studio {current} to {want_version}.")
    else:
        target = receipt
        tag = normalise_tag(opts.version) if opts.version else latest_tag()
        want_version = tag.lstrip("v")
        say(f"FLE Studio {current} is installed; the target is {want_version}.")
    main = target.main
    assert main is not None  # read_receipt and load_previous both ask for it
    extras = tuple(r for r in target.extras if r.name not in opts.drop)
    python = target.python or receipt.python or DEFAULT_PYTHON
    options = target.options
    if extras:
        say("Kept next to FLE Studio: " + ", ".join(r.name for r in extras))
    if opts.dry_run:
        say("Dry run. Would run: " + shlex.join(install_command(uv, main, extras, python, tag, options)))
        return 0

    already = same_version(want_version, current) and not opts.reinstall and not opts.rollback
    if already:
        say(f"{want_version} is already installed; checking the service.")
    else:
        receipt, launcher = install(
            host, uv, data_dir, receipt, env, launcher, main, extras, python, tag, say, current, options
        )
    shown = _installed_version(host, launcher) if launcher else ""
    if not shown or not same_version(shown, want_version):
        raise Problem(
            f"The installed program reports {shown or 'no version'}, expected {want_version}.",
            "Look at what uv installed (uv tool list), or go back: fle-studio update --rollback",
        )

    port = configured_port(data_dir)
    if host.platform == "darwin":
        for line in dock_starter.refresh(host, launcher, port):
            say(line)
    code = settle(host, find_services(host, port), port, want_version, say, restart=not opts.no_restart)
    say(f"FLE Studio {want_version} is installed.")
    if not already:
        say("To undo this: fle-studio update --rollback")
    return code


def _where_to_look(service: Service) -> str:
    if service.manager == "launchd":
        try:
            log = plistlib.loads(service.definition.read_bytes()).get("StandardErrorPath")
        except (OSError, ValueError, plistlib.InvalidFileException):
            log = None
        return f"Read its log: {log or '~/Library/Logs'}"
    return f"Read its log: journalctl --user -u {service.name} -n 50"


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="fle-studio update",
        description="Install a new FLE Studio, keep the packages installed next to it, restart the service and "
        "check which version runs.",
    )
    ap.add_argument("--version", metavar="vX.Y.Z", default="", help="release to install (default: the latest)")
    ap.add_argument(
        "--rollback",
        "--zurueck",
        dest="rollback",
        action="store_true",
        help="go back to the version before the last update",
    )
    ap.add_argument("--reinstall", action="store_true", help="install even when that version is already installed")
    ap.add_argument(
        "--drop", metavar="PACKAGE", action="append", default=[], help="leave a package out of the new install"
    )
    ap.add_argument("--no-restart", action="store_true", help="install only; do not restart the service")
    ap.add_argument("--dry-run", action="store_true", help="show the install command and do nothing")
    ap.add_argument("--home", help="data folder (as for fle-studio itself)")
    return ap


def cli(argv: Sequence[str], host: Host | None = None) -> int:
    args = build_parser().parse_args(list(argv))
    if args.home:
        os.environ["FLE_STUDIO_HOME"] = str(Path(args.home).expanduser())
    opts = Options(args.version, args.rollback, args.reinstall, args.dry_run, args.no_restart, tuple(args.drop))
    if opts.rollback and opts.version:
        print("--rollback and --version exclude each other.", file=sys.stderr)
        return 2
    try:
        return update(opts, host or Host(), paths.data_dir())
    except Problem as problem:
        print(f"\nThe update did not work: {problem.text()}", file=sys.stderr)
        return 1

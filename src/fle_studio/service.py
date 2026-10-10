"""A llama-server that runs as a systemd service (Linux).

Such a server belongs to systemd: systemd starts it with the unit's own command, restarts it when the unit says so
and keeps its output in the journal. Connected to a service, FLE Studio reads the unit's state, follows its journal,
shows how far the main process has read the model, and can ask systemd to start, stop or restart the unit. It never
changes the unit.

Starting or stopping a system service needs privileges. FLE Studio first asks systemd itself (enough as root or with
a polkit rule for the account), then ``sudo -n``. It never asks for a password; without either, the buttons say what
to add, for example this sudoers line (``sudo visudo -f /etc/sudoers.d/fle-studio``)::

    alice ALL=(root) NOPASSWD: /usr/bin/systemctl --no-block start rhea.service, \\
        /usr/bin/systemctl --no-block stop rhea.service, /usr/bin/systemctl --no-block restart rhea.service
"""

from __future__ import annotations

import getpass
import json
import os
import re
import shlex
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from . import gguf, paths

ACTIONS = ("start", "stop", "restart")
UNIT_RE = re.compile(r"^[A-Za-z0-9@._:\\-]+\.service$")
PROPS = ("LoadState", "ActiveState", "SubState", "Result", "MainPID", "ExecMainCode", "ExecMainStatus",
         "UnitFileState", "Description", "User", "WorkingDirectory", "ActiveEnterTimestamp", "InactiveEnterTimestamp",
         "InactiveExitTimestamp", "ExecMainStartTimestamp", "ExecStart", "Environment")
# llama-server options FLE Studio reads from the unit's command; long spellings map to the short ones.
OPTIONS = {"-m": "-m", "--model": "-m", "-md": "-md", "--model-draft": "-md", "--host": "--host",
           "--port": "--port", "-a": "-a", "--alias": "-a", "--moe-expert-cache": "--moe-expert-cache",
           "-lv": "-lv", "--verbosity": "-lv", "--log-verbosity": "-lv"}
ENV_OPTIONS = {"LLAMA_ARG_MODEL": "-m", "LLAMA_ARG_MODEL_DRAFT": "-md", "LLAMA_ARG_HOST": "--host",
               "LLAMA_ARG_PORT": "--port", "LLAMA_ARG_ALIAS": "-a", "LLAMA_ARG_MOE_EXPERT_CACHE": "--moe-expert-cache",
               "LLAMA_ARG_LOG_VERBOSITY": "-lv"}
VERBOSE_FLAGS = ("-v", "--verbose", "--log-verbose")  # log everything
# llama.cpp logs its library lines (the FLE expert-cache hit rate among them) at level 4; without -lv it shows up to 3.
DEFAULT_LOG_LEVEL = 3
CACHE_LOG_LEVEL = 4
SECRET_OPTIONS = ("--api-key", "--hf-token", "-hft")  # shown as <hidden>
DENIED = re.compile(r"access denied|interactive authentication required|not authorized|permission denied", re.I)
SUDO_DENIED = re.compile(r"password is required|not allowed to|terminal is required|not in the sudoers|"
                         r"may not run sudo|no askpass", re.I)
# systemd reports Linux signal numbers; Python's signal module only knows the host's own (Windows has no SIGKILL).
LINUX_SIGNALS = dict(enumerate(
    ("SIGHUP SIGINT SIGQUIT SIGILL SIGTRAP SIGABRT SIGBUS SIGFPE SIGKILL SIGUSR1 SIGSEGV SIGUSR2 SIGPIPE SIGALRM "
     "SIGTERM SIGSTKFLT SIGCHLD SIGCONT SIGSTOP SIGTSTP SIGTTIN SIGTTOU SIGURG SIGXCPU SIGXFSZ SIGVTALRM SIGPROF "
     "SIGWINCH SIGIO SIGPWR SIGSYS").split(), start=1))
RESULTS = {
    "exit-code": "exited with an error",
    "signal": "was ended by a signal",
    "core-dump": "crashed and left a core dump",
    "timeout": "did not finish starting or stopping in time",
    "oom-kill": "was killed because the computer ran out of memory",
    "watchdog": "stopped answering its watchdog",
    "start-limit-hit": "failed too often in a row, so systemd gave up (`systemctl reset-failed {unit}` lets it "
                       "start again)",
    "exec-condition": "did not start: its start condition (ExecCondition) said no",
    "condition": "did not start: one of its conditions was not met",
    "resources": "could not get what it needs to run (a missing file, directory or user?)",
    "protocol": "did not keep to its service protocol",
}


class ServiceError(RuntimeError):
    """systemctl refused or failed; the message says what to do about it."""


def available() -> bool:
    """FLE Studio can work with services: Linux with systemd."""
    return paths.IS_LINUX and shutil.which("systemctl") is not None


def unit_name(text: str) -> str:
    name = text.strip()
    if name and not name.endswith(".service"):
        name += ".service"
    if not UNIT_RE.match(name):
        raise ValueError(f"“{text.strip()}” is not the name of a service (like rhea.service).")
    return name


def environment() -> dict[str, str]:
    """For systemctl and journalctl: English messages, no colors, no pager."""
    return dict(os.environ, LC_ALL="C.UTF-8", SYSTEMD_COLORS="0", SYSTEMD_PAGER="", SYSTEMD_LESS="")


def _run(cmd: list[str], timeout: float = 15.0) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=timeout,
                          env=environment(), stdin=subprocess.DEVNULL, check=False)


def _message(r: subprocess.CompletedProcess[str]) -> str:
    return " ".join((r.stderr or r.stdout or "").split())[:600]


_unix_times = True  # systemctl show --timestamp=unix (systemd 248 and later)


@dataclass(frozen=True)
class Unit:
    name: str
    user: bool = False  # a unit of the account's own service manager (systemctl --user)

    def systemctl(self) -> list[str]:
        return ["systemctl", "--user"] if self.user else ["systemctl"]

    def show(self) -> dict[str, str]:
        global _unix_times
        props = [f"--property={p}" for p in PROPS]
        while True:
            r = _run([*self.systemctl(), "show", *(["--timestamp=unix"] if _unix_times else []), *props, self.name])
            if r.returncode != 0 and _unix_times and "timestamp" in (r.stderr or ""):
                _unix_times = False  # an older systemd; timestamps then stay text and are not shown
                continue
            break
        if r.returncode != 0:
            raise ServiceError(_message(r) or f"systemctl show {self.name} failed (code {r.returncode})")
        out: dict[str, str] = {}
        for line in r.stdout.splitlines():
            key, sep, value = line.partition("=")
            if sep and key in PROPS and key not in out:  # the first ExecStart= line is the command
                out[key] = value
        return out

    def journal_command(self, lines: int = 200) -> list[str]:
        which = ["--user-unit", self.name] if self.user else ["--unit", self.name]
        return ["journalctl", *which, "--follow", f"--lines={lines}", "--output=json", "--output-fields=MESSAGE,_PID",
                "--all", "--no-pager"]


def find(name: str) -> tuple[Unit, dict[str, str]]:
    """The unit with this name and its properties: a system service, else one of the account's user services."""
    if not available():
        raise ValueError("Services can be used on Linux with systemd only.")
    name = unit_name(name)
    unit = Unit(name)
    props = unit.show()
    state = props.get("LoadState", "")
    if state == "not-found":
        try:
            mine = Unit(name, user=True)
            user_props = mine.show()
            if user_props.get("LoadState") == "loaded":
                return mine, user_props
        except (ServiceError, OSError, subprocess.SubprocessError):
            pass
        raise ValueError(f"There is no service {name} on this computer.")
    if state == "masked":
        raise ValueError(f"{name} is masked: systemd will not start it.")
    if state != "loaded":
        raise ValueError(f"systemd cannot load {name} ({state or 'unknown state'}).")
    return unit, props


# ------------------------------------------------------------------------------------------------ the command


def argv(props: dict[str, str]) -> list[str]:
    """The unit's command, from `ExecStart={ path=… ; argv[]=… ; ignore_errors=… }`."""
    m = re.search(r"argv\[\]=(.*?) ; ignore_errors=", props.get("ExecStart", ""))
    if not m:
        return []
    try:
        return shlex.split(m.group(1))
    except ValueError:
        return m.group(1).split()


def options(props: dict[str, str]) -> dict[str, str]:
    """Model, draft model, host, port, alias, expert-cache slots and log level as the command (or the unit's
    LLAMA_ARG_* variables) set them. -v counts as the highest log level."""
    found: dict[str, str] = {}
    try:
        env = shlex.split(props.get("Environment", ""))
    except ValueError:
        env = props.get("Environment", "").split()
    for item in env:
        key, sep, value = item.partition("=")
        if sep and key in ENV_OPTIONS:
            found[ENV_OPTIONS[key]] = value
    args = argv(props)
    i = 1
    while i < len(args):
        flag, eq, value = args[i].partition("=") if args[i].startswith("--") else (args[i], "", "")
        if flag in VERBOSE_FLAGS:
            found["-lv"] = "99"
        elif flag in OPTIONS:
            if eq:
                found[OPTIONS[flag]] = value
            elif i + 1 < len(args):
                found[OPTIONS[flag]] = args[i + 1]
                i += 1
        i += 1
    return found


def address(props: dict[str, str]) -> str:
    """http://host:port of the server, or "" when the command does not tell (a wrapper script, a socket)."""
    opts = options(props)
    args = argv(props)
    if "--port" not in opts and not (args and "llama-server" in Path(args[0]).name):
        return ""
    host = opts.get("--host", "127.0.0.1")
    if host.endswith(".sock"):
        return ""
    if host in ("", "0.0.0.0", "::", "[::]"):
        host = "127.0.0.1"
    if ":" in host and not host.startswith("["):
        host = f"[{host}]"
    try:
        port = int(opts.get("--port", "8080"))
    except ValueError:
        return ""
    return f"http://{host}:{port}"


def _number(value: str | None, default: int) -> int:
    try:
        return int(value) if value else default
    except ValueError:
        return default


def model_files(props: dict[str, str]) -> list[Path]:
    opts = options(props)
    workdir = props.get("WorkingDirectory", "").lstrip("-!")
    files: list[Path] = []
    for flag in ("-m", "-md"):
        if not opts.get(flag):
            continue
        path = Path(opts[flag])
        if not path.is_absolute() and workdir.startswith("/"):
            path = Path(workdir) / path
        files.extend(gguf.split_parts(path) if flag == "-m" else [path])
    return files


def model_bytes(props: dict[str, str]) -> int:
    total = 0
    for path in model_files(props):
        try:
            total += path.stat().st_size
        except OSError:
            pass
    return total


def command_line(props: dict[str, str]) -> str:
    """The command for display, with API keys and tokens replaced by <hidden>."""
    out: list[str] = []
    hide = False
    for arg in argv(props):
        if hide:
            out.append("<hidden>")
            hide = False
            continue
        flag, eq, _ = arg.partition("=")
        if flag in SECRET_OPTIONS:
            if eq:
                out.append(f"{flag}=<hidden>")
                continue
            hide = True
        out.append(shlex.quote(arg))
    return " ".join(out)


# ------------------------------------------------------------------------------------------------ state


def timestamp(value: str | None) -> float | None:
    """Seconds since the epoch from systemctl's `@1791629250` form; None when unset or in the text form."""
    if value and value.startswith("@"):
        try:
            return float(value[1:])
        except ValueError:
            return None
    return None


def main_pid(props: dict[str, str]) -> int:
    value = props.get("MainPID", "0")
    return int(value) if value.isdigit() else 0


def failure(unit: Unit, props: dict[str, str]) -> str:
    """What went wrong in the unit's last run, or "" when nothing did."""
    result = props.get("Result", "")
    if result in ("", "success"):
        return ""
    what = RESULTS.get(result, f"stopped ({result})").format(unit=unit.name)
    code, status = props.get("ExecMainCode", ""), props.get("ExecMainStatus", "")
    detail = ""
    if code == "1" and status not in ("", "0"):
        detail = f" (exit code {status})"
    elif code in ("2", "3") and status.isdigit():
        name = LINUX_SIGNALS.get(int(status))
        detail = f" ({name})" if name else f" (signal {status})"
    return f"{unit.name} {what}{detail}."


def summary(unit: Unit, props: dict[str, str]) -> dict[str, Any]:
    active = props.get("ActiveState", "")
    since = {"active": "ActiveEnterTimestamp", "activating": "InactiveExitTimestamp",
             "reloading": "ActiveEnterTimestamp", "deactivating": "ActiveEnterTimestamp"}.get(
        active, "InactiveEnterTimestamp")
    opts = options(props)
    expert_cache = _number(opts.get("--moe-expert-cache"), 0) > 0
    log_level = _number(opts.get("-lv"), DEFAULT_LOG_LEVEL)
    return {
        "unit": unit.name,
        "user": unit.user,
        "description": props.get("Description", ""),
        "active": active,
        "sub": props.get("SubState", ""),
        "result": props.get("Result", ""),
        "pid": main_pid(props),
        "enabled": props.get("UnitFileState", ""),
        "runs_as": props.get("User", "") or ("you" if unit.user else "root"),
        "since": timestamp(props.get(since)),
        "command": command_line(props),
        "model": opts.get("-m", ""),
        "alias": opts.get("-a", ""),
        # the FLE expert cache is on, and whether the journal carries its hit rate (log level 4 and up)
        "expert_cache": expert_cache,
        "log_level": log_level,
        "cache_in_log": expert_cache and log_level >= CACHE_LOG_LEVEL,
    }


# ------------------------------------------------------------------------------------------------ control


def permission_help(unit: Unit) -> str:
    systemctl = shutil.which("systemctl") or "/usr/bin/systemctl"
    rule = ", ".join(f"{systemctl} --no-block {a} {unit.name}" for a in ACTIONS)
    try:
        user = getpass.getuser()
    except (KeyError, OSError):
        user = "youruser"
    return (f"FLE Studio may not start or stop {unit.name}. Allow it without a password, for example with this "
            f"line in /etc/sudoers.d/fle-studio (edit it with: sudo visudo -f /etc/sudoers.d/fle-studio): "
            f"{user} ALL=(root) NOPASSWD: {rule}")


def act(unit: Unit, action: str) -> str:
    """Ask systemd to start, stop or restart the unit, without waiting for it. Returns "systemctl" or "sudo"."""
    if action not in ACTIONS:
        raise ValueError(f"unknown service action {action!r}")
    tail = ["--no-block", action, unit.name]
    if unit.user:
        r = _run(["systemctl", "--user", *tail])
        if r.returncode == 0:
            return "systemctl"
        raise ServiceError(_message(r) or f"systemctl --user could not {action} {unit.name}")
    r = _run(["systemctl", "--no-ask-password", *tail])
    if r.returncode == 0:
        return "systemctl"
    first = _message(r)
    if not DENIED.search(first) or getattr(os, "geteuid", lambda: -1)() == 0:
        raise ServiceError(first or f"systemctl could not {action} {unit.name}")
    if shutil.which("sudo"):
        r = _run(["sudo", "-n", "systemctl", *tail])
        if r.returncode == 0:
            return "sudo"
        second = _message(r)
        if second and not SUDO_DENIED.search(second):
            raise ServiceError(second)
    raise ServiceError(permission_help(unit))


def journal_entry(raw: bytes) -> tuple[str, int, float] | None:
    """(message, PID, time) from one line of `journalctl --output=json`; None for anything else journalctl says."""
    try:
        entry = json.loads(raw)
    except ValueError:
        return None
    if not isinstance(entry, dict):
        return None
    message = entry.get("MESSAGE")
    if isinstance(message, list):  # not valid UTF-8: journalctl sends the bytes
        message = bytes(b & 0xFF for b in message if isinstance(b, int)).decode("utf-8", "replace")
    if not isinstance(message, str):
        message = ""
    try:
        pid = int(entry.get("_PID") or 0)
    except (TypeError, ValueError):
        pid = 0
    try:
        when = int(entry.get("__REALTIME_TIMESTAMP") or 0) / 1e6
    except (TypeError, ValueError):
        when = 0.0
    return message.rstrip("\r\n"), pid, when

#!/usr/bin/env python3
"""Stand-ins for systemctl, journalctl and sudo, for the service tests (POSIX only)::

    python fake_systemd.py systemctl|journalctl|sudo ARGS...

The state lives in $FAKE_SYSTEMD_DIR: one <unit>.json per unit (properties as `systemctl show` prints them, plus
"argv" and "env" for the command) and one <unit>.journal with JSON lines in the shape of `journalctl --output=json`.
`start` runs the unit's command through a detached runner that records the main PID and the state changes and
writes every output line to the journal, as systemd and journald do.

Knobs: FAKE_SYSTEMCTL_DENY=1 refuses start/stop/restart without "root" (like polkit without a rule);
FAKE_SUDO_DENY=1 makes sudo ask for a password. sudo runs its command with FAKE_AS_ROOT=1.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

HOME = Path(os.environ.get("FAKE_SYSTEMD_DIR", "."))
DEFAULTS = {
    "LoadState": "loaded",
    "ActiveState": "inactive",
    "SubState": "dead",
    "Result": "success",
    "MainPID": "0",
    "ExecMainCode": "0",
    "ExecMainStatus": "0",
    "UnitFileState": "disabled",
    "Description": "",
    "User": "",
    "WorkingDirectory": "",
    "ActiveEnterTimestamp": "",
    "InactiveEnterTimestamp": "",
    "InactiveExitTimestamp": "",
    "ExecMainStartTimestamp": "",
}


def load(name: str) -> dict[str, Any] | None:
    try:
        return json.loads((HOME / f"{name}.json").read_text())
    except (OSError, ValueError):
        return None


def update(name: str, **props: str) -> None:
    unit = load(name) or {}
    unit.update(props)
    tmp = HOME / f"{name}.json.tmp"
    tmp.write_text(json.dumps(unit))
    tmp.replace(HOME / f"{name}.json")


def journal(name: str, message: str, pid: int = 1) -> None:
    entry = {"MESSAGE": message, "_PID": str(pid), "__REALTIME_TIMESTAMP": str(int(time.time() * 1e6))}
    with (HOME / f"{name}.journal").open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")


def stamp() -> str:
    return f"@{int(time.time())}"


# ------------------------------------------------------------------------------------------------ systemctl


def systemctl(args: list[str]) -> int:
    props = [a.split("=", 1)[1] for a in args if a.startswith("--property=")]
    flags = {a for a in args if a.startswith("--") and not a.startswith("--property=")}
    verb, name = [a for a in args if not a.startswith("--")][:2]
    controls = verb in ("start", "stop", "restart")
    if controls and os.environ.get("FAKE_SYSTEMCTL_DENY") == "1" and os.environ.get("FAKE_AS_ROOT") != "1":
        sys.stderr.write(f"Failed to {verb} {name}: Access denied\n")
        return 1
    unit = load(name)
    if verb == "show":
        for prop in props:
            if unit is None:
                value = {"LoadState": "not-found", "ActiveState": "inactive", "SubState": "dead"}.get(prop, "")
            elif prop == "ExecStart":
                argv = unit["argv"]
                value = (
                    f"{{ path={argv[0]} ; argv[]={' '.join(argv)} ; ignore_errors=no ; start_time=[n/a] ; "
                    "stop_time=[n/a] ; pid=0 ; code=(null) ; status=0/0 }"
                )
            elif prop == "Environment":
                value = " ".join(f"{k}={v}" for k, v in unit.get("env", {}).items())
            else:
                value = str(unit.get(prop, DEFAULTS.get(prop, "")))
            print(f"{prop}={value}")
        return 0
    if unit is None:
        sys.stderr.write(f"Failed to {verb} {name}: Unit {name} not found.\n")
        return 5
    if verb in ("stop", "restart"):
        stop(name, wait=verb == "restart" or "--no-block" not in flags)
    if verb in ("start", "restart"):
        start(name)
    return 0


def start(name: str) -> None:
    if (load(name) or {}).get("ActiveState") in ("active", "activating"):
        return
    update(name, ActiveState="activating", SubState="start", InactiveExitTimestamp=stamp(), Result="success")
    subprocess.Popen(
        [sys.executable, __file__, "run", name],
        start_new_session=True,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def stop(name: str, wait: bool) -> None:
    pid = int((load(name) or {}).get("MainPID") or 0)
    if not pid:
        return
    update(name, ActiveState="deactivating", SubState="stop-sigterm")
    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    for _ in range(400):
        if not wait or (load(name) or {}).get("ActiveState") in ("inactive", "failed"):
            return
        time.sleep(0.05)


def run(name: str) -> None:
    """The unit's main process under 'systemd': record it, journal its output, record how it ended."""
    unit = load(name) or {}
    argv = unit["argv"]
    cmd = [sys.executable, *argv] if argv[0].endswith(".py") else argv
    child = subprocess.Popen(
        cmd,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env=dict(os.environ, **unit.get("env", {})),
    )
    update(
        name,
        ActiveState="active",
        SubState="running",
        MainPID=str(child.pid),
        ExecMainStartTimestamp=stamp(),
        ActiveEnterTimestamp=stamp(),
    )
    journal(name, f"Started {name} - {unit.get('Description') or name}.")
    assert child.stdout is not None
    for raw in iter(child.stdout.readline, b""):
        line = raw.decode("utf-8", "replace").rstrip("\r\n")
        if line:
            journal(name, line, child.pid)
    rc = child.wait()
    code, status = ("2", str(-rc)) if rc < 0 else ("1", str(rc))
    if (load(name) or {}).get("ActiveState") == "deactivating" or rc == 0:
        update(
            name,
            ActiveState="inactive",
            SubState="dead",
            MainPID="0",
            Result="success",
            ExecMainCode=code,
            ExecMainStatus=status,
            InactiveEnterTimestamp=stamp(),
        )
        journal(name, f"Stopped {name}.")
        return
    result = "signal" if rc < 0 else "exit-code"
    update(
        name,
        ActiveState="failed",
        SubState="failed",
        MainPID="0",
        Result=result,
        ExecMainCode=code,
        ExecMainStatus=status,
        InactiveEnterTimestamp=stamp(),
    )
    journal(name, f"{name}: Main process exited, code={'killed' if rc < 0 else 'exited'}, status={status}")
    journal(name, f"{name}: Failed with result '{result}'.")


# ------------------------------------------------------------------------------------------------ journalctl, sudo


def journalctl(args: list[str]) -> int:
    name, lines, follow = "", 10, False
    for i, arg in enumerate(args):
        if arg in ("--unit", "--user-unit", "-u") and i + 1 < len(args):
            name = args[i + 1]
        elif arg.startswith("--lines="):
            lines = int(arg.split("=", 1)[1])
        elif arg in ("--follow", "-f"):
            follow = True
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
    path = HOME / f"{name}.journal"
    try:
        data = path.read_bytes()
    except OSError:
        data = b""
    backlog = data.splitlines(keepends=True)
    sys.stdout.buffer.write(b"".join(backlog[-lines:] if lines else []))
    sys.stdout.buffer.flush()
    pos = len(data)
    while follow:
        try:
            with path.open("rb") as f:
                f.seek(pos)
                chunk = f.read()
        except OSError:
            chunk = b""
        whole = chunk[: chunk.rfind(b"\n") + 1]
        if whole:
            sys.stdout.buffer.write(whole)
            sys.stdout.buffer.flush()
            pos += len(whole)
        time.sleep(0.05)
    return 0


def sudo(args: list[str]) -> int:
    if os.environ.get("FAKE_SUDO_DENY") == "1":
        sys.stderr.write("sudo: a password is required\n")
        return 1
    if args[:1] == ["-n"]:
        args = args[1:]
    return subprocess.call(args, env=dict(os.environ, FAKE_AS_ROOT="1"))


def main(argv: list[str]) -> int:
    tool, args = argv[0], argv[1:]
    if tool == "run":
        run(args[0])
        return 0
    return {"systemctl": systemctl, "journalctl": journalctl, "sudo": sudo}[tool](args)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

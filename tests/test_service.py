"""Service mode: a llama-server that runs as a systemd unit, read and controlled through systemctl and journalctl.

The parsing tests use the shapes systemd 255 prints. The lifecycle tests put fake systemctl, journalctl and sudo
(tests/fake_systemd.py) on PATH; `start` then really runs tests/fake_llama_server.py as the unit's main process.
"""

from __future__ import annotations

import json
import os
import signal
import socket
import sys
import time
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import pytest

from fle_studio import service
from fle_studio.engine import Engine, EngineBusy
from fle_studio.launch import Plan

HERE = Path(__file__).parent
FAKE_SERVER = HERE / "fake_llama_server.py"
FAKE_SYSTEMD = HERE / "fake_systemd.py"

RHEA = {  # systemctl show rhea.service on the reference machine (shortened), plus an API key to hide
    "ActiveState": "inactive", "SubState": "dead", "Result": "success", "MainPID": "0", "User": "sirfyyn",
    "ExecStart": "{ path=/mnt/fastmodels/rhea-prod-20261007-p51/bin/llama-server ; argv[]=/mnt/fastmodels/"
                 "rhea-prod-20261007-p51/bin/llama-server -m /mnt/fastmodels/rhea-hybrid-m2/G3.gguf --host 0.0.0.0 "
                 "--port 18420 --alias rhea --parallel 2 --api-key sk-secret -md /mnt/fastmodels/mtp.gguf -t 16 "
                 "-C 0x0000FFFF --cpu-strict 1 ; ignore_errors=no ; start_time=[n/a] ; stop_time=[n/a] ; pid=0 ; "
                 "code=(null) ; status=0/0 }",
    "ActiveEnterTimestamp": "@1791629250", "InactiveEnterTimestamp": "@1791629250",
}


# ------------------------------------------------------------------------------------------------ parsing


def test_the_command_gives_address_model_and_alias() -> None:
    assert service.address(RHEA) == "http://127.0.0.1:18420"
    opts = service.options(RHEA)
    assert opts["-m"] == "/mnt/fastmodels/rhea-hybrid-m2/G3.gguf"
    assert opts["-md"] == "/mnt/fastmodels/mtp.gguf"
    assert opts["-a"] == "rhea"


def test_api_keys_never_reach_the_page() -> None:
    line = service.command_line(RHEA)
    assert "sk-secret" not in line
    assert "--api-key <hidden>" in line
    assert "sk-secret" not in json.dumps(service.summary(service.Unit("rhea.service"), RHEA))
    assert "<hidden>" in service.command_line({"ExecStart": "{ path=/x ; argv[]=/x/llama-server --api-key=abc ; "
                                                             "ignore_errors=no }"})


def test_address_from_environment_or_none() -> None:
    wrapper = {"ExecStart": "{ path=/usr/local/bin/start.sh ; argv[]=/usr/local/bin/start.sh ; ignore_errors=no }"}
    assert service.address(wrapper) == ""
    by_env = {**wrapper, "Environment": "LLAMA_ARG_HOST=0.0.0.0 LLAMA_ARG_PORT=9000 HF_TOKEN=x"}
    assert service.address(by_env) == "http://127.0.0.1:9000"
    plain = {"ExecStart": "{ path=/opt/llama-server ; argv[]=/opt/llama-server -m m.gguf ; ignore_errors=no }"}
    assert service.address(plain) == "http://127.0.0.1:8080"  # llama-server's default port


def test_failures_in_words() -> None:
    unit = service.Unit("rhea.service")
    assert service.failure(unit, {"Result": "success"}) == ""
    killed = service.failure(unit, {"Result": "signal", "ExecMainCode": "2", "ExecMainStatus": "9"})
    assert killed == "rhea.service was ended by a signal (SIGKILL)."
    exited = service.failure(unit, {"Result": "exit-code", "ExecMainCode": "1", "ExecMainStatus": "134"})
    assert exited == "rhea.service exited with an error (exit code 134)."
    assert "ExecCondition" in service.failure(unit, {"Result": "exec-condition"})


def test_names_times_and_journal_lines() -> None:
    assert service.unit_name("rhea") == "rhea.service"
    assert service.unit_name(" llama@2.service ") == "llama@2.service"
    with pytest.raises(ValueError):
        service.unit_name("rm -rf")
    assert service.timestamp("@1791629250") == 1791629250.0
    assert service.timestamp("Sat 2026-10-10 12:47:30 CEST") is None
    assert service.timestamp("") is None
    raw = json.dumps({"MESSAGE": [104, 105, 255], "_PID": "42", "__REALTIME_TIMESTAMP": "1791629250768762"})
    assert service.journal_entry(raw.encode()) == ("hi�", 42, 1791629250.768762)
    assert service.journal_entry(b"Hint: You are currently not seeing messages from other users.") is None


# ------------------------------------------------------------------------------------------------ fake systemd


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def wait_for(probe: Callable[[], Any], what: str, timeout: float = 30.0) -> Any:
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        got = probe()
        if got:
            return got
        time.sleep(0.1)
    raise AssertionError(f"timed out waiting for {what}")


class FakeSystemd:
    def __init__(self, home: Path) -> None:
        self.home = home

    def add(self, name: str, argv: list[str], env: dict[str, str] | None = None, **props: str) -> None:
        unit = {"LoadState": "loaded", "ActiveState": "inactive", "SubState": "dead", "Description": "fake server",
                "User": "tester", "UnitFileState": "enabled", "argv": argv, "env": env or {}, **props}
        (self.home / f"{name}.json").write_text(json.dumps(unit))

    def unit(self, name: str) -> dict[str, Any]:
        return json.loads((self.home / f"{name}.json").read_text())

    def journal(self, name: str, message: str, pid: int, when: float) -> None:
        entry = {"MESSAGE": message, "_PID": str(pid), "__REALTIME_TIMESTAMP": str(int(when * 1e6))}
        with (self.home / f"{name}.journal").open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")


@pytest.fixture
def systemd(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[FakeSystemd]:
    if os.name == "nt":
        pytest.skip("systemd services exist on Linux only; the fakes need a POSIX shell")
    home = tmp_path / "systemd"
    bindir = tmp_path / "bin"
    home.mkdir()
    bindir.mkdir()
    for tool in ("systemctl", "journalctl", "sudo"):
        script = bindir / tool
        script.write_text(f'#!/bin/sh\nexec "{sys.executable}" "{FAKE_SYSTEMD}" {tool} "$@"\n')
        script.chmod(0o755)
    monkeypatch.setenv("PATH", f"{bindir}{os.pathsep}{os.environ.get('PATH', '')}")
    monkeypatch.setenv("FAKE_SYSTEMD_DIR", str(home))
    monkeypatch.setattr(service, "available", lambda: True)  # also on the macOS test runner
    fake = FakeSystemd(home)
    yield fake
    for path in home.glob("*.json"):  # never leave a fake server behind
        pid = int(json.loads(path.read_text()).get("MainPID") or 0)
        if pid:
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass


@pytest.fixture
def engine(tmp_path: Path) -> Iterator[Engine]:
    eng = Engine(tmp_path / "data")
    yield eng
    eng.stop()


def server_unit(fake: FakeSystemd, tmp_path: Path, name: str = "fake.service", **env: str) -> int:
    port = free_port()
    model = tmp_path / "fake-model.gguf"
    model.write_bytes(b"GGUF" + bytes(8192))
    fake.add(name, [str(FAKE_SERVER), "-m", str(model), "--host", "0.0.0.0", "--port", str(port), "--alias", "fake"],
             env={"FAKE_LOAD_SECONDS": "0.6", **env})
    return port


def settle(engine: Engine, want: str, timeout: float = 30.0) -> dict[str, Any]:
    """Re-read the unit like the live sampler does, until the engine reaches `want`."""
    def probe() -> dict[str, Any] | None:
        engine.refresh_attached()
        st = engine.status()
        return st if st["state"] == want else None
    return wait_for(probe, f"state {want}", timeout)


def log_text(engine: Engine) -> str:
    return "\n".join(line["text"] for line in engine.log_since(0)["lines"])


# ------------------------------------------------------------------------------------------------ lifecycle


def test_attach_start_ready_stop(systemd: FakeSystemd, engine: Engine, tmp_path: Path) -> None:
    port = server_unit(systemd, tmp_path)
    st = engine.attach("", "", "fake")  # the address comes from the unit's command
    assert st["service"]["unit"] == "fake.service"
    assert st["state"] == "stopped" and st["attached"]
    assert st["model"]["name"] == "fake"
    assert engine.base_url == f"http://127.0.0.1:{port}"
    plan = Plan(engine="custom", binary="llama-server", args=[], env={}, port=free_port(), model={})
    with pytest.raises(EngineBusy, match="connected to fake.service"):
        engine.start(plan)

    st = engine.service_action("start")
    assert st["state"] == "starting"
    st = settle(engine, "attached")
    assert st["reachable"] and st["service"]["active"] == "active"
    main_pid = int(systemd.unit("fake.service")["MainPID"])
    assert main_pid and st["service"]["pid"] == main_pid == st["pid"]
    assert st["load"]["total"] == (tmp_path / "fake-model.gguf").stat().st_size
    wait_for(lambda: "listening on" in log_text(engine), "the server's log in the journal")
    assert "Started fake.service" in log_text(engine)

    engine.service_action("stop")
    wait_for(lambda: systemd.unit("fake.service")["ActiveState"] == "inactive", "systemd to stop the unit")
    engine.refresh_attached()
    st = engine.status()  # once systemd says the stop is done, the page says so too, without a grace period
    assert st["state"] == "stopped", st["state"]
    assert st["service"]["active"] == "inactive" and not st["error"]
    engine.stop()  # disconnect: FLE Studio lets go of the unit and its journal
    assert engine.unit is None and engine._journal is None and engine.status()["service"] is None


def test_a_crash_reads_like_systemd_says_it(systemd: FakeSystemd, engine: Engine, tmp_path: Path) -> None:
    server_unit(systemd, tmp_path, FAKE_CRASH_AFTER="0.3")
    engine.attach("", "", "fake.service")
    engine.service_action("start")
    st = settle(engine, "crashed")
    assert st["error"] == "fake.service exited with an error (exit code 134)."
    assert st["exit_code"] == 134


def test_earlier_runs_fill_the_log_but_not_the_numbers(systemd: FakeSystemd, engine: Engine,
                                                       tmp_path: Path) -> None:
    server_unit(systemd, tmp_path)
    old = time.time() - 3600
    systemd.journal("fake.service", "moe-cache: steps=512 hits=900 misses=100 hit-rate=90.0 %", 4242, old)
    engine.attach("", "", "fake.service")
    wait_for(lambda: "hit-rate=90.0" in log_text(engine), "the backlog")
    assert engine.cache == {}
    systemd.journal("fake.service", "moe-cache: steps=1024 hits=800 misses=200 hit-rate=80.0 %", 4243, time.time())
    wait_for(lambda: engine.cache.get("hit_rate") == 80.0, "a fresh line to count")


def test_without_permission_sudo_steps_in(systemd: FakeSystemd, engine: Engine, tmp_path: Path,
                                          monkeypatch: pytest.MonkeyPatch) -> None:
    server_unit(systemd, tmp_path)
    monkeypatch.setenv("FAKE_SYSTEMCTL_DENY", "1")
    engine.attach("", "", "fake.service")
    engine.service_action("start")
    assert "asked systemd to start fake.service (through sudo)" in log_text(engine)
    settle(engine, "attached")

    monkeypatch.setenv("FAKE_SUDO_DENY", "1")
    with pytest.raises(service.ServiceError) as err:
        engine.service_action("stop")
    assert "NOPASSWD" in str(err.value) and "--no-block stop fake.service" in str(err.value)
    assert engine.status()["state"] == "attached"  # nothing changed


def test_unknown_or_unsupported(systemd: FakeSystemd, engine: Engine, monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.raises(ValueError, match="There is no service nothere.service"):
        engine.attach("", "", "nothere")
    assert engine.unit is None
    monkeypatch.setattr(service, "available", lambda: False)
    with pytest.raises(ValueError, match="Linux with systemd"):
        engine.attach("", "", "fake.service")
    with pytest.raises(ValueError, match="not connected to a service"):
        engine.service_action("start")

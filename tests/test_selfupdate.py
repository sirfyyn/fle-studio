"""`fle-studio update`: receipt, install command, service restart, version check, error texts, rollback.

Nothing here touches the real uv, launchd, systemd or the network: `Machine` stands in for all of it (the `Host` of
selfupdate.py) and keeps a uv receipt on disk that its fake `uv tool install` rewrites, like the real one does.
"""

from __future__ import annotations

import os
import plistlib
import stat
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest

from fle_studio import dock_starter, selfupdate
from fle_studio.selfupdate import Host, Options, Problem, Requirement, Result
from fle_studio.versions import same_version

GIT = "https://github.com/sirfyyn/fle-studio"
# What uv 0.13 wrote on the Mac on 2026-10-27 (the plugin path is shortened).
MAC_RECEIPT = f"""[tool]
requirements = [
    {{ name = "fle-studio", git = "{GIT}?rev=v0.4.0" }},
    {{ name = "fle-studio-ginny", directory = "/Users/me/fle-studio-ginny" }},
]
python = "3.12"
entrypoints = [
    {{ name = "fle-studio", install-path = "/Users/me/.local/bin/fle-studio", from = "fle-studio" }},
]
"""


# ------------------------------------------------------------------------------------------------ receipt


def test_the_receipt_of_the_mac_gives_studio_and_the_plugin() -> None:
    r = selfupdate.parse_receipt(MAC_RECEIPT)
    assert r.python == "3.12" and r.launcher == "/Users/me/.local/bin/fle-studio"
    assert r.main == Requirement("fle-studio", "git", GIT, (), "v0.4.0")
    assert r.extras == (Requirement("fle-studio-ginny", "directory", "/Users/me/fle-studio-ginny"),)


def test_receipt_variants_of_other_uv_versions_and_sources() -> None:
    text = (
        "[tool]\nrequirements = [\n"
        f'    {{ name = "fle-studio", extras = ["pdf"], git = "{GIT}", rev = "v0.3.2" }},\n'
        '    { name = "a", editable = "/src/a" },\n'
        '    { name = "b", path = "/tmp/b-1.0-py3-none-any.whl" },\n'
        '    { name = "c", url = "https://example.org/c.whl" },\n'
        '    { name = "d", specifier = ">=1.2", extras = ["x", "y"] },\n'
        '    { name = "e", git = "https://example.org/e?subdirectory=pkg&tag=v2" },\n'
        ']\npython = "3.11"\n'
    )
    r = selfupdate.parse_receipt(text)
    assert r.main == Requirement("fle-studio", "git", GIT, ("pdf",), "v0.3.2")
    kinds = {x.name: (x.kind, x.where) for x in r.extras}
    assert kinds == {
        "a": ("editable", "/src/a"),
        "b": ("path", "/tmp/b-1.0-py3-none-any.whl"),
        "c": ("url", "https://example.org/c.whl"),
        "d": ("index", ">=1.2"),
        "e": ("git", "https://example.org/e"),
    }
    assert next(x for x in r.extras if x.name == "d").extras == ("x", "y")
    e = next(x for x in r.extras if x.name == "e")
    assert (e.rev, e.subdirectory) == ("v2", "pkg")
    assert r.python == "3.11" and r.launcher == ""


def test_the_private_index_of_a_plugin_is_kept() -> None:
    """uv keeps --index-url in [tool.options]; a plugin from a private index could not be found again without it."""
    text = (
        f'[tool]\nrequirements = [\n    {{ name = "fle-studio", git = "{GIT}?rev=v0.4.0" }},\n'
        '    { name = "fle-studio-ginny" },\n]\npython = "3.12"\n\n'
        '[tool.options]\nindex-url = "https://git.example.org/pypi/simple"\n'
        'extra-index-url = ["https://a.example.org/simple", "https://b.example.org/simple"]\n'
    )
    r = selfupdate.parse_receipt(text)
    assert r.options == (
        "--index-url",
        "https://git.example.org/pypi/simple",
        "--extra-index-url",
        "https://a.example.org/simple",
        "--extra-index-url",
        "https://b.example.org/simple",
    )
    assert r.main is not None
    cmd = selfupdate.install_command(["uv"], r.main, r.extras, r.python, "v0.4.1", r.options)
    assert cmd[:7] == ["uv", "tool", "install", "--python", "3.12", "--force", "--index-url"]
    assert cmd[-3:] == ["--with", "fle-studio-ginny", f"git+{GIT}@v0.4.1"]


def test_the_index_survives_the_record_for_rollback(tmp_path: Path) -> None:
    r = selfupdate.parse_receipt(MAC_RECEIPT + '\n[tool.options]\nindex-url = "https://x/simple"\n')
    selfupdate.save_previous(tmp_path, r, "0.4.0")
    assert selfupdate.load_previous(tmp_path)[1].options == ("--index-url", "https://x/simple")


def test_a_receipt_without_requirements_has_no_studio() -> None:
    assert selfupdate.parse_receipt("").main is None
    assert selfupdate.parse_receipt("[tool]\npython = '3.12'\n").requirements == ()


def test_windows_launcher_path_in_the_receipt() -> None:
    text = (
        'requirements = [{ name = "fle-studio", git = "https://x" }]\n'
        'entrypoints = [{ name = "fle-studio", install-path = "C:\\\\Users\\\\me\\\\fle-studio.exe" }]\n'
    )
    assert selfupdate.parse_receipt(text).launcher == "C:\\Users\\me\\fle-studio.exe"


# ------------------------------------------------------------------------------------------------ install command


def test_install_command_gives_the_plugin_back() -> None:
    r = selfupdate.parse_receipt(MAC_RECEIPT)
    assert r.main is not None
    cmd = selfupdate.install_command(["uv"], r.main, r.extras, r.python, "v0.4.1")
    assert cmd == [
        "uv",
        "tool",
        "install",
        "--python",
        "3.12",
        "--force",
        "--with",
        "/Users/me/fle-studio-ginny",
        f"git+{GIT}@v0.4.1",
    ]


@pytest.mark.parametrize(
    ("req", "expected"),
    [
        (Requirement("p", "directory", "/src/p"), ["--with", "/src/p"]),
        (Requirement("p", "editable", "/src/p"), ["--with-editable", "/src/p"]),
        (Requirement("p", "path", "/tmp/p.whl"), ["--with", "/tmp/p.whl"]),
        (Requirement("p", "url", "https://x/p.whl"), ["--with", "p @ https://x/p.whl"]),
        (Requirement("p", "index", ">=1.2", ("x",)), ["--with", "p[x]>=1.2"]),
        (
            Requirement("p", "git", "https://x/p", (), "v2", "pkg"),
            ["--with", "p @ git+https://x/p@v2#subdirectory=pkg"],
        ),
    ],
)
def test_every_kind_of_package_comes_back(req: Requirement, expected: list[str]) -> None:
    assert selfupdate.with_args(req) == expected


def test_a_folder_package_with_extras_goes_back_as_a_file_address(tmp_path: Path) -> None:
    """A real absolute path of this platform: `/src/p` is not absolute on Windows."""
    folder = tmp_path / "p"
    assert selfupdate.with_args(Requirement("p", "directory", str(folder), ("x",))) == [
        "--with",
        f"p[x] @ {folder.as_uri()}",
    ]


def test_studio_with_its_own_extras_keeps_them() -> None:
    main = Requirement("fle-studio", "git", GIT, ("pdf",), "v0.4.0")
    assert selfupdate.main_spec(main, "v0.4.1") == f"fle-studio[pdf] @ git+{GIT}@v0.4.1"
    assert selfupdate.main_spec(main) == f"fle-studio[pdf] @ git+{GIT}@v0.4.0"  # no tag: the revision it has


def test_studio_from_a_folder_has_no_release_to_move_to() -> None:
    with pytest.raises(Problem) as err:
        selfupdate.main_spec(Requirement("fle-studio", "directory", "/src/fle-studio"), "v0.4.1")
    assert "not from GitHub" in err.value.cause and "/src/fle-studio" in err.value.remedy


def test_packages_whose_folder_is_gone_are_found(tmp_path: Path) -> None:
    here, gone = tmp_path / "here", tmp_path / "gone"
    here.mkdir()
    reqs = [
        Requirement("a", "directory", str(here)),
        Requirement("b", "directory", str(gone)),
        Requirement("c", "git", "https://x/c"),
    ]
    assert [r.name for r in selfupdate.missing_extras(reqs)] == ["b"]


# ------------------------------------------------------------------------------------------------ versions


def test_installed_and_running_version_are_compared_as_versions() -> None:
    assert same_version("v0.4.1", "0.4.1") and same_version("0.4.1", "0.4.1.0")
    assert not same_version("0.4.0", "0.4.1") and not same_version("", "0.4.1") and not same_version("main", "main")


@pytest.mark.parametrize("text", ["v0.4.1", "0.4.1", "v0.5.0rc1"])
def test_release_names_are_taken(text: str) -> None:
    assert selfupdate.normalise_tag(text) == (text if text.startswith("v") else f"v{text}")


@pytest.mark.parametrize("text", ["main", "v0.4.1; rm -rf /", "../x", "", "latest"])
def test_other_names_never_reach_a_command_line(text: str) -> None:
    with pytest.raises(Problem):
        selfupdate.normalise_tag(text)


# ------------------------------------------------------------------------------------------------ the stand-in machine


class Machine:
    """Stands in for the machine: uv, launcher, launchd/systemd, codesign and the running server."""

    def __init__(self, tmp: Path, running: str | None = "0.4.1", installed: str = "0.4.1") -> None:
        self.tmp = tmp
        self.uv_dir = tmp / "uv-tools"
        self.env = self.uv_dir / "fle-studio"
        self.bin = tmp / "bin"
        self.launcher = self.bin / "fle-studio"
        self.plugin = tmp / "fle-studio-ginny"
        self.data = tmp / "data"
        self.apps = tmp / "Applications"
        for folder in (self.env, self.bin, self.plugin, self.data, self.apps):
            folder.mkdir(parents=True)
        self.launcher.write_text("#!/bin/sh\n")
        self.launcher.chmod(0o755)
        self.running: str | None = running
        self.installed = installed
        self.calls: list[list[str]] = []
        self.install_result = Result(0, "Installed 2 packages")
        self.forget_extras = False
        self.loaded = True
        self.active = True
        self.restart_result = Result(0, "")
        self.pings: list[int] = []
        self.write_receipt(f"v{installed}", [str(self.plugin)])

    def write_receipt(self, rev: str, extras: Sequence[str]) -> None:
        lines = [f'    {{ name = "fle-studio", git = "{GIT}?rev={rev}" }},']
        lines += [f'    {{ name = "fle-studio-ginny", directory = "{p}" }},' for p in extras]
        (self.env / "uv-receipt.toml").write_text(
            "[tool]\nrequirements = [\n" + "\n".join(lines) + '\n]\npython = "3.12"\nentrypoints = [\n'
            f'    {{ name = "fle-studio", install-path = "{self.launcher}", from = "fle-studio" }},\n]\n',
            encoding="utf-8",
        )

    def run(self, cmd: Sequence[str], stream: bool = False) -> Result:
        cmd = list(cmd)
        self.calls.append(cmd)
        if cmd[0] == "/usr/bin/uv" and cmd[1:3] == ["tool", "dir"]:
            return Result(0, str(self.bin if "--bin" in cmd else self.uv_dir) + "\n")
        if cmd[0] == "/usr/bin/uv" and cmd[1:3] == ["tool", "install"]:
            if self.install_result.code:
                return self.install_result
            extras = [cmd[i + 1] for i, a in enumerate(cmd) if a == "--with"]
            self.installed = cmd[-1].rsplit("@", 1)[-1].lstrip("v")
            self.write_receipt("v" + self.installed, [] if self.forget_extras else extras)
            return self.install_result
        if cmd[0] == str(self.launcher):
            return Result(0, f"FLE Studio {self.installed}\n")
        if cmd[:2] == ["launchctl", "print"]:
            return Result(0 if self.loaded else 113, "")
        if cmd[:2] == ["launchctl", "kickstart"] or cmd[:3] == ["systemctl", "--user", "restart"]:
            if not self.restart_result.code:
                self.running = self.installed
            return self.restart_result
        if cmd[:3] == ["systemctl", "--user", "is-active"]:
            return Result(0 if self.active else 3, "")
        if cmd[0] == "codesign":
            return Result(0, "")
        return Result(127, f"unexpected call {cmd}")

    def ping(self, port: int) -> str | None:
        self.pings.append(port)
        return self.running

    def host(self, platform: str = "darwin") -> Host:
        return Host(
            run=self.run,
            ping=self.ping,
            sleep=lambda _s: None,
            platform=platform,
            home=self.tmp / "home",
            uid=os.getuid() if hasattr(os, "getuid") else 0,
            user="max",
            apps_dirs=(self.apps,),
            which=lambda name: f"/usr/bin/{name}",
        )

    def launchd_service(self, label: str = "de.example.fle-studio", port_args: Sequence[str] = ()) -> Path:
        folder = self.tmp / "home" / "Library" / "LaunchAgents"
        folder.mkdir(parents=True, exist_ok=True)
        plist = folder / f"{label}.plist"
        plist.write_bytes(
            plistlib.dumps(
                {
                    "Label": label,
                    "ProgramArguments": [str(self.launcher), "--no-browser", *port_args],
                    "KeepAlive": True,
                }
            )
        )
        return plist

    def systemd_service(self, name: str = "fle-studio.service", exec_line: str | None = None) -> Path:
        folder = self.tmp / "home" / ".config" / "systemd" / "user"
        folder.mkdir(parents=True, exist_ok=True)
        unit = folder / name
        unit.write_text(
            "[Service]\nExecStart="
            + (exec_line or f"{self.launcher} --service rhea.service --no-browser --port 8642")
            + "\n"
        )
        return unit

    def installs(self) -> list[list[str]]:
        return [c for c in self.calls if c[1:3] == ["tool", "install"]]

    def kicks(self) -> list[list[str]]:
        return [
            c for c in self.calls if c[:2] == ["launchctl", "kickstart"] or c[:3] == ["systemctl", "--user", "restart"]
        ]


@pytest.fixture
def machine(tmp_path: Path) -> Machine:
    return Machine(tmp_path)


def run_update(m: Machine, platform: str = "darwin", **opts: Any) -> tuple[int, list[str]]:
    said: list[str] = []
    code = selfupdate.update(Options(**opts), m.host(platform), m.data, said.append)
    return code, said


# ------------------------------------------------------------------------------------------------ the update


def test_update_installs_with_the_plugin_restarts_the_service_and_names_both_versions(machine: Machine) -> None:
    machine.launchd_service()
    code, said = run_update(machine, version="v0.4.2")
    assert code == 0
    assert machine.installs() == [
        [
            "/usr/bin/uv",
            "tool",
            "install",
            "--python",
            "3.12",
            "--force",
            "--with",
            str(machine.plugin),
            f"git+{GIT}@v0.4.2",
        ]
    ]
    assert [c[1] for c in machine.kicks()] == ["kickstart"] and machine.kicks()[0][-1].endswith(
        "/de.example.fle-studio"
    )
    text = "\n".join(said)
    assert "0.4.1 is installed; the target is 0.4.2" in text
    assert "Kept next to FLE Studio: fle-studio-ginny" in text
    assert "Running: 0.4.2 on port 8642" in text
    assert "fle-studio update --rollback" in text
    assert "0.4.2" in (machine.data / "update.log").read_text(encoding="utf-8")  # the log has it too


def test_the_plugin_is_still_there_after_the_update(machine: Machine) -> None:
    run_update(machine, version="v0.4.2")
    after = selfupdate.parse_receipt((machine.env / "uv-receipt.toml").read_text(encoding="utf-8"))
    assert [r.name for r in after.extras] == ["fle-studio-ginny"]


def test_a_stale_service_is_restarted_even_when_the_version_is_already_installed(machine: Machine) -> None:
    """27.10.2026: uv had installed 0.4.0 by hand, the service still ran 0.3.2."""
    machine.running = "0.3.2"
    machine.launchd_service()
    code, said = run_update(machine, version="v0.4.1")
    assert code == 0 and machine.installs() == []  # nothing to install
    assert len(machine.kicks()) == 1
    assert "Running: 0.4.1 on port 8642" in "\n".join(said)


def test_a_service_that_already_runs_the_version_is_left_alone(machine: Machine) -> None:
    machine.launchd_service()
    code, said = run_update(machine, version="v0.4.1")
    assert code == 0 and machine.installs() == [] and machine.kicks() == []
    assert "nothing to restart" in "\n".join(said)


def test_a_service_that_does_not_come_up_is_reported_with_where_to_look(machine: Machine) -> None:
    plist = machine.launchd_service()
    data = plistlib.loads(plist.read_bytes())
    data["StandardErrorPath"] = "/Users/me/Library/Logs/fle-studio.log"
    plist.write_bytes(plistlib.dumps(data))
    machine.restart_result = Result(0, "")
    host = machine.host()
    host.ping = lambda port: None  # the server never answers
    with pytest.raises(Problem) as err:
        selfupdate.update(Options(version="v0.4.2"), host, machine.data, lambda _t: None)
    assert "nothing answers on port 8642" in err.value.cause and "fle-studio.log" in err.value.remedy


def test_a_port_held_by_another_version_is_named(machine: Machine) -> None:
    machine.launchd_service()
    host = machine.host()
    host.ping = lambda port: "0.3.2"  # another program keeps answering
    with pytest.raises(Problem) as err:
        selfupdate.update(Options(version="v0.4.2"), host, machine.data, lambda _t: None)
    assert "answers with 0.3.2, expected 0.4.2" in err.value.cause and "lsof" in err.value.remedy


def test_the_port_of_the_service_is_the_one_that_is_asked(machine: Machine) -> None:
    machine.launchd_service(port_args=["--port", "9100"])
    run_update(machine, version="v0.4.2")
    assert 9100 in machine.pings and 8642 not in machine.pings


def test_no_service_and_nothing_running_says_how_to_start(machine: Machine) -> None:
    machine.running = None
    code, said = run_update(machine, version="v0.4.2")
    assert code == 0 and machine.kicks() == []
    assert "no Studio is running" in "\n".join(said)


def test_a_studio_started_by_hand_cannot_be_restarted_and_the_exit_code_says_so(machine: Machine) -> None:
    code, said = run_update(machine, version="v0.4.2")  # running 0.4.1, no service
    assert code == 3 and machine.kicks() == []
    text = "\n".join(said)
    assert "Installed: 0.4.2. Running: 0.4.1" in text and "Ctrl+C" in text


def test_a_service_that_is_not_loaded_is_left_alone(machine: Machine) -> None:
    machine.launchd_service()
    machine.loaded = False
    _, said = run_update(machine, version="v0.4.2")
    assert machine.kicks() == [] and any("not loaded" in line for line in said)


# systemd unit files hold POSIX paths and are split like a shell line; with Windows paths that makes no sense.
posix_units = pytest.mark.skipif(sys.platform == "win32", reason="unit files hold POSIX paths")


@posix_units
def test_the_systemd_user_unit_is_restarted_on_linux(machine: Machine) -> None:
    machine.systemd_service()
    code, said = run_update(machine, platform="linux", version="v0.4.2")
    assert code == 0 and machine.kicks() == [["systemctl", "--user", "restart", "fle-studio.service"]]
    assert "Restarted systemd fle-studio.service" in "\n".join(said)
    assert not any(c[0] == "codesign" for c in machine.calls)  # the Dock starter is a Mac thing


@posix_units
def test_systemd_without_a_user_session_says_how_to_get_one(machine: Machine) -> None:
    machine.systemd_service()
    machine.restart_result = Result(1, "Failed to connect to bus: No medium found")
    with pytest.raises(Problem) as err:
        run_update(machine, platform="linux", version="v0.4.2")
    assert "XDG_RUNTIME_DIR" in err.value.remedy


def test_a_stopped_systemd_unit_is_not_started_by_the_update(machine: Machine) -> None:
    machine.systemd_service()
    machine.active = False
    run_update(machine, platform="linux", version="v0.4.2")
    assert machine.kicks() == []


def test_dry_run_changes_nothing(machine: Machine) -> None:
    machine.launchd_service()
    code, said = run_update(machine, dry_run=True, version="v0.4.2")
    assert code == 0 and machine.installs() == [] and machine.kicks() == []
    assert (
        "Would run: /usr/bin/uv tool install" in "\n".join(said)
        and not (machine.data / "update-previous.json").exists()
    )


def test_windows_is_pointed_to_the_button(machine: Machine) -> None:
    with pytest.raises(Problem) as err:
        run_update(machine, platform="win32", version="v0.4.2")
    assert "Settings" in err.value.remedy and machine.calls == []


def test_without_uv_the_installer_is_named(machine: Machine, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("UV_INSTALL_DIR", raising=False)
    monkeypatch.delenv("XDG_BIN_HOME", raising=False)
    host = machine.host()
    host.which = lambda name: None
    host.home = machine.tmp / "nohome"
    with pytest.raises(Problem) as err:
        selfupdate.update(Options(version="v0.4.2"), host, machine.data, lambda _t: None)
    assert "uv was not found" in err.value.cause and "install.sh" in err.value.remedy


def test_not_a_uv_tool_install(machine: Machine) -> None:
    (machine.env / "uv-receipt.toml").unlink()
    with pytest.raises(Problem) as err:
        run_update(machine, version="v0.4.2")
    assert "not installed as a uv tool" in err.value.cause and "install.sh" in err.value.remedy


# ------------------------------------------------------------------------------------------------ what goes wrong


def test_a_missing_plugin_folder_stops_before_anything_changes(machine: Machine) -> None:
    machine.plugin.rmdir()
    with pytest.raises(Problem) as err:
        run_update(machine, version="v0.4.2")
    assert "not there any more" in err.value.cause and "--drop fle-studio-ginny" in err.value.remedy
    assert machine.installs() == []


def test_drop_leaves_the_package_out(machine: Machine) -> None:
    machine.plugin.rmdir()
    run_update(machine, version="v0.4.2", drop=("fle-studio-ginny",))
    assert "--with" not in machine.installs()[0]


def test_missing_git_is_named_with_the_fix(machine: Machine) -> None:
    host = machine.host()
    host.which = lambda name: None if name == "git" else f"/usr/bin/{name}"
    with pytest.raises(Problem) as err:
        selfupdate.update(Options(version="v0.4.2"), host, machine.data, lambda _t: None)
    assert "git was not found" in err.value.cause and "xcode-select" in err.value.remedy


@pytest.mark.parametrize(
    ("output", "cause", "remedy"),
    [
        (
            "error: Failed to fetch: https://github.com/x\n  dns error: failed to lookup address",
            "No connection",
            "again",
        ),
        ("fatal: couldn't find remote ref v9.9.9", "no release v9.9.9", "releases"),
        ("error: Git operation failed: reference is not a tree", "no release v9.9.9", "without --version"),
        ("something nobody planned for", "exit code 2", "--rollback"),
    ],
)
def test_uv_complaints_become_a_cause_and_a_remedy(machine: Machine, output: str, cause: str, remedy: str) -> None:
    machine.install_result = Result(2, output)
    with pytest.raises(Problem) as err:
        run_update(machine, version="v9.9.9")
    assert cause in err.value.cause and remedy in err.value.remedy
    assert output.splitlines()[-1] in (machine.data / "update.log").read_text(encoding="utf-8")  # uv's text is kept


def test_nothing_changes_in_the_receipt_when_uv_fails(machine: Machine) -> None:
    machine.install_result = Result(2, "fatal: couldn't find remote ref")
    before = (machine.env / "uv-receipt.toml").read_text(encoding="utf-8")
    with pytest.raises(Problem):
        run_update(machine, version="v9.9.9")
    assert (machine.env / "uv-receipt.toml").read_text(encoding="utf-8") == before


@pytest.mark.skipif(not hasattr(os, "getuid"), reason="file owners are a POSIX thing")
def test_an_environment_owned_by_someone_else_gets_the_exact_chown(machine: Machine) -> None:
    host = machine.host()
    host.uid = os.getuid() + 1  # as if the files belonged to another user, root after a sudo run
    with pytest.raises(Problem) as err:
        selfupdate.update(Options(version="v0.4.2"), host, machine.data, lambda _t: None)
    assert "belongs to another user" in err.value.cause
    assert f"sudo chown -R max {machine.env}" in err.value.remedy
    assert machine.installs() == []


@pytest.mark.skipif(not hasattr(os, "getuid"), reason="file owners are a POSIX thing")
def test_permission_denied_from_uv_is_explained_even_when_the_check_was_fine(machine: Machine) -> None:
    machine.install_result = Result(2, "error: failed to open file `uv-receipt.toml`: Permission denied (os error 13)")
    with pytest.raises(Problem) as err:
        run_update(machine, version="v0.4.2")
    assert "permission denied" in err.value.cause and f"sudo chown -R max {machine.env}" in err.value.remedy


def test_foreign_owner_is_found_deep_inside_and_a_plain_tree_is_clean(tmp_path: Path) -> None:
    (tmp_path / "a" / "b").mkdir(parents=True)
    (tmp_path / "a" / "b" / "uv-receipt.toml").write_text("x")
    if hasattr(os, "getuid"):
        assert selfupdate.foreign_owned(tmp_path, os.getuid()) is None
        assert selfupdate.foreign_owned(tmp_path, os.getuid() + 1) == tmp_path


def test_a_plugin_lost_on_the_way_is_caught(machine: Machine) -> None:
    """The very failure of 0.3.2 to 0.4.0: the install ran, the plugin was gone."""
    machine.forget_extras = True
    with pytest.raises(Problem) as err:
        run_update(machine, version="v0.4.2")
    assert "fle-studio-ginny" in err.value.cause and "--rollback" in err.value.remedy


def test_a_program_that_reports_another_version_is_caught(machine: Machine) -> None:
    host = machine.host()
    real = host.run

    def lying_run(cmd: Sequence[str], stream: bool = False) -> Result:
        return Result(0, "FLE Studio 0.0.1\n") if cmd[0] == str(machine.launcher) else real(cmd, stream)

    host.run = lying_run
    with pytest.raises(Problem) as err:
        selfupdate.update(Options(version="v0.4.2"), host, machine.data, lambda _t: None)
    assert "reports 0.0.1, expected 0.4.2" in err.value.cause


# ------------------------------------------------------------------------------------------------ rollback


def test_rollback_installs_the_version_before_with_the_packages_it_had(machine: Machine) -> None:
    machine.launchd_service()
    run_update(machine, version="v0.4.2")  # remembers 0.4.1 as it was, with the plugin
    assert (machine.data / "update-previous.json").is_file()
    machine.calls.clear()
    code, said = run_update(machine, rollback=True)
    assert code == 0
    assert machine.installs() == [
        [
            "/usr/bin/uv",
            "tool",
            "install",
            "--python",
            "3.12",
            "--force",
            "--with",
            str(machine.plugin),
            f"git+{GIT}@v0.4.1",
        ]
    ]
    assert "Going back from FLE Studio 0.4.2 to 0.4.1" in "\n".join(said)
    assert "Running: 0.4.1 on port 8642" in "\n".join(said)  # the service was restarted onto the old version
    # going back twice goes forward again: the record now holds 0.4.2
    assert selfupdate.load_previous(machine.data)[0] == "0.4.2"
    machine.calls.clear()
    run_update(machine, rollback=True)  # and again: forward to 0.4.2, with the plugin still along
    assert machine.installs()[0][-1] == f"git+{GIT}@v0.4.2" and str(machine.plugin) in machine.installs()[0]


def test_rollback_without_a_record_says_what_to_do(machine: Machine) -> None:
    with pytest.raises(Problem) as err:
        run_update(machine, rollback=True)
    assert "no earlier version" in err.value.cause and "--version" in err.value.remedy


def test_the_record_survives_a_round_trip(tmp_path: Path) -> None:
    receipt = selfupdate.parse_receipt(MAC_RECEIPT)
    selfupdate.save_previous(tmp_path, receipt, "0.4.0")
    version, back = selfupdate.load_previous(tmp_path)
    assert version == "0.4.0" and back.requirements == receipt.requirements and back.python == "3.12"


def test_a_damaged_record_is_no_record(tmp_path: Path) -> None:
    (tmp_path / "update-previous.json").write_text('{"v": 1, "version": "0.4.0", "requirements": [{"x": 1}]}')
    with pytest.raises(Problem):
        selfupdate.load_previous(tmp_path)
    (tmp_path / "update-previous.json").write_text("not json")
    with pytest.raises(Problem):
        selfupdate.load_previous(tmp_path)


def test_rolling_back_to_the_running_version_is_refused(machine: Machine) -> None:
    selfupdate.save_previous(machine.data, selfupdate.parse_receipt(MAC_RECEIPT), machine.installed)
    with pytest.raises(Problem, match="already the version"):
        run_update(machine, rollback=True)


def test_rollback_and_version_exclude_each_other(capsys: pytest.CaptureFixture[str]) -> None:
    assert selfupdate.cli(["--rollback", "--version", "v0.4.0"]) == 2
    assert "exclude each other" in capsys.readouterr().err
    assert selfupdate.build_parser().parse_args(["--zurueck"]).rollback is True


def test_cli_prints_the_problem_and_returns_1(
    machine: Machine, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("FLE_STUDIO_HOME", str(machine.data))
    machine.plugin.rmdir()
    assert selfupdate.cli(["--version", "v0.4.2"], machine.host()) == 1
    err = capsys.readouterr().err
    assert "The update did not work" in err and "What to do:" in err


# ------------------------------------------------------------------------------------------------ the latest release


def test_latest_release_is_asked_from_github(monkeypatch: pytest.MonkeyPatch) -> None:
    from fake_github import FakeGitHub, release

    with FakeGitHub([release("v0.4.2"), release("v0.4.1")]) as gh:
        monkeypatch.setenv("FLE_STUDIO_GITHUB_API", gh.url)
        assert selfupdate.latest_tag() == "v0.4.2"
        assert gh.paths == ["/repos/sirfyyn/fle-studio/releases/latest"]


def test_no_network_for_the_latest_release_is_explained(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FLE_STUDIO_GITHUB_API", "http://127.0.0.1:9")  # nothing listens on port 9
    with pytest.raises(Problem) as err:
        selfupdate.latest_tag()
    assert "Could not ask GitHub" in err.value.cause and "--version" in err.value.remedy


# ------------------------------------------------------------------------------------------------ finding services


def test_launchd_agents_are_found_by_the_program_they_start(machine: Machine) -> None:
    machine.launchd_service("de.example.fle-studio")
    other = machine.tmp / "home" / "Library" / "LaunchAgents" / "de.example.other.plist"
    other.write_bytes(plistlib.dumps({"Label": "de.example.other", "ProgramArguments": ["/usr/bin/true"]}))
    (machine.tmp / "home" / "Library" / "LaunchAgents" / "broken.plist").write_text("not a plist")
    found = selfupdate.find_services(machine.host(), 8642)
    assert [(s.manager, s.name, s.port) for s in found] == [("launchd", "de.example.fle-studio", 8642)]


@posix_units
def test_systemd_units_are_found_by_their_exec_line(machine: Machine) -> None:
    machine.systemd_service()
    machine.systemd_service("other.service", exec_line="/usr/bin/sleep 1000")
    machine.systemd_service("second.service", exec_line=f"-{machine.launcher} --port=9001")
    found = selfupdate.find_services(machine.host("linux"), 8642)
    assert [(s.name, s.port) for s in found] == [("fle-studio.service", 8642), ("second.service", 9001)]


def test_the_unit_of_the_server_is_found_as_it_is_written(machine: Machine) -> None:
    """fle-studio.service on Titan: %h in ExecStart, the engine unit and the port as arguments."""
    unit = machine.systemd_service(
        exec_line="%h/.local/bin/fle-studio --service rhea.service --no-browser --no-tls --port 8642"
    )
    host = machine.host("linux")
    (host.home / ".local" / "bin").mkdir(parents=True)  # %h/.local/bin/fle-studio need not exist to be recognised
    assert [(s.name, s.port, s.definition) for s in selfupdate.find_services(host, 1)] == [
        ("fle-studio.service", 8642, unit)
    ]


def test_on_windows_no_service_is_looked_for_and_the_update_says_where_to_go(machine: Machine) -> None:
    """No launchd, no systemd: the command line names the way that works; a running Studio is told to restart."""
    with pytest.raises(Problem) as err:
        run_update(machine, platform="win32", version="v0.4.2")
    assert "Settings" in err.value.remedy and "install.ps1" in err.value.remedy
    said: list[str] = []
    assert selfupdate.settle(machine.host("win32"), [], 8642, "0.4.2", selfupdate.Log(machine.data, said.append)) == 3
    assert "started by hand" in said[0] and "start fle-studio again" in said[0]


def test_no_services_on_windows_or_without_folders(machine: Machine) -> None:
    assert selfupdate.find_services(machine.host("win32"), 8642) == []
    assert selfupdate.find_services(machine.host("darwin"), 8642) == []


def test_the_configured_port_is_read_without_starting_studio(tmp_path: Path) -> None:
    assert selfupdate.configured_port(tmp_path) == 8642
    (tmp_path / "settings.json").write_text('{"server": {"port": 9000}}')
    assert selfupdate.configured_port(tmp_path) == 9000
    (tmp_path / "settings.json").write_text("broken")
    assert selfupdate.configured_port(tmp_path) == 8642


def test_waiting_for_the_version_stops_when_it_is_there(machine: Machine) -> None:
    answers = iter([None, "0.4.1", "0.4.2"])
    host = machine.host()
    host.ping = lambda port: next(answers)
    assert selfupdate.wait_for_version(host, 8642, "0.4.2", 10) == "0.4.2"
    host.ping = lambda port: "0.4.1"
    assert selfupdate.wait_for_version(host, 8642, "0.4.2", 3) == "0.4.1"  # gives up, says what it saw


# ------------------------------------------------------------------------------------------------ the Dock starter


def make_starter(machine: Machine, text: str = "#!/bin/bash\n# the old one\n") -> Path:
    program = machine.apps / dock_starter.APP_NAME / "Contents" / "MacOS" / "starter"
    program.parent.mkdir(parents=True)
    program.write_text(text)
    program.chmod(0o644)
    return program


def test_the_shim_runs_the_installed_program_and_survives_odd_paths() -> None:
    text = dock_starter.shim_text("/Users/o'neil/.local/bin/fle-studio", 8642)
    assert text.startswith("#!/bin/bash") and 'exec "$LAUNCHER" open' in text
    assert "LAUNCHER='/Users/o'\\''neil/.local/bin/fle-studio'" in text
    assert "http://127.0.0.1:8642/" in text  # the fallback when the program is gone


def test_an_old_starter_is_replaced_and_signed_again(machine: Machine) -> None:
    program = make_starter(machine)
    lines = dock_starter.refresh(machine.host(), "/x/fle-studio", 8642)
    assert "Refreshed the Dock starter and signed it again" in lines[0]
    assert program.read_text() == dock_starter.shim_text("/x/fle-studio", 8642)
    if sys.platform != "win32":  # Windows has no execute bit; the Dock starter is a Mac thing anyway
        assert program.stat().st_mode & stat.S_IXUSR
    signing = [c for c in machine.calls if c[0] == "codesign"]
    assert signing[0][:4] == ["codesign", "--force", "--sign", "-"] and signing[1][1] == "--verify"


def test_a_current_starter_is_left_alone_and_not_signed_again(machine: Machine) -> None:
    make_starter(machine, dock_starter.shim_text("/x/fle-studio", 8642))
    lines = dock_starter.refresh(machine.host(), "/x/fle-studio", 8642)
    assert "is current" in lines[0] and not any(c[0] == "codesign" for c in machine.calls)


def test_no_starter_means_no_line_and_no_new_app(machine: Machine) -> None:
    assert dock_starter.refresh(machine.host(), "/x/fle-studio", 8642) == []
    assert not (machine.apps / dock_starter.APP_NAME).exists()


def test_a_failed_signature_is_reported_with_the_command(machine: Machine) -> None:
    make_starter(machine)
    host = machine.host()
    host.run = lambda cmd, stream=False: Result(1, "no identity") if cmd[0] == "codesign" else Result(0, "")
    lines = dock_starter.refresh(host, "/x/fle-studio", 8642)
    assert "signature is not valid" in lines[0] and "codesign --force --sign -" in lines[0]


def test_the_update_refreshes_the_starter_on_the_mac_only(machine: Machine) -> None:
    program = make_starter(machine)
    run_update(machine, platform="linux", version="v0.4.2")
    assert "the old one" in program.read_text()
    run_update(machine, platform="darwin", version="v0.4.2")
    assert "fle-studio" in program.read_text() and "open" in program.read_text()

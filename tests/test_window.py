"""`fle-studio open`: the window the Dock starter opens, and the check that the Studio behind it is current."""

from __future__ import annotations

from pathlib import Path

import pytest
from test_selfupdate import Machine

from fle_studio import __version__, window


def test_chrome_gets_a_profile_of_its_own() -> None:
    cmd = window.chrome_command("/c/chrome", "http://127.0.0.1:8642/", Path("/data/FLE Studio/chrome"))
    assert cmd[0] == "/c/chrome" and "--app=http://127.0.0.1:8642/" in cmd
    assert "--user-data-dir=/data/FLE Studio/chrome" in cmd  # one argument, spaces and all
    assert "--no-first-run" in cmd


def test_the_running_current_studio_is_left_alone(tmp_path: Path) -> None:
    m = Machine(tmp_path, running=__version__)
    m.launchd_service()
    said: list[str] = []
    window.ensure_current_studio(m.host(), 8642, said.append)
    assert said == [] and m.kicks() == []


def test_an_outdated_studio_behind_a_service_is_restarted_before_the_window_opens(tmp_path: Path) -> None:
    m = Machine(tmp_path, running="0.3.2", installed=__version__)
    m.launchd_service()
    said: list[str] = []
    window.ensure_current_studio(m.host(), 8642, said.append)
    assert len(m.kicks()) == 1
    assert any("runs 0.3.2, installed is" in line for line in said) and any(
        f"now answers with {__version__}" in line for line in said
    )


def test_a_service_on_another_port_is_not_touched(tmp_path: Path) -> None:
    m = Machine(tmp_path, running="0.3.2", installed=__version__)
    m.launchd_service(port_args=["--port", "9100"])
    window.ensure_current_studio(m.host(), 8642, lambda _t: None)
    assert m.kicks() == []


def test_a_studio_started_by_hand_is_not_killed_but_named(tmp_path: Path) -> None:
    m = Machine(tmp_path, running="0.3.2")
    said: list[str] = []
    window.ensure_current_studio(m.host(), 8642, said.append)
    assert m.kicks() == [] and "started by hand" in said[0]


def test_nothing_running_and_no_service_starts_studio_in_the_background(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    m = Machine(tmp_path, running=None)
    started: list[list[str]] = []
    monkeypatch.setattr(window, "detached", lambda cmd: started.append(list(cmd)))
    window.ensure_current_studio(m.host(), 8642, lambda _t: None)
    assert started and started[0][1:] == ["-m", "fle_studio", "--no-browser"]


def test_dry_run_opens_nothing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    m = Machine(tmp_path, running="0.3.2")
    m.launchd_service()
    monkeypatch.setenv("FLE_STUDIO_HOME", str(tmp_path / "data"))
    monkeypatch.setattr(window, "detached", lambda cmd: pytest.fail("must not start anything"))
    monkeypatch.setattr(window.webbrowser, "open", lambda url: pytest.fail("must not open a browser"))
    said: list[str] = []
    assert window.open_window(m.host(), said.append, dry_run=True) == 0
    assert m.kicks() == [] and any("Would check the Studio on port 8642" in line for line in said)


def test_the_window_opens_in_chrome_with_the_profile_in_the_data_folder(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    m = Machine(tmp_path, running=__version__)
    monkeypatch.setenv("FLE_STUDIO_HOME", str(tmp_path / "data"))
    monkeypatch.setattr(window, "chrome_path", lambda host: "/c/chrome")
    started: list[list[str]] = []
    monkeypatch.setattr(window, "detached", lambda cmd: started.append(list(cmd)))
    assert window.open_window(m.host(), lambda _t: None) == 0
    assert started == [window.chrome_command("/c/chrome", "http://127.0.0.1:8642/", tmp_path / "data" / "chrome")]


def test_without_chrome_the_default_browser_opens_the_address(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    m = Machine(tmp_path, running=__version__)
    monkeypatch.setenv("FLE_STUDIO_HOME", str(tmp_path / "data"))
    monkeypatch.setattr(window, "chrome_path", lambda host: "")
    opened: list[str] = []
    monkeypatch.setattr(window.webbrowser, "open", opened.append)
    window.open_window(m.host(), lambda _t: None)
    assert opened == ["http://127.0.0.1:8642/"]


def test_a_problem_with_the_service_does_not_keep_the_window_from_opening(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    m = Machine(tmp_path, running="0.3.2", installed=__version__)
    m.launchd_service()
    from fle_studio.selfupdate import Result

    m.restart_result = Result(1, "launchd refuses")
    monkeypatch.setenv("FLE_STUDIO_HOME", str(tmp_path / "data"))
    monkeypatch.setattr(window, "chrome_path", lambda host: "")
    monkeypatch.setattr(window.webbrowser, "open", lambda url: None)
    said: list[str] = []
    assert window.open_window(m.host(), said.append) == 0
    assert any("launchd refuses" in line for line in said) and any("default browser" in line for line in said)

"""The update check and the one-click update, without a network and without a real uv."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

import pytest
from fake_github import FakeGitHub, release

from fle_studio import __version__, paths, updates
from fle_studio.config import Config

FAKE_UV = Path(__file__).parent / "fake_uv.py"
STUDIO_PIP = {"kind": "pip", "source": "git", "service": False, "launcher": ""}
STUDIO_TOOL = {"kind": "uv-tool", "source": "git", "service": False, "launcher": ""}


# ------------------------------------------------------------------------------------------------ versions


@pytest.mark.parametrize(
    ("newer", "older"),
    [
        ("0.2.0", "0.1.1"),
        ("v0.10.0", "0.9.9"),
        ("1.0", "0.99.99"),
        ("0.2.0", "0.2.0rc1"),
        ("0.2.0rc2", "0.2.0rc1"),
        ("0.2.0rc1", "0.2.0b3"),
        ("0.2.0b1", "0.2.0a9"),
        ("0.2.0a1", "0.2.0.dev5"),
        ("0.2.0.post1", "0.2.0"),
        ("0.2.1.dev1", "0.2.0"),
        ("2!0.1", "1.9"),
        ("0.2.0", "0.2.0-rc.1"),
    ],
)
def test_pep440_order(newer: str, older: str) -> None:
    assert updates.is_newer(newer, older)
    assert not updates.is_newer(older, newer)


@pytest.mark.parametrize(
    ("a", "b"), [("1.0", "1.0.0"), ("v0.2.0", "0.2.0"), ("0.2.0-rc1", "0.2.0rc1"), ("0.2.0+local", "0.2.0")]
)
def test_pep440_equal(a: str, b: str) -> None:
    assert not updates.is_newer(a, b) and not updates.is_newer(b, a)


@pytest.mark.parametrize("text", ["", "latest", "v", "1..2", "0.2.0-final-x", "main"])
def test_not_a_version(text: str) -> None:
    assert updates.parse_version(text) is None
    assert not updates.is_newer(text, "0.1.0")


def test_prerelease_flag() -> None:
    assert updates.parse_version("0.2.0rc1").prerelease  # type: ignore[union-attr]
    assert updates.parse_version("0.2.0.dev1").prerelease  # type: ignore[union-attr]
    assert not updates.parse_version("0.2.0").prerelease  # type: ignore[union-attr]
    assert not updates.parse_version("0.2.0.post1").prerelease  # type: ignore[union-attr]


# ------------------------------------------------------------------------------------------------ picking a release


def test_pick_release_skips_drafts_prereleases_and_odd_tags() -> None:
    items = [
        release("v0.4.0", draft=True),
        release("v0.3.0rc1", prerelease=True),
        release("nightly"),
        release("v0.2.0", "Notes"),
        release("v0.1.1"),
    ]
    assert updates.pick_release(items, False)["tag"] == "v0.2.0"  # type: ignore[index]
    assert updates.pick_release(items, True)["tag"] == "v0.3.0rc1"  # type: ignore[index]
    assert updates.pick_release([release("v0.9.0", prerelease=True)], False) is None


def test_a_release_flagged_prerelease_by_github_is_one_even_with_a_plain_tag() -> None:
    assert updates.pick_release([release("v0.3.0", prerelease=True)], False) is None


def test_release_link_is_built_not_taken_from_the_network() -> None:
    found = updates.pick_release(release("v0.2.0", "x"), False)
    assert found["url"] == "https://github.com/sirfyyn/fle-studio/releases/tag/v0.2.0"  # type: ignore[index]


def test_notes_are_plain_text_and_cut() -> None:
    body = "# Title\n\n" + "paragraph\n\n" * 2000 + "\x00\x07<script>alert(1)</script>"
    notes = updates.pick_release(release("v0.2.0", body), False)["notes"]  # type: ignore[index]
    assert len(notes) <= updates.NOTES_MAX + 4 and notes.endswith("…") and "\x00" not in notes
    short = updates.pick_release(release("v0.2.0", "a\r\nb\x07"), False)["notes"]  # type: ignore[index]
    assert short == "a\nb"


# ------------------------------------------------------------------------------------------------ engine updates


def engine_entry(**kw: Any) -> dict[str, Any]:
    return {
        "id": "fle-cuda-old",
        "variant": "fle-cuda",
        "kind": "fle",
        "tag": "fle-1",
        "backend": "cuda",
        "installed": 100.0,
        **kw,
    }


def engine_release(tag: str, with_build: bool = True) -> dict[str, Any]:
    from fle_studio import engines

    osname, arch = engines._os_arch()
    plat = {"linux": "ubuntu", "win": "win", "macos": "macos"}[osname]
    ext = "zip" if osname == "win" else "tar.gz"
    name = f"fle-{tag}-bin-{plat}-cuda-12.8-{arch}.{ext}"
    return release(tag, assets=[name] if with_build else ["something-else.zip"])


def test_engine_update_found_when_the_tag_differs() -> None:
    found = updates.engine_updates([engine_entry()], engine_release("fle-2"))
    assert len(found) == 1 and found[0]["installed"] == "fle-1" and found[0]["latest"] == "fle-2"
    assert found[0]["id"] == "fle-cuda@fle-2" and found[0]["url"].endswith("/releases/tag/fle-2")


def test_no_engine_update_when_current_foreign_or_without_a_build() -> None:
    assert updates.engine_updates([engine_entry(tag="fle-2")], engine_release("fle-2")) == []
    assert updates.engine_updates([engine_entry(kind="upstream")], engine_release("fle-2")) == []
    assert updates.engine_updates([engine_entry()], engine_release("fle-2", with_build=False)) == []
    old_and_new = [engine_entry(), engine_entry(id="x", tag="fle-2", installed=200.0)]
    assert updates.engine_updates(old_and_new, engine_release("fle-2")) == []


# ------------------------------------------------------------------------------------------------ how it is installed


def test_receipt_of_a_git_install() -> None:
    text = (
        '[tool]\nrequirements = [{ name = "fle-studio", git = "https://github.com/sirfyyn/fle-studio", '
        'rev = "v0.1.1" }]\n'
        'python = "3.12"\nentrypoints = [\n    { name = "fle-studio", install-path = "/home/me/.local/bin/fle-studio", '
        'from = "fle-studio" },\n]\n'
    )
    assert updates.parse_receipt(text) == {"source": "git", "launcher": "/home/me/.local/bin/fle-studio"}


def test_receipt_of_a_folder_and_a_wheel_install() -> None:
    folder = '[tool]\nrequirements = [{ name = "fle-studio", directory = "/src/fle-studio" }]\n'
    wheel = '[tool]\nrequirements = [{ name = "fle-studio", path = "/tmp/fle_studio-0.1.1-py3-none-any.whl" }]\n'
    assert updates.parse_receipt(folder)["source"] == "folder"
    assert updates.parse_receipt(wheel)["source"] == "wheel"


def test_receipt_with_a_windows_launcher_path() -> None:
    text = (
        'requirements = [{ name = "fle-studio", git = "https://x" }]\n'
        'entrypoints = [{ name = "fle-studio", '
        'install-path = "C:\\\\Users\\\\me\\\\.local\\\\bin\\\\fle-studio.exe" }]\n'
    )
    assert updates.parse_receipt(text)["launcher"] == "C:\\Users\\me\\.local\\bin\\fle-studio.exe"


def test_detect_uv_tool_from_a_receipt(tmp_path: Path) -> None:
    (tmp_path / "uv-receipt.toml").write_text(
        'requirements = [{ name = "fle-studio", git = "https://x" }]\n', encoding="utf-8"
    )
    info = updates.detect_install(prefix=tmp_path, env={}, tty=True)
    assert info["kind"] == "uv-tool" and info["source"] == "git" and info["service"] is False


@pytest.mark.parametrize(
    ("direct", "kind", "source"),
    [
        (None, "pip", "index"),
        ({"url": "file:///src/x", "dir_info": {"editable": True}}, "source", "folder"),
        ({"url": "file:///src/x", "dir_info": {}}, "pip", "folder"),
        ({"url": "https://github.com/x/y", "vcs_info": {"vcs": "git"}}, "pip", "git"),
        ({"url": "file:///tmp/x.whl", "archive_info": {}}, "pip", "wheel"),
    ],
)
def test_detect_pip_installs(tmp_path: Path, direct: dict[str, Any] | None, kind: str, source: str) -> None:
    info = updates.detect_install(prefix=tmp_path, env={}, direct_url=direct, tty=True)
    assert (info["kind"], info["source"]) == (kind, source)


def test_service_mode_is_recognised() -> None:
    assert updates.in_service({"INVOCATION_ID": "abc"}, tty=False)
    assert updates.in_service({"JOURNAL_STREAM": "8:1"}, tty=False)
    assert updates.in_service({"FLE_STUDIO_SERVICE": "1"}, tty=True)
    assert updates.in_service({}, tty=True, engine_unit=True)
    assert not updates.in_service({"INVOCATION_ID": "abc"}, tty=True)  # a terminal inside a systemd session
    assert not updates.in_service({}, tty=False)


def test_advice_one_click_only_for_a_plain_uv_tool_from_github() -> None:
    ok = updates.advice(STUDIO_TOOL, "v0.2.0", True, True)
    assert (
        ok["can_auto"]
        and ok["command"] == "uv tool install --python 3.12 --force git+https://github.com/sirfyyn/fle-studio@v0.2.0"
    )
    service = updates.advice({**STUDIO_TOOL, "service": True}, "v0.2.0", True, True)
    assert not service["can_auto"] and "service" in service["reason"] and service["command"]
    assert not updates.advice({**STUDIO_TOOL, "source": "wheel"}, "v0.2.0", True, True)["can_auto"]
    assert "uv was not found" in updates.advice(STUDIO_TOOL, "v0.2.0", False, True)["reason"]
    assert "git" in updates.advice(STUDIO_TOOL, "v0.2.0", True, False)["reason"]


def test_advice_for_pip_and_source_installs() -> None:
    pip = updates.advice(STUDIO_PIP, "v0.2.0", True, True)
    assert not pip["can_auto"] and "pip install --upgrade" in pip["command"] and "v0.2.0" in pip["command"]
    assert sys.executable in pip["command"]
    source = updates.advice(
        {"kind": "source", "source": "folder", "service": False, "launcher": ""}, "v0.2.0", True, True
    )
    assert not source["can_auto"] and source["command"] == "git pull"


def test_no_command_for_a_tag_that_is_not_a_plain_release_tag() -> None:
    for bad in ("v0.2.0; rm -rf /", "main", "../x", "v0.2.0 --extra", ""):
        assert updates.update_command(STUDIO_PIP, bad) == ""


def test_user_args_do_not_open_a_second_tab(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "argv", ["fle-studio", "--port", "9000"])
    assert updates.user_args() == ["--port", "9000", "--no-browser"]
    monkeypatch.setattr(sys, "argv", ["fle-studio", "--no-browser"])
    assert updates.user_args() == ["--no-browser"]
    assert updates.restart_command()[-1] == "--no-browser"


# ------------------------------------------------------------------------------------------------ the check


class NoEngines:
    def __init__(self, installed: list[dict[str, Any]] | None = None) -> None:
        self._installed = installed or []

    def installed(self) -> list[dict[str, Any]]:
        return self._installed


def make_updates(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    gh: FakeGitHub | None,
    engines: list[dict[str, Any]] | None = None,
    **settings: Any,
) -> updates.Updates:
    monkeypatch.setenv("FLE_STUDIO_GITHUB_API", gh.url if gh else "http://127.0.0.1:9")  # port 9: nothing listens
    config = Config(tmp_path / "settings.json")
    if settings:
        config.update({"updates": settings})
    return updates.Updates(tmp_path, config, NoEngines(engines))


def test_nothing_is_asked_until_somebody_checks(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    with FakeGitHub([release("v9.0.0")]) as gh:
        u = make_updates(tmp_path, monkeypatch, gh)
        assert u.view()["app"] is None and gh.requests == []
        assert u.due()


def test_check_finds_a_newer_release_and_asks_without_identity(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    with FakeGitHub([release("v9.0.0", "# New\n\n- one")]) as gh:
        u = make_updates(tmp_path, monkeypatch, gh)
        view = u.check()
        assert view["app"]["version"] == "9.0.0" and view["app"]["notes"].startswith("# New")
        assert view["app"]["dismissed"] is False and view["current"] == __version__ and view["error"] == ""
        assert gh.paths == ["/repos/sirfyyn/fle-studio/releases/latest"]
        sent = gh.requests[0][1]
        assert not {"authorization", "cookie", "x-fle-token"} & set(sent)
        assert sent["user-agent"].startswith("fle-studio/")


def test_an_older_or_equal_release_is_no_update(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    with FakeGitHub([release(f"v{__version__}")]) as gh:
        assert make_updates(tmp_path, monkeypatch, gh).check()["app"] is None


def test_the_answer_is_kept_across_runs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    with FakeGitHub([release("v9.0.0")]) as gh:
        make_updates(tmp_path, monkeypatch, gh).check()
        again = make_updates(tmp_path, monkeypatch, gh)
        assert again.view()["app"]["version"] == "9.0.0" and not again.due()  # no new request needed
        assert len(gh.requests) == 1


def test_once_a_day(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    with FakeGitHub([release("v9.0.0")]) as gh:
        u = make_updates(tmp_path, monkeypatch, gh)
        u.check()
        u.check()
        u.check()
        assert len(gh.requests) == 1
        now = time.time()
        assert not u.due(now + 23 * 3600) and u.due(now + 25 * 3600)
        monkeypatch.setattr(updates.time, "time", lambda: now + 25 * 3600)
        u.check()
        assert len(gh.requests) == 2


def test_check_now_asks_again_but_not_in_a_flurry(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    with FakeGitHub([release("v9.0.0")]) as gh:
        u = make_updates(tmp_path, monkeypatch, gh)
        u.check(force=True)
        u.check(force=True)  # pressed twice at once
        assert len(gh.requests) == 1
        u._last_manual = 0.0
        u.check(force=True)
        assert len(gh.requests) == 2


def test_turned_off_means_no_request_at_all(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    with FakeGitHub([release("v9.0.0")]) as gh:
        u = make_updates(tmp_path, monkeypatch, gh, check=False)
        assert not u.due() and not u.due(time.time() + 10**7)
        with pytest.raises(updates.UpdateError):
            u.check()
        with pytest.raises(updates.UpdateError):
            u.check(force=True)
        assert gh.requests == [] and u.view()["enabled"] is False


def test_offline_stays_quiet_and_retries_in_an_hour(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    u = make_updates(tmp_path, monkeypatch, None)
    view = u.check()
    assert view["app"] is None and "No connection" in view["error"]
    now = time.time()
    assert not u.due(now + 1800) and u.due(now + 3700)  # sooner than a day, because nothing got through
    assert all(not e["dismissed"] for e in view["engines"])


def test_offline_keeps_an_earlier_answer(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    with FakeGitHub([release("v9.0.0")]) as gh:
        u = make_updates(tmp_path, monkeypatch, gh)
        u.check()
    monkeypatch.setenv("FLE_STUDIO_GITHUB_API", "http://127.0.0.1:9")
    u._last_manual = 0.0
    view = u.check(force=True)
    assert view["error"] and view["app"]["version"] == "9.0.0"


def test_a_github_error_is_not_offline(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    with FakeGitHub([release("v9.0.0")], status=403) as gh:
        u = make_updates(tmp_path, monkeypatch, gh)
        view = u.check()
        assert view["error"] == "GitHub answered 403" and view["app"] is None
        assert not u.due(time.time() + 3700)  # a rate limit is not helped by asking again within the hour


def test_prereleases_only_with_the_switch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    items = [release("v9.1.0rc1", prerelease=True), release("v9.0.0")]
    with FakeGitHub(items) as gh:
        assert make_updates(tmp_path, monkeypatch, gh).check()["app"]["version"] == "9.0.0"
        assert gh.paths == ["/repos/sirfyyn/fle-studio/releases/latest"]
    with FakeGitHub(items) as gh:
        view = (
            make_updates(tmp_path / "b", monkeypatch, gh, prereleases=True).check()
            if (tmp_path / "b").mkdir() is None
            else {}
        )
        assert view["app"]["version"] == "9.1.0rc1" and view["app"]["prerelease"] is True
        assert gh.paths[0].startswith("/repos/sirfyyn/fle-studio/releases?")


def test_later_hides_that_version_only(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    with FakeGitHub([release("v9.0.0")]) as gh:
        u = make_updates(tmp_path, monkeypatch, gh)
        u.check()
        u.dismiss("9.0.0")
        assert u.view()["app"]["dismissed"] is True
        gh.releases.insert(0, release("v9.1.0"))
        u._last_manual = 0.0
        assert u.check(force=True)["app"]["dismissed"] is False  # the next version shows again


def test_engine_update_is_part_of_the_check(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    with FakeGitHub([release(f"v{__version__}")], engine=engine_release("fle-2")) as gh:
        u = make_updates(tmp_path, monkeypatch, gh, engines=[engine_entry()])
        view = u.check()
        assert view["app"] is None and [e["latest"] for e in view["engines"]] == ["fle-2"]
        assert gh.paths[-1] == "/repos/sirfyyn/FastLocalExperts/releases/latest"
        u.dismiss(engines=True)
        assert u.view()["engines"][0]["dismissed"] is True


def test_no_engine_request_without_a_fle_engine_or_with_a_pinned_tag(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with FakeGitHub([release("v9.0.0")], engine=engine_release("fle-2")) as gh:
        make_updates(tmp_path, monkeypatch, gh, engines=[engine_entry(kind="upstream")]).check()
        assert gh.paths == ["/repos/sirfyyn/fle-studio/releases/latest"]


# ------------------------------------------------------------------------------------------------ the update run


@pytest.fixture
def run_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Updates with a found release v9.0.0, a plain uv tool install and the stand-in for uv."""
    out = tmp_path / "uv-calls.jsonl"
    monkeypatch.setenv("FAKE_UV_OUT", str(out))
    monkeypatch.setenv("FAKE_UV_VERSION", "9.0.0")
    monkeypatch.delenv("FAKE_UV_EXIT", raising=False)
    monkeypatch.setattr(updates, "find_uv", lambda: [sys.executable, str(FAKE_UV)])
    monkeypatch.setattr(updates, "find_git", lambda: True)
    monkeypatch.setattr(updates, "launcher_command", lambda install, uv: [sys.executable, str(FAKE_UV), "--launcher"])
    monkeypatch.setattr(updates, "detect_install", lambda engine_unit=False: dict(STUDIO_TOOL))
    monkeypatch.setattr(paths, "IS_WIN", False)
    with FakeGitHub([release("v9.0.0", "Notes")]) as gh:
        u = make_updates(tmp_path, monkeypatch, gh)
        u.check()
        restarts: list[Any] = []
        u.on_restart = restarts.append
        yield {"u": u, "out": out, "restarts": restarts, "tmp": tmp_path}


def wait_run(u: updates.Updates, want: str = "") -> dict[str, Any]:
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        if u.run["state"] not in ("running", "idle") and (not want or u.run["state"] == want):
            return u.run
        time.sleep(0.05)
    raise AssertionError(f"the update did not finish: {u.run}")


def test_one_click_update_installs_then_restarts_with_the_same_command_line(run_env: dict[str, Any]) -> None:
    u: updates.Updates = run_env["u"]
    u.apply("v9.0.0")
    assert wait_run(u)["state"] == "restarting"
    call = json.loads(run_env["out"].read_text(encoding="utf-8").splitlines()[0])
    assert call == [
        "tool",
        "install",
        "--python",
        "3.12",
        "--force",
        "git+https://github.com/sirfyyn/fle-studio@v9.0.0",
    ]
    assert run_env["restarts"] == [updates.restart_command()]
    assert run_env["restarts"][0][-1] == "--no-browser"
    assert "restarting" in (run_env["tmp"] / "update.log").read_text(encoding="utf-8")


def test_a_failed_install_leaves_the_old_version_running(
    run_env: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("FAKE_UV_EXIT", "3")
    u: updates.Updates = run_env["u"]
    u.apply("v9.0.0")
    run = wait_run(u)
    assert run["state"] == "failed" and "old version keeps running" in run["message"] and "code 3" in run["message"]
    assert any("failed to fetch" in line for line in run["tail"]) and run["log"].endswith("update.log")
    assert run_env["restarts"] == []  # nothing was restarted
    assert u.view()["run"]["state"] == "failed"
    u.ack()
    assert u.view()["run"]["state"] == "idle"


def test_a_new_version_that_does_not_report_itself_is_a_failure(
    run_env: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("FAKE_UV_VERSION", "0.0.1")
    u: updates.Updates = run_env["u"]
    u.apply("v9.0.0")
    run = wait_run(u)
    assert run["state"] == "failed" and "does not start as v9.0.0" in run["message"] and run_env["restarts"] == []


def test_apply_refuses_what_the_check_did_not_find(run_env: dict[str, Any]) -> None:
    u: updates.Updates = run_env["u"]
    for tag in ("v9.0.1", "v0.0.1", "main", "", "v9.0.0; echo hi"):
        with pytest.raises(updates.UpdateError):
            u.apply(tag)
    assert not run_env["out"].exists()


def test_apply_refuses_installs_that_cannot_update_themselves(
    run_env: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    u: updates.Updates = run_env["u"]
    cases = [
        ({**STUDIO_TOOL, "service": True}, "service"),
        (STUDIO_PIP, "pip"),
        ({**STUDIO_TOOL, "source": "folder"}, "folder"),
    ]
    for install, word in cases:
        monkeypatch.setattr(updates, "detect_install", lambda engine_unit=False, i=install: dict(i))
        with pytest.raises(updates.UpdateError, match=word):
            u.apply("v9.0.0")
    monkeypatch.setattr(updates, "detect_install", lambda engine_unit=False: dict(STUDIO_TOOL))
    monkeypatch.setattr(updates, "find_uv", lambda: None)
    with pytest.raises(updates.UpdateError, match="uv"):
        u.apply("v9.0.0")
    assert not run_env["out"].exists() and u.run["state"] == "idle"


def test_the_view_says_what_to_do_instead(run_env: dict[str, Any], monkeypatch: pytest.MonkeyPatch) -> None:
    u: updates.Updates = run_env["u"]
    assert u.view()["app"]["can_auto"] is True
    monkeypatch.setattr(updates, "detect_install", lambda engine_unit=False: dict(STUDIO_PIP))
    app = u.view()["app"]
    assert app["can_auto"] is False and app["command"].endswith('@v9.0.0"') and app["reason"]


def test_two_updates_at_once_are_refused(run_env: dict[str, Any]) -> None:
    u: updates.Updates = run_env["u"]
    u.run = {"state": "running", "tag": "v9.0.0", "message": "", "tail": []}
    with pytest.raises(updates.UpdateError, match="already"):
        u.apply("v9.0.0")


def test_windows_gets_a_helper_instead_of_an_install_from_inside(
    run_env: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """The program file is locked on Windows: the update runs from a script after FLE Studio has ended."""
    started: list[Any] = []

    class FakePopen:
        def __init__(self, cmd: list[str], **kw: Any) -> None:
            started.append((cmd, kw))

    monkeypatch.setattr(paths, "IS_WIN", True)
    monkeypatch.setattr(updates.subprocess, "Popen", FakePopen)
    monkeypatch.setattr(sys, "argv", ["fle-studio", "--port", "9000"])
    u: updates.Updates = run_env["u"]
    u.apply("v9.0.0")
    assert u.run["state"] == "restarting" and run_env["restarts"] == [None]  # just end: the helper starts it again
    assert not run_env["out"].exists()  # nothing was installed from inside the running program
    # Popen is patched process-wide: a sampler thread (vm_stat on macOS) may land here too
    cmd = next(c for c, _ in started if c and c[0] == "powershell.exe")
    assert "-File" in cmd and cmd[-2] == "-Job"
    job = json.loads(Path(cmd[-1]).read_text(encoding="utf-8"))
    assert job["spec"] == "git+https://github.com/sirfyyn/fle-studio@v9.0.0" and job["tag"] == "v9.0.0"
    assert job["start"][-2:] == ["9000", "--no-browser"] and job["uv"][0] == sys.executable
    script = Path(cmd[cmd.index("-File") + 1]).read_text(encoding="utf-8")
    assert "Wait-Process" in script and "tool install" in script and "Start-Process" in script


def test_the_helper_script_is_part_of_the_package() -> None:
    text = updates.windows_helper_text()
    assert text.startswith("# Written out by FLE Studio") and "param([string]$Job)" in text


def test_result_of_a_windows_update_is_shown_once(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    u = make_updates(tmp_path, monkeypatch, None)
    assert u.view()["result"] is None
    (tmp_path / "update-result.json").write_text(
        "\ufeff" + json.dumps({"status": "failed", "tag": "v9.0.0", "code": 1}), encoding="utf-8"
    )
    assert u.view()["result"]["status"] == "failed"
    u.ack()
    assert u.view()["result"] is None

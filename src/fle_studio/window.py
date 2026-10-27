"""`fle-studio open`: open FLE Studio in its own window, and make sure the Studio behind it is the installed version.

This is what the Dock starter runs (see dock_starter.py). The steps:

1. Ask the server on the Studio port which version it runs.
2. Not running, or not the installed version: restart the service that runs Studio (launchd, systemd user unit) and
   wait for the right version. Without a service, a Studio that is not running is started in the background; one that
   was started by hand and is outdated cannot be restarted from here, which the window then says in the log.
3. Open the address in Chrome's app mode with a profile of its own (`<data folder>/chrome`), so no cache or sign-in of
   another site can leave an old page behind. Without Chrome: the default browser.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import webbrowser
from collections.abc import Callable, Sequence
from pathlib import Path

from . import __version__, paths, selfupdate
from .selfupdate import Host, Problem

WAIT = 30  # seconds for a restarted Studio to answer


def chrome_path(host: Host) -> str:
    """Chrome's program, or "" when it is not installed."""
    if host.platform == "darwin":
        app = Path("/Applications/Google Chrome.app/Contents/MacOS/Google Chrome")
        return str(app) if app.is_file() else ""
    for name in ("google-chrome", "google-chrome-stable", "chromium", "chromium-browser"):
        found = host.which(name)
        if found:
            return found
    return ""


def chrome_command(chrome: str, url: str, profile: Path) -> list[str]:
    return [chrome, f"--app={url}", f"--user-data-dir={profile}", "--no-first-run", "--no-default-browser-check"]


def detached(command: Sequence[str]) -> None:
    """Start a program that goes on living when this one ends."""
    subprocess.Popen(
        list(command),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )


def ensure_current_studio(host: Host, port: int, say: Callable[[str], None]) -> None:
    """Make the Studio on `port` the installed version where that can be done; say what could not be done."""
    running = host.ping(port)
    if running is not None and selfupdate.same_version(running, __version__):
        return
    services = [s for s in selfupdate.find_services(host, port) if s.port == port]
    if services:
        say(f"Studio on port {port} runs {running or 'nothing'}, installed is {__version__}: restarting the service.")
        for service in services:
            restarted, sentence = selfupdate.restart_service(host, service)
            say(sentence)
            if restarted:
                seen = selfupdate.wait_for_version(host, port, __version__, WAIT)
                say(f"Studio now answers with {seen}." if seen else f"Nothing answers on port {port} after {WAIT} s.")
        return
    if running is None:
        say("Studio is not running and no service starts it: starting it in the background.")
        detached([sys.executable, "-m", "fle_studio", "--no-browser"])
        selfupdate.wait_for_version(host, port, __version__, WAIT)
        return
    say(
        f"Studio on port {port} runs {running}, installed is {__version__}, and it was started by hand: "
        "stop it (Ctrl+C) and start fle-studio again."
    )


def open_window(host: Host, say: Callable[[str], None], dry_run: bool = False) -> int:
    port = selfupdate.configured_port()
    url = f"http://127.0.0.1:{port}/"
    if dry_run:
        say(f"Would check the Studio on port {port} against {__version__}, then open {url}")
    else:
        try:
            ensure_current_studio(host, port, say)
        except Problem as problem:
            say(problem.text())  # still open the window: the page explains more than a closed app
    chrome = chrome_path(host)
    if not chrome:
        say(f"No Chrome: opening {url} in the default browser.")
        if not dry_run:
            webbrowser.open(url)
        return 0
    command = chrome_command(chrome, url, paths.data_dir() / "chrome")
    say(("Would start: " if dry_run else "Starting: ") + " ".join(command))
    if not dry_run:
        detached(command)
    return 0


def cli(argv: Sequence[str], host: Host | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="fle-studio open",
        description="Open FLE Studio in its own window; restart a service that runs an older version first.",
    )
    ap.add_argument("--dry-run", action="store_true", help="say what would happen and do nothing")
    ap.add_argument("--home", help="data folder (as for fle-studio itself)")
    args = ap.parse_args(list(argv))
    if args.home:
        os.environ["FLE_STUDIO_HOME"] = str(Path(args.home).expanduser())
    return open_window(host or Host(), print, args.dry_run)

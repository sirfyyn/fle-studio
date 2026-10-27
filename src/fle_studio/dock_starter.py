"""The Dock starter on macOS: "FLE Studio Starter.app", a small app that opens FLE Studio in its own window.

The app's program is a three-line shim that runs `fle-studio open`. All the logic (is the running service the
version that is installed? restart it if not; open the window with its own browser profile) lives in the installed
package (window.py), so it is always the logic of the installed version and the app never goes stale again.

The shim only changes when this file changes. `fle-studio update` writes it into an existing app and signs the app
again (ad hoc, `codesign --sign -`), because changing a file inside a signed app breaks its signature. An app that
is already current is not touched, so it is not signed again either.
"""

from __future__ import annotations

import stat
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .selfupdate import Host

APP_NAME = "FLE Studio Starter.app"


def shim_text(launcher: str, port: int) -> str:
    """The program of the app. `launcher` is the full path of `fle-studio`, `port` the fallback address."""
    return f"""#!/bin/bash
# FLE Studio Starter. Written by `fle-studio update`; the work is done by `fle-studio open`.
LOG="$HOME/Library/Logs/fle-studio-starter.log"
exec >>"$LOG" 2>&1
echo "$(date '+%F %T') start"
LAUNCHER={_quote(launcher)}
if [ -x "$LAUNCHER" ]; then
  exec "$LAUNCHER" open
fi
echo "fle-studio is not at $LAUNCHER: opening the address in the default browser"
open "http://127.0.0.1:{port}/"
"""


def _quote(text: str) -> str:
    return "'" + text.replace("'", "'\\''") + "'"


def refresh(host: Host, launcher: str, port: int) -> list[str]:
    """Bring the existing starter app(s) up to date and sign them. Returns lines to show; never raises."""
    lines: list[str] = []
    for folder in host.apps_dirs:
        app = Path(folder) / APP_NAME
        program = app / "Contents" / "MacOS" / "starter"
        if not program.is_file():
            continue
        text = shim_text(launcher, port)
        try:
            if program.read_text(encoding="utf-8") == text:
                lines.append(f"Dock starter is current: {app}")
                continue
            program.write_text(text, encoding="utf-8")
            program.chmod(program.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
        except OSError as exc:
            lines.append(
                f"Could not write the Dock starter {app} ({exc}). Run fle-studio update again, or delete the app."
            )
            continue
        sign = host.run(["codesign", "--force", "--sign", "-", str(app)], False)
        check = host.run(["codesign", "--verify", "--deep", "--strict", str(app)], False)
        if sign.code == 0 and check.code == 0:
            lines.append(f"Refreshed the Dock starter and signed it again (ad hoc): {app}")
        else:
            detail = (sign.out + check.out).strip()[:200]
            lines.append(
                f"Refreshed the Dock starter {app}, but its signature is not valid ({detail}). "
                f"Sign it by hand: codesign --force --sign - '{app}'"
            )
    return lines

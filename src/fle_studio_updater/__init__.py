"""Updates: the daily look for a newer FLE Studio and FLE engine, and the one-click update.

A plugin that ships with FLE Studio (entry point `updater`). It adds the /api/update routes, the notice above the
engine card and the Updates card in Settings. The update ends the process and runs it again; the launcher
(__main__.py) says how with `studio.restart_process`.
"""

from __future__ import annotations

from importlib import resources
from typing import Any


def register(ctx: Any) -> None:
    from fle_studio.server import ApiError  # noqa: PLC0415 - loaded when the plugin starts, after the server

    from . import updates  # noqa: PLC0415

    studio = ctx.studio
    service = updates.service(studio)
    service.on_restart = lambda argv: studio.restart_process(argv) if studio.restart_process else None
    updates.install(ctx.api, ApiError)
    ctx.on_stop(updates.start_background(studio))
    ctx.describe("Updates", "Tells you when a new FLE Studio or engine is out, and updates in one click.")
    ctx.web(resources.files(__package__) / "web")
    ctx.boot("boot.js")

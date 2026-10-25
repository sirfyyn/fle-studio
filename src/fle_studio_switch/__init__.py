"""The model switcher: a plugin that ships with FLE Studio (entry point `switch` of the group `fle_studio.plugins`).

It adds the entries of a model switcher next to an attached service to the model menu of the chat (see switcher.py
and the setting `switch_url`). Without the plugin the model menu of a service has only the link to the Engine page.
"""

from __future__ import annotations

from importlib import resources
from typing import Any

from .switcher import Switch


def register(ctx: Any) -> None:
    studio = ctx.studio
    switch = Switch(lambda: studio.config.get("switch_url"))

    def status(handler: Any, query: dict[str, str]) -> None:
        handler.send_json(200, switch.status())

    def start(handler: Any, query: dict[str, str]) -> None:
        body = handler.read_json()
        try:
            handler.send_json(200, switch.start(str(body.get("target") or "")))
        except ValueError as e:
            handler.send_json(400, {"error": {"message": str(e)}})
        except RuntimeError as e:
            handler.send_json(409, {"error": {"message": str(e)}})

    ctx.describe("Model switcher", "Switch the model of an attached service from the model menu (setting switch_url).")
    ctx.api("GET", r"^/api/engine/switch$", status)
    ctx.api("POST", r"^/api/engine/switch$", start)
    ctx.web(resources.files(__package__) / "web")
    ctx.boot("boot.js")

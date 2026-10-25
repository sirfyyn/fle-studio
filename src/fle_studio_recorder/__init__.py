"""The recorder: what goes through the /v1 address, kept for you. A plugin of FLE Studio's package (entry `recorder`).

Two things share one hook into the proxy (hook `proxy_record`, see plugins.py):

capture    plain files on this computer, one line of JSON per answer (off until you switch it on in Settings)
livechat   the same answers as a conversation that only lives in memory, shown on the Live page

An answer that streams is put back together once, by the recorder of the request, and both get the result. The
Live page, its sidebar entry and the Capture card in Settings come with the plugin.
"""

from __future__ import annotations

from importlib import resources
from typing import Any

from .capture import Capture
from .livechat import LiveChat


def register(ctx: Any) -> None:
    from fle_studio.server import ApiError  # noqa: PLC0415 - loaded when the plugin starts, after the server

    studio = ctx.studio
    live = LiveChat()
    capture = Capture(
        studio.data_dir,
        lambda: studio.config.get("capture.enabled"),
        lambda: studio.config.get("capture.max_mb"),
        live,
        studio.live.watched,
    )

    def live_chat(h: Any, query: dict[str, str]) -> None:
        h.send_json(200, live.summary() if query.get("summary") else live.snapshot())

    def live_chat_clear(h: Any, query: dict[str, str]) -> None:
        h.send_json(200, {"deleted": live.clear()})

    def live_chat_delete(h: Any, query: dict[str, str], id: str) -> None:  # noqa: A002
        if not live.delete(id):
            raise ApiError(404, "That conversation is already gone")
        h.send_json(200, {"ok": True})

    def live_chat_star(h: Any, query: dict[str, str], id: str) -> None:  # noqa: A002
        """Copy a Live conversation (up to one message, if the body names it) into the chat archive as a favorite."""
        message = h.read_json().get("message")
        try:
            data = live.star_data(id, str(message) if message else None)
        except KeyError:
            raise ApiError(404, "That conversation or message is already gone") from None
        saved = studio.chats.save(None, data)
        live.mark_starred(id)
        h.send_json(200, saved)

    def capture_status(h: Any, query: dict[str, str]) -> None:
        h.send_json(200, capture.status())

    def capture_clear(h: Any, query: dict[str, str]) -> None:
        if h.read_json().get("confirm") is not True:
            raise ApiError(400, 'Send {"confirm": true} to delete all captured files')
        h.send_json(200, {"deleted": capture.clear()})

    ident = r"(?P<id>[0-9a-f]{12})"
    for method, pattern, fn in (
        ("GET", r"^/api/livechat$", live_chat),
        ("DELETE", r"^/api/livechat$", live_chat_clear),
        ("DELETE", rf"^/api/livechat/{ident}$", live_chat_delete),
        ("POST", rf"^/api/livechat/{ident}/star$", live_chat_star),
        ("GET", r"^/api/capture$", capture_status),
        ("POST", r"^/api/capture/clear$", capture_clear),
    ):
        ctx.api(method, pattern, fn)

    studio.recorder = capture  # for tests and for other plugins: .live is the in-memory conversations
    ctx.hook("proxy_record", capture.begin)
    ctx.on_stop(lambda: setattr(studio, "recorder", None))
    ctx.describe("Recorder", "The Live page (answers as a conversation, in memory) and Capture (answers as files).")
    ctx.page("Live", icon="live", script="live.js", path="/live", nav=False)  # linked from the top of the chat list
    ctx.web(resources.files(__package__) / "web")
    ctx.boot("boot.js")

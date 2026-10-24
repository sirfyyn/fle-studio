"""Shared test setup."""

from __future__ import annotations

import socketserver
from http.server import HTTPServer


def _quick_server_bind(self: HTTPServer) -> None:
    # HTTPServer.server_bind calls socket.getfqdn(), a reverse lookup for a name no test uses. On some CI machines
    # (macOS runners) it takes many seconds per server.
    socketserver.TCPServer.server_bind(self)
    self.server_name, self.server_port = str(self.server_address[0]), self.server_address[1]


HTTPServer.server_bind = _quick_server_bind  # type: ignore[method-assign]

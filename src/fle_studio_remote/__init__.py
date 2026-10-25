"""Remote access: one FLE Studio computes for another, and the TLS that goes with it.

A plugin that ships with FLE Studio (entry point `remote`).

host side   TLS with a certificate of its own, pairing codes, clients with tokens and rights (`remote.py`, `tlsgen.py`)
client side hosts added by address or SSH, everything under /h/<host id>/ passed on to the host, the host chooser in the
            sidebar and the Remote access cards in Settings (`web/boot.js`)

What the core gives it: the hooks `request_auth`, `request_open`, `request_enforce`, `forward`, `tls_wrap` and `cli`
(plugins.py), and a transport hook in the page's api.js. Without the plugin FLE Studio answers on the address it is told
to, with the access key and without TLS, and it has no hosts.
"""

from __future__ import annotations

import socket
from importlib import resources
from typing import Any


def register(ctx: Any) -> None:
    from . import remote, tlsgen  # noqa: PLC0415

    studio = ctx.studio

    def tls_wrap(studio: Any, sock: socket.socket, host: str) -> tuple[socket.socket, str]:
        cert, key, fingerprint = tlsgen.ensure(studio.data_dir, remote.host_name(studio))
        # The handshake happens in the thread of the connection, so a slow client cannot hold up the others.
        wrapped = tlsgen.server_context(cert, key).wrap_socket(sock, server_side=True, do_handshake_on_connect=False)
        return wrapped, fingerprint

    def forward(handler: Any, url: Any) -> bool:
        remote.forward(handler, url)
        return True

    def cli(event: str, studio: Any, args: Any, **info: Any) -> bool:
        if event == "pair_code":  # `fle-studio --pair-code`: a code for the data folder, which the running server reads
            made = remote.service(studio).clients.new_code()
            print(f"Pairing code (10 minutes, one use): {made['pretty']}")
            if studio.config.get("server.tls") and not info.get("loopback"):
                _, _, fp = tlsgen.ensure(studio.data_dir, remote.host_name(studio))
                print(f"Certificate fingerprint (SHA-256): {tlsgen.pretty(fp)}")
            print(f"In the client: Add host…, address {socket.gethostname()}:{info.get('port')}, this code.")
            return True
        if event == "started":
            if studio.listen.get("tls"):
                print(f"Host mode: TLS, certificate fingerprint (SHA-256) {tlsgen.pretty(studio.tls_fingerprint)}")
                print("Pair a client with a code from Settings → Remote access, or run: fle-studio --pair-code")
            if getattr(args, "pair", False):
                made = remote.service(studio).clients.new_code()
                print(f"Pairing code (10 minutes, one use): {made['pretty']}")
        return False

    remote.install(ctx.api)
    ctx.hook("request_auth", remote.authenticate)
    ctx.hook("request_open", lambda path: path == "/api/remote/pair")
    ctx.hook("request_enforce", remote.enforce)
    ctx.hook("forward", forward)
    ctx.hook("tls_wrap", tls_wrap)
    ctx.hook("cli", cli)
    ctx.on_stop(lambda: remote.shutdown(studio))  # ends the ssh tunnels this Studio started
    ctx.describe("Remote access", "Use another computer's FLE Studio as a host, or be a host with TLS and pairing.")
    ctx.web(resources.files(__package__) / "web")
    ctx.boot("boot.js", early=True)

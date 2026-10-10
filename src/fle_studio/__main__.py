"""`fle-studio`: start the local server and open the app in the browser."""

from __future__ import annotations

import argparse
import json
import os
import secrets
import signal
import sys
import threading
import urllib.request
import webbrowser
from pathlib import Path

from . import __version__


def _running(port: int) -> bool:
    """Is an FLE Studio already answering on this port?"""
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/api/ping", timeout=1.5) as r:
            return json.loads(r.read().decode()).get("app") == "fle-studio"
    except (OSError, ValueError):
        return False


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="fle-studio", description="Local launcher for the FLE engine and llama.cpp: "
                                 "chat, live inference data, model downloads and launch profiles.")
    ap.add_argument("--host", help="address to listen on (default 127.0.0.1; 0.0.0.0 opens it to the network)")
    ap.add_argument("--port", type=int, help="port (default 8642)")
    ap.add_argument("--no-browser", action="store_true", help="do not open the browser")
    ap.add_argument("--home", help="data folder (settings, chats, engines, models); default: per-user app data")
    ap.add_argument("--attach", metavar="URL", help="use an already running llama.cpp / FLE server")
    ap.add_argument("--version", action="version", version=f"FLE Studio {__version__}")
    args = ap.parse_args(argv)
    if args.home:
        os.environ["FLE_STUDIO_HOME"] = str(Path(args.home).expanduser())

    from .server import Studio, make_server  # noqa: PLC0415 - after FLE_STUDIO_HOME is set

    studio = Studio()
    cfg = studio.config
    host = args.host or cfg.get("server.host")
    port = args.port or int(cfg.get("server.port"))
    local_host = host in ("127.0.0.1", "localhost", "::1")
    if local_host and _running(port):
        url = f"http://127.0.0.1:{port}/"
        print(f"FLE Studio is already running: {url}")
        if not args.no_browser:
            webbrowser.open(url)
        return 0
    if not local_host and not cfg.get("access_key"):
        cfg.update({"access_key": secrets.token_urlsafe(12)})
        print("FLE Studio is open to the network. Access key for other machines (also in Settings → Network):")
        print(f"  {cfg.get('access_key')}")
    try:
        server = make_server(studio, host, port)
    except OSError as exc:
        print(f"FLE Studio could not start its web server: {exc}", file=sys.stderr)
        return 1
    shown = "127.0.0.1" if host in ("0.0.0.0", "::") or local_host else host
    url = f"http://{shown}:{server.server_address[1]}/"
    print(f"FLE Studio {__version__} — {url}")
    print(f"Data folder: {studio.data_dir}")
    print("Press Ctrl+C to stop (this also stops a model FLE Studio started).")

    if args.attach:
        try:
            studio.engine.attach(args.attach, cfg.get("attach_key"))
        except Exception as exc:  # startup must go on; the page shows the state
            print(f"Could not attach to {args.attach}: {exc}", file=sys.stderr)
    elif cfg.get("launch.autostart") and cfg.get("last_model"):
        def autostart() -> None:
            try:
                plan = studio.make_plan(cfg.get("last_model"), cfg.get("launch.profile"))
                studio.engine.start(plan)
                print(f"Loading {plan.model.get('name')} …")
            except Exception as exc:  # report and keep serving
                print(f"Autostart failed: {getattr(exc, 'message', exc)}", file=sys.stderr)

        threading.Thread(target=autostart, daemon=True).start()

    def stop(signum: int, _frame: object) -> None:
        threading.Thread(target=server.shutdown, daemon=True).start()

    signal.signal(signal.SIGTERM, stop)
    if hasattr(signal, "SIGBREAK"):
        signal.signal(signal.SIGBREAK, stop)  # Ctrl+Break and closing the console on Windows

    if cfg.get("server.open_browser") and not args.no_browser:
        threading.Timer(0.8, webbrowser.open, args=(url,)).start()
    try:
        server.serve_forever(poll_interval=0.5)
    except KeyboardInterrupt:
        pass
    finally:
        print("Stopping FLE Studio …")
        studio.engine.shutdown()
        server.server_close()
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""A stand-in for `ssh -N -L LOCAL:HOST:PORT target`: it really forwards TCP, so tests can go through a "tunnel".

    FLE_STUDIO_SSH='["python", "fake_ssh.py"]'

Environment:
  FAKE_SSH_LOG   file that gets the arguments of each start, one JSON list per line
  FAKE_SSH_FAIL  when set: print this text to stderr and exit with code 255 (a refused login)
"""

from __future__ import annotations

import json
import os
import socket
import sys
import threading


def pipe(a: socket.socket, b: socket.socket) -> None:
    try:
        while True:
            data = a.recv(65536)
            if not data:
                break
            b.sendall(data)
    except OSError:
        pass
    finally:
        for s in (a, b):
            try:
                s.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            s.close()


def main(argv: list[str]) -> int:
    log = os.environ.get("FAKE_SSH_LOG")
    if log:
        with open(log, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(argv) + "\n")
    fail = os.environ.get("FAKE_SSH_FAIL")
    if fail:
        print(fail, file=sys.stderr, flush=True)
        return 255
    spec = argv[argv.index("-L") + 1]
    bind, lport, rhost, rport = spec.split(":")
    server = socket.socket()
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind((bind, int(lport)))
    server.listen(16)
    while True:
        conn, _ = server.accept()
        try:
            upstream = socket.create_connection((rhost, int(rport)), timeout=5)
        except OSError:
            conn.close()
            continue
        threading.Thread(target=pipe, args=(conn, upstream), daemon=True).start()
        threading.Thread(target=pipe, args=(upstream, conn), daemon=True).start()


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

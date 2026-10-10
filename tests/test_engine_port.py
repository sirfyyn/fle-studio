"""FLE Studio refuses to start an engine on a port where a server already listens."""

import socket
from pathlib import Path

import pytest

from fle_studio.engine import Engine, EngineBusy, port_taken
from fle_studio.launch import Plan


def test_start_refuses_a_taken_port(tmp_path: Path) -> None:
    with socket.socket() as server:
        server.bind(("127.0.0.1", 0))
        server.listen()
        port = server.getsockname()[1]
        assert port_taken(port)
        engine = Engine(tmp_path)
        plan = Plan(engine="custom", binary="llama-server", args=[], env={}, port=port, model={})
        with pytest.raises(EngineBusy, match=f"Port {port} is already in use"):
            engine.start(plan)
        assert engine.state == "stopped"


def test_a_closed_port_is_free() -> None:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    assert not port_taken(port)

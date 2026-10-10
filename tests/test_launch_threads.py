"""The thread count of the automatic FLE command.

One thread per physical core is the default. On a processor with one thread per core that puts a compute thread on
every core, and the engine's own threads (GPU driver, expert-cache copies) have to wait for one; the FLE command then
keeps one core free. A number set by hand is always used as it is.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

from fle_studio.config import DEFAULTS
from fle_studio.launch import auto_plan
from fle_studio.system import GiB

MODEL = {"id": "m", "name": "M", "path": "/models/m.gguf",
         "info": {"context_length": 32768, "gpu_bytes": 2 * GiB, "expert_bytes": 20 * GiB,
                  "kv_bytes_per_token_q8": 1024, "moe": True}}
GPU = {"vendor": "nvidia", "index": 0, "name": "RTX", "memory_total": 16 * GiB, "memory_used": 0}


def plan(tmp_path: Path, *, physical: int, logical: int, threads: int = 0, kind: str = "fle") -> Any:
    settings = copy.deepcopy(DEFAULTS)
    settings["launch"]["threads"] = threads
    sysinfo = {"cpu": {"physical": physical, "logical": logical}, "memory": {"total": 64 * GiB}, "gpus": [GPU]}
    engine = {"kind": kind, "path": "/engines/llama-server", "backend": "cuda"}
    return auto_plan(model=MODEL, engine=engine, settings=settings, sysinfo=sysinfo, port=8080, data_dir=tmp_path)


def threads_of(p: Any) -> str | None:
    return p.args[p.args.index("-t") + 1] if "-t" in p.args else None


def test_fle_keeps_a_core_free_without_smt(tmp_path: Path) -> None:
    p = plan(tmp_path, physical=8, logical=8)
    assert threads_of(p) == "7"
    assert any("one core stays free" in n for n in p.notes)


def test_fle_uses_every_physical_core_with_smt(tmp_path: Path) -> None:
    p = plan(tmp_path, physical=16, logical=32)  # the second thread of each core stays free anyway
    assert threads_of(p) == "16"
    assert not any("one core stays free" in n for n in p.notes)


def test_a_number_set_by_hand_is_kept(tmp_path: Path) -> None:
    assert threads_of(plan(tmp_path, physical=8, logical=8, threads=8)) == "8"


def test_a_single_core_keeps_its_thread(tmp_path: Path) -> None:
    assert threads_of(plan(tmp_path, physical=1, logical=1)) == "1"


def test_upstream_engine_leaves_threads_to_llama_cpp(tmp_path: Path) -> None:
    assert threads_of(plan(tmp_path, physical=8, logical=8, kind="upstream")) is None
